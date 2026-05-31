"""Milvus Lite hybrid dense + sparse search with RRF / weighted fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker, WeightedRanker

from embedding.hybrid import (
    DEFAULT_MRL_DIM,
    DEFAULT_VECTORIZER_PATH,
    EmbeddedRecord,
    HybridEmbedder,
)
from paths import DEFAULT_COLLECTION, DEFAULT_MILVUS_DB

DEFAULT_DB_PATH = DEFAULT_MILVUS_DB
CONTEXT_SEPARATOR = "\n---\n"


@dataclass
class SearchHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class HybridSearcher:
    """Hybrid retrieval over dense Qwen3 vectors and sparse BM25-style vectors."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        dense_dim: int = DEFAULT_MRL_DIM,
        embedder: HybridEmbedder | None = None,
        vectorizer_path: Path | str = DEFAULT_VECTORIZER_PATH,
    ) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.client = MilvusClient(self.db_path)
        self.embedder = embedder or HybridEmbedder(mrl_dim=dense_dim)
        if self.embedder.vectorizer is None:
            self.embedder.load_vectorizer(vectorizer_path)

    def create_collection(self, *, recreate: bool = False) -> None:
        if recreate and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field("sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", datatype=DataType.VARCHAR, max_length=65535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_ivf_flat",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_records(self, records: list[EmbeddedRecord]) -> int:
        if not records:
            return 0

        payload: list[dict[str, Any]] = []
        for record in records:
            row = {
                "id": record.id,
                "dense_vector": record.dense_vector,
                "sparse_vector": record.sparse_vector,
                "text": record.text,
            }
            for key, value in record.metadata.items():
                if isinstance(value, list):
                    row[key] = ", ".join(str(item) for item in value)
                else:
                    row[key] = value
            payload.append(row)

        self.client.insert(collection_name=self.collection_name, data=payload)
        return len(payload)

    def _build_ranker(
        self,
        fusion: Literal["rrf", "weighted"],
        alpha: float,
    ):
        if fusion == "weighted":
            sparse_weight = 1.0 - alpha
            return WeightedRanker(alpha, sparse_weight)
        return RRFRanker()

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        fusion: Literal["rrf", "weighted"] = "rrf",
        alpha: float = 0.7,
        dense_nprobe: int = 16,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Run hybrid search and return:
          1) a formatted context string block
          2) a list of source metadata dictionaries
        """
        if not self.client.has_collection(self.collection_name):
            raise RuntimeError(
                f"Collection `{self.collection_name}` does not exist. Run hybrid ingestion first."
            )

        dense_query, sparse_query = self.embedder.embed_query(query)

        dense_request = AnnSearchRequest(
            data=[dense_query.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": dense_nprobe}},
            limit=limit,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=limit,
        )

        ranker = self._build_ranker(fusion=fusion, alpha=alpha)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=["text", "chunk_type", "collection", "model", "references"],
        )

        if not results:
            return "", []

        context_blocks: list[str] = []
        sources: list[dict[str, Any]] = []

        for rank, hit in enumerate(results[0], start=1):
            entity = hit.get("entity", {})
            text = entity.get("text", "")
            metadata = {
                "id": hit.get("id"),
                "rank": rank,
                "score": float(hit.get("distance", 0.0)),
                "chunk_type": entity.get("chunk_type"),
                "collection": entity.get("collection"),
                "model": entity.get("model"),
                "references": entity.get("references"),
                "text": text,
            }
            sources.append(metadata)
            context_blocks.append(
                "\n".join(
                    [
                        f"Rank: {rank}",
                        f"Score: {metadata['score']:.6f}",
                        f"Chunk Type: {metadata.get('chunk_type') or 'unknown'}",
                        f"Collection: {metadata.get('collection') or 'n/a'}",
                        f"Model: {metadata.get('model') or 'n/a'}",
                        f"References: {metadata.get('references') or 'n/a'}",
                        "",
                        text,
                    ]
                )
            )

        return CONTEXT_SEPARATOR.join(context_blocks), sources


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Query the hybrid Milvus index.")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--alpha", type=float, default=0.7, help="Dense weight for weighted fusion")
    args = parser.parse_args()

    searcher = HybridSearcher()
    context, sources = searcher.search(
        args.query,
        limit=args.limit,
        fusion=args.fusion,
        alpha=args.alpha,
    )
    print(context)
    print("\nSources:")
    for source in sources:
        print(source)


if __name__ == "__main__":
    main()
