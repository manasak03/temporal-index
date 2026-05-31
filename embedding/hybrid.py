"""Hybrid dense (Qwen3) + sparse (BM25/Tfidf) embedding engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from chunking.semantic import ChunkRecord, build_all_chunks
from paths import (
    DEFAULT_COLLECTION,
    DEFAULT_MILVUS_DB,
    DEFAULT_MODELS_CSV,
    DEFAULT_VECTORIZER,
    DEFAULT_WAITLIST_CSV,
)

DEFAULT_MODEL_NAME = os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_MRL_DIM = int(os.environ.get("QWEN_MRL_DIM", "1024"))
QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "
DEFAULT_VECTORIZER_PATH = DEFAULT_VECTORIZER


@dataclass
class EmbeddedRecord:
    id: str
    text: str
    dense_vector: list[float]
    sparse_vector: dict[int, float]
    metadata: dict[str, Any]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool embeddings from the final non-padding token (<|endoftext|> / EOS position)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def truncate_and_normalize_mrl(embeddings: Tensor, target_dim: int) -> Tensor:
    truncated = embeddings[:, :target_dim]
    return F.normalize(truncated, p=2, dim=1)


class HybridEmbedder:
    """Dense Qwen3 embeddings + local sparse TF-IDF vectors for hybrid retrieval."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        mrl_dim: int = DEFAULT_MRL_DIM,
        device: str | None = None,
        max_length: int = 8192,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.mrl_dim = mrl_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_float32 = self.device in {"cpu", "mps"}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32 if use_float32 else None,
        )
        self.model.to(self.device)
        if use_float32:
            self.model = self.model.float()
        self.model.eval()

        self.vectorizer: TfidfVectorizer | None = None

    def fit_sparse(self, corpus: Sequence[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[\w\-./]+\b",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.vectorizer.fit(corpus)

    def save_vectorizer(self, path: Path | str = DEFAULT_VECTORIZER_PATH) -> None:
        if self.vectorizer is None:
            raise RuntimeError("Sparse vectorizer is not fitted.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, output)

    def load_vectorizer(self, path: Path | str = DEFAULT_VECTORIZER_PATH) -> None:
        self.vectorizer = joblib.load(path)

    def _prepare_texts(self, texts: Sequence[str], *, is_query: bool) -> list[str]:
        if is_query:
            return [f"{QUERY_INSTRUCTION}{text}" for text in texts]
        return list(texts)

    @torch.no_grad()
    def embed_dense(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        prepared = self._prepare_texts(texts, is_query=is_query)
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings = truncate_and_normalize_mrl(embeddings, self.mrl_dim)
            all_embeddings.append(embeddings.float().cpu().numpy())

        return np.vstack(all_embeddings)

    def _ensure_vectorizer(self) -> TfidfVectorizer:
        if self.vectorizer is None:
            raise RuntimeError("Sparse vectorizer is not fitted. Call fit_sparse() during ingestion.")
        return self.vectorizer

    @staticmethod
    def _sparse_matrix_row_to_dict(row) -> dict[int, float]:
        coo = row.tocoo()
        return {int(col): float(value) for col, value in zip(coo.col, coo.data)}

    def embed_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        vectorizer = self._ensure_vectorizer()
        matrix = vectorizer.transform(texts)
        return [self._sparse_matrix_row_to_dict(matrix.getrow(i)) for i in range(matrix.shape[0])]

    def build_records(self, chunks: Iterable[ChunkRecord]) -> list[EmbeddedRecord]:
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        texts = [chunk.text for chunk in chunk_list]
        if self.vectorizer is None:
            self.fit_sparse(texts)

        dense_vectors = self.embed_dense(texts, is_query=False)
        sparse_vectors = self.embed_sparse(texts)

        records: list[EmbeddedRecord] = []
        for chunk, dense, sparse in zip(chunk_list, dense_vectors, sparse_vectors):
            records.append(
                EmbeddedRecord(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    dense_vector=dense.astype(float).tolist(),
                    sparse_vector=sparse,
                    metadata=chunk.metadata,
                )
            )
        return records

    def embed_query(self, query: str) -> tuple[np.ndarray, dict[int, float]]:
        dense = self.embed_dense([query], is_query=True)[0]
        sparse = self.embed_sparse([query])[0]
        return dense, sparse


def run_ingestion(
    *,
    models_path: Path | str = DEFAULT_MODELS_CSV,
    waitlist_path: Path | str = DEFAULT_WAITLIST_CSV,
    db_path: Path | str = DEFAULT_MILVUS_DB,
    collection_name: str = DEFAULT_COLLECTION,
    recreate_collection: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    mrl_dim: int = DEFAULT_MRL_DIM,
) -> int:
    """End-to-end ingestion: semantic chunk -> hybrid embed -> Milvus index."""
    from retrieval.hybrid import HybridSearcher

    chunks = build_all_chunks(models_path=models_path, waitlist_path=waitlist_path)
    print(f"Built {len(chunks)} semantic chunks.")

    embedder = HybridEmbedder(model_name=model_name, mrl_dim=mrl_dim)
    print("Embedding chunks (dense + sparse); this may take several minutes on CPU...")
    records = embedder.build_records(chunks)
    embedder.save_vectorizer()
    print(f"Embedded {len(records)} records.")

    print("Connecting to Milvus Lite and indexing...")
    try:
        searcher = HybridSearcher(
            db_path=db_path,
            collection_name=collection_name,
            dense_dim=mrl_dim,
            embedder=embedder,
        )
    except Exception as exc:
        if "Open local milvus failed" in str(exc) or "opened by another program" in str(exc):
            raise SystemExit(
                f"Could not open {db_path} — Milvus Lite allows only one process at a time. "
                "Stop the API server (uvicorn) and any other process using the DB, then retry."
            ) from exc
        raise
    searcher.create_collection(recreate=recreate_collection)
    inserted = searcher.insert_records(records)
    return inserted


def main() -> None:
    inserted = run_ingestion()
    print(f"Hybrid ingestion complete. Inserted {inserted} records into Milvus.")


if __name__ == "__main__":
    main()
