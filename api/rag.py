"""RAG orchestration: hybrid retrieval + local LLM generation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from pymilvus import MilvusClient

from api.config import (
    MILVUS_COLLECTION,
    MILVUS_DB_PATH,
    RETRIEVAL_ALPHA,
    RETRIEVAL_FUSION,
    RETRIEVAL_LIMIT,
)
from api.generation import generate_answer, stream_answer
from api.metrics import PipelineMetrics, log_pipeline_metrics
from retrieval.hybrid import HybridSearcher


@lru_cache(maxsize=1)
def get_searcher() -> HybridSearcher:
    return HybridSearcher(db_path=MILVUS_DB_PATH, collection_name=MILVUS_COLLECTION)


def milvus_health() -> dict[str, Any]:
    db_path = MILVUS_DB_PATH
    status: dict[str, Any] = {
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "collection": MILVUS_COLLECTION,
        "collection_ready": False,
    }
    if not db_path.exists():
        return status

    client = MilvusClient(str(db_path))
    status["collection_ready"] = client.has_collection(MILVUS_COLLECTION)
    return status


def format_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for source in sources:
        collection = source.get("collection")
        references = source.get("references")
        chunk_type = source.get("chunk_type")
        source_path = None
        if collection:
            source_path = f"data/chunks/{collection}"
            if references:
                source_path = f"{source_path}/refs/{references}"

        formatted.append(
            {
                "id": source.get("id"),
                "rank": source.get("rank"),
                "score": source.get("score"),
                "chunk_type": chunk_type,
                "collection": collection,
                "model": source.get("model"),
                "references": references,
                "text": source.get("text"),
                "retrieval_method": "hybrid",
                "source_path": source_path,
            }
        )
    return formatted


async def run_rag_chat(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    limit: int = RETRIEVAL_LIMIT,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    fusion_alpha = RETRIEVAL_ALPHA if alpha is None else alpha
    started = time.perf_counter()
    searcher = get_searcher()

    retrieval_started = time.perf_counter()
    context, sources = searcher.search(
        query,
        limit=limit,
        fusion=RETRIEVAL_FUSION,  # type: ignore[arg-type]
        alpha=fusion_alpha,
    )
    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

    answer, spec, generation = await generate_answer(
        query=query,
        context=context,
        history=history,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    total_ms = int((time.perf_counter() - started) * 1000)

    metrics = PipelineMetrics.build(
        retrieval_ms=retrieval_ms,
        generation=generation,
        total_ms=total_ms,
        chunks_retrieved=len(sources),
        context_chars=len(context),
        model=spec.id,
        backend=spec.backend,
        fusion_alpha=fusion_alpha,
        fusion_method=RETRIEVAL_FUSION,
    )
    log_pipeline_metrics(
        query,
        metrics,
        history_turns=len(history or []),
    )

    return {
        "answer": answer,
        "sources": format_sources(sources),
        "latency_ms": total_ms,
        "model": spec.id,
        "backend": spec.backend,
        "metrics": metrics.to_dict(),
    }


async def run_rag_chat_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    limit: int = RETRIEVAL_LIMIT,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    alpha: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    fusion_alpha = RETRIEVAL_ALPHA if alpha is None else alpha
    started = time.perf_counter()
    searcher = get_searcher()

    retrieval_started = time.perf_counter()
    context, sources = searcher.search(
        query,
        limit=limit,
        fusion=RETRIEVAL_FUSION,  # type: ignore[arg-type]
        alpha=fusion_alpha,
    )
    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
    formatted_sources = format_sources(sources)

    yield {
        "type": "sources",
        "sources": formatted_sources,
        "retrieval_ms": retrieval_ms,
    }

    answer_parts: list[str] = []
    spec = None
    generation = None

    async for delta, resolved_spec, metrics in stream_answer(
        query=query,
        context=context,
        history=history,
        model=model,
        temperature=temperature,
        top_p=top_p,
    ):
        spec = resolved_spec
        if delta:
            answer_parts.append(delta)
            yield {"type": "token", "delta": delta}
        if metrics is not None:
            generation = metrics

    if spec is None or generation is None:
        raise RuntimeError("Generation stream ended without metrics.")

    answer = "".join(answer_parts).strip()
    total_ms = int((time.perf_counter() - started) * 1000)
    pipeline_metrics = PipelineMetrics.build(
        retrieval_ms=retrieval_ms,
        generation=generation,
        total_ms=total_ms,
        chunks_retrieved=len(sources),
        context_chars=len(context),
        model=spec.id,
        backend=spec.backend,
        fusion_alpha=fusion_alpha,
        fusion_method=RETRIEVAL_FUSION,
    )
    log_pipeline_metrics(
        query,
        pipeline_metrics,
        history_turns=len(history or []),
    )

    yield {
        "type": "done",
        "answer": answer,
        "sources": formatted_sources,
        "latency_ms": total_ms,
        "model": spec.id,
        "backend": spec.backend,
        "metrics": pipeline_metrics.to_dict(),
    }
