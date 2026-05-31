"""Pipeline metrics collection and terminal logging for RAG requests."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("temporal_index.pipeline")


@dataclass
class GenerationMetrics:
    generation_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineMetrics:
    retrieval_ms: int
    generation_ms: int
    total_ms: int
    chunks_retrieved: int
    context_chars: int
    model: str
    backend: str
    fusion_alpha: float
    fusion_method: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        retrieval_ms: int,
        generation: GenerationMetrics,
        total_ms: int,
        chunks_retrieved: int,
        context_chars: int,
        model: str,
        backend: str,
        fusion_alpha: float,
        fusion_method: str,
    ) -> PipelineMetrics:
        return cls(
            retrieval_ms=retrieval_ms,
            generation_ms=generation.generation_ms,
            total_ms=total_ms,
            chunks_retrieved=chunks_retrieved,
            context_chars=context_chars,
            model=model,
            backend=backend,
            fusion_alpha=fusion_alpha,
            fusion_method=fusion_method,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
            tokens_per_second=generation.tokens_per_second,
        )


def _fmt_tokens(value: int | None) -> str:
    return str(value) if value is not None else "n/a"


def _fmt_tps(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def log_pipeline_metrics(
    query: str,
    metrics: PipelineMetrics,
    *,
    history_turns: int = 0,
) -> None:
    preview = query.replace("\n", " ").strip()
    if len(preview) > 96:
        preview = f"{preview[:93]}..."

    lines = [
        "",
        "═" * 62,
        " RAG Pipeline",
        "─" * 62,
        f" Query       : {preview!r}",
        f" Model       : {metrics.model} ({metrics.backend})",
        f" History     : {history_turns} prior turn(s)",
        "─" * 62,
        (
            f" Retrieval   : {metrics.retrieval_ms:>5} ms · "
            f"{metrics.chunks_retrieved} chunks · "
            f"{metrics.context_chars:,} ctx chars · "
            f"α={metrics.fusion_alpha} ({metrics.fusion_method})"
        ),
        (
            f" Generation  : {metrics.generation_ms:>5} ms · "
            f"prompt={_fmt_tokens(metrics.prompt_tokens)} · "
            f"completion={_fmt_tokens(metrics.completion_tokens)} · "
            f"total={_fmt_tokens(metrics.total_tokens)} tok"
        ),
        f" Throughput  : {_fmt_tps(metrics.tokens_per_second)} tok/s (completion)",
        f" Total       : {metrics.total_ms:>5} ms",
        "═" * 62,
    ]
    logger.info("\n".join(lines))
