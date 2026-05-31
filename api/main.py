"""FastAPI server for Temporal Index."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.config import CORS_ORIGINS
from api.generation import check_model_ready, llm_health_summary
from api.model_registry import resolve_model
from api.rag import milvus_health, run_rag_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("temporal_index.pipeline").setLevel(logging.INFO)

app = FastAPI(
    title="Temporal Index API",
    description="Hybrid retrieval + local LLM generation (Ollama + MLX) over indexed catalog and waitlist data.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceItem(BaseModel):
    id: str | None = None
    rank: int | None = None
    score: float | None = None
    chunk_type: str | None = None
    collection: str | None = None
    model: str | None = None
    references: str | None = None
    text: str | None = None
    retrieval_method: str | None = None
    source_path: str | None = None


class PipelineMetricsResponse(BaseModel):
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: int
    model: str | None = None
    backend: str | None = None
    metrics: PipelineMetricsResponse | None = None


@app.get("/api/health")
async def health() -> dict:
    milvus = milvus_health()
    llm = await llm_health_summary()

    healthy = milvus.get("collection_ready") and llm.get("any_backend_available")
    return {
        "status": "ok" if healthy else "degraded",
        "milvus": milvus,
        "ollama": llm.get("ollama"),
        "mlx": llm.get("mlx"),
        "models": llm.get("models"),
    }


@app.get("/api/models")
async def list_models() -> dict:
    llm = await llm_health_summary()
    return {"models": llm["models"]}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    milvus = milvus_health()
    if not milvus.get("collection_ready"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Milvus hybrid index is not ready. Run `python -m embedding.hybrid` "
                f"to build `{milvus.get('collection')}`."
            ),
        )

    spec = resolve_model(request.model)
    model_status = await check_model_ready(spec)
    if not model_status.get("ready"):
        backend_hint = (
            f"Install mlx-lm and set MLX_MODEL_PATHS for `{spec.id}`."
            if spec.is_mlx
            else f"Start Ollama and run `ollama pull {spec.id}`."
        )
        detail = model_status.get("detail") or backend_hint
        raise HTTPException(status_code=503, detail=detail)

    try:
        result = await run_rag_chat(
            query=request.query.strip(),
            history=[message.model_dump() for message in request.history],
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            alpha=request.alpha,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {exc}") from exc

    return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
