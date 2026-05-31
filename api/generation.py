"""Route generation to Ollama or MLX based on model selection."""

from __future__ import annotations

from typing import Any

from api.llm import LLMClient
from api.metrics import GenerationMetrics
from api.mlx_llm import MLXLLMClient, mlx_runtime_available
from api.model_registry import ModelSpec, list_catalog, resolve_model


def get_ollama_client() -> LLMClient:
    return LLMClient()


def get_mlx_client() -> MLXLLMClient:
    return MLXLLMClient()


def _ollama_has_model(model_id: str, available: list[str]) -> bool:
    if model_id in available:
        return True
    base = model_id.split(":")[0]
    return any(
        name == model_id
        or name.split(":")[0] == base
        or (name or "").startswith(f"{model_id}:")
        or (name or "").startswith(f"{base}:")
        for name in available
    )


async def check_model_ready(spec: ModelSpec) -> dict[str, Any]:
    if spec.is_mlx:
        mlx_health = await get_mlx_client().health()
        ready = mlx_health.get("available", False)
        detail = None
        if ready:
            detail = f"mlx-lm · {spec.mlx_path or 'path not set'}"
        else:
            detail = mlx_health.get("error")
        return {
            "id": spec.id,
            "backend": spec.backend,
            "ready": ready,
            "detail": detail,
            "mlx_path": spec.mlx_path,
        }

    client = get_ollama_client()
    try:
        ollama = await client.health()
        available = ollama.get("available_models") or []
        ready = _ollama_has_model(spec.id, available)
        return {
            "id": spec.id,
            "backend": spec.backend,
            "ready": ready,
            "detail": None if ready else f"Run: ollama pull {spec.id}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": spec.id,
            "backend": spec.backend,
            "ready": False,
            "detail": str(exc),
        }


async def list_models_status() -> list[dict[str, Any]]:
    statuses = []
    for spec in list_catalog():
        statuses.append(await check_model_ready(spec))
    return statuses


async def llm_health_summary() -> dict[str, Any]:
    ollama_status: dict[str, Any]
    try:
        ollama_status = await get_ollama_client().health()
        ollama_status["available"] = ollama_status.get("reachable", False)
    except Exception as exc:  # noqa: BLE001
        ollama_status = {"available": False, "reachable": False, "error": str(exc)}

    mlx_status = await get_mlx_client().health()
    models = await list_models_status()

    any_llm = ollama_status.get("available") or mlx_status.get("available")
    return {
        "ollama": ollama_status,
        "mlx": mlx_status,
        "models": models,
        "any_backend_available": any_llm,
    }


async def generate_answer(
    query: str,
    context: str,
    history: list[dict[str, str]] | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> tuple[str, ModelSpec, GenerationMetrics]:
    spec = resolve_model(model)

    if spec.is_mlx:
        if not mlx_runtime_available():
            raise RuntimeError(
                "mlx-lm is required for MLX models. Install with: pip install mlx-lm"
            )
        answer, metrics = await get_mlx_client().generate_answer(
            query=query,
            context=context,
            history=history,
            model=spec.id,
            temperature=temperature,
            top_p=top_p,
        )
        return answer, spec, metrics

    answer, metrics = await get_ollama_client().generate_answer(
        query=query,
        context=context,
        history=history,
        model=spec.id,
        temperature=temperature,
        top_p=top_p,
    )
    return answer, spec, metrics
