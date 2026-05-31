"""Model catalog: Ollama vs Apple MLX backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

Backend = Literal["ollama", "mlx"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: Backend
    label: str
    mlx_path: str | None = None

    @property
    def is_mlx(self) -> bool:
        return self.backend == "mlx"


def _load_mlx_path_overrides() -> dict[str, str]:
    raw = os.environ.get("MLX_MODEL_PATHS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError as exc:
        raise ValueError("MLX_MODEL_PATHS must be valid JSON, e.g. {\"gemma4:31b-mlx\": \"mlx-community/...\"}") from exc


def _default_catalog() -> list[ModelSpec]:
    overrides = _load_mlx_path_overrides()
    return [
        ModelSpec("qwen2.5:3b", "ollama", "qwen2.5:3b"),
        ModelSpec("gemma2", "ollama", "gemma2"),
        ModelSpec("llama3", "ollama", "llama3"),
        ModelSpec("llama3.1:latest", "ollama", "llama3.1:latest"),
        ModelSpec("qwen3.6:27b", "ollama", "qwen3.6:27b"),
        ModelSpec("gemma4:26b", "ollama", "gemma4:26b"),
        ModelSpec(
            "gemma4:31b-mlx",
            "mlx",
            "gemma4:31b-mlx",
            mlx_path=overrides.get(
                "gemma4:31b-mlx",
                os.environ.get(
                    "MLX_GEMMA4_31B_PATH",
                    "mlx-community/gemma-3-27b-it-4bit",
                ),
            ),
        ),
    ]


_CATALOG: dict[str, ModelSpec] = {spec.id: spec for spec in _default_catalog()}


def list_catalog() -> list[ModelSpec]:
    return list(_CATALOG.values())


def infer_backend(model_id: str) -> Backend:
    if model_id in _CATALOG:
        return _CATALOG[model_id].backend
    if model_id.endswith("-mlx") or "/" in model_id:
        return "mlx"
    return "ollama"


def resolve_model(model_id: str | None, *, default_id: str | None = None) -> ModelSpec:
    chosen = (model_id or default_id or os.environ.get("OLLAMA_MODEL", "llama3.1:latest")).strip()
    if not chosen:
        raise ValueError("No model specified.")

    if chosen in _CATALOG:
        return _CATALOG[chosen]

    backend = infer_backend(chosen)
    if backend == "mlx":
        overrides = _load_mlx_path_overrides()
        mlx_path = overrides.get(chosen, chosen)
        return ModelSpec(chosen, "mlx", chosen, mlx_path=mlx_path)

    return ModelSpec(chosen, "ollama", chosen)
