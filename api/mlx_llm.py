"""Apple MLX local generation via mlx-lm (not Ollama)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from api.config import MLX_MAX_TOKENS
from api.metrics import GenerationMetrics
from api.model_registry import ModelSpec, resolve_model
from api.chat_history import normalize_conversation_history
from api.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple[Any, Any]] = {}


def mlx_runtime_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def _require_mlx_path(spec: ModelSpec) -> str:
    if not spec.mlx_path:
        raise ValueError(
            f"MLX model `{spec.id}` has no Hugging Face path. Set MLX_MODEL_PATHS or MLX_GEMMA4_31B_PATH."
        )
    return spec.mlx_path


def _load_model(path: str) -> tuple[Any, Any]:
    if path not in _model_cache:
        from mlx_lm import load

        logger.info("Loading MLX weights from %s (first request may take a while)", path)
        _model_cache[path] = load(path)
    return _model_cache[path]


def _build_chat_messages(
    query: str,
    context: str,
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context or "No context retrieved."),
        },
    ]
    for item in normalize_conversation_history(history, query):
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": query})
    return messages


def _count_tokens(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text)
        return len(encoded) if encoded is not None else 0
    return len(text.split())


def _generate_sync(
    spec: ModelSpec,
    query: str,
    context: str,
    history: list[dict[str, str]] | None,
    *,
    temperature: float,
    top_p: float | None,
) -> tuple[str, GenerationMetrics]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    path = _require_mlx_path(spec)
    model, tokenizer = _load_model(path)
    messages = _build_chat_messages(query, context, history)

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "has_chat_template", True):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"

    kwargs: dict[str, Any] = {
        "max_tokens": MLX_MAX_TOKENS,
        "sampler": make_sampler(temp=temperature, top_p=top_p or 0.0),
        "verbose": False,
    }

    started = time.perf_counter()
    text = generate(model, tokenizer, prompt=prompt, **kwargs)
    generation_ms = int((time.perf_counter() - started) * 1000)

    if not text or not str(text).strip():
        raise RuntimeError("MLX model returned an empty response.")

    answer = str(text).strip()
    prompt_tokens = _count_tokens(tokenizer, prompt)
    completion_tokens = _count_tokens(tokenizer, answer)
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_second = None
    if completion_tokens > 0 and generation_ms > 0:
        tokens_per_second = completion_tokens / (generation_ms / 1000)

    metrics = GenerationMetrics(
        generation_ms=generation_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tokens_per_second=tokens_per_second,
    )
    return answer, metrics


def _stream_sync(
    spec: ModelSpec,
    query: str,
    context: str,
    history: list[dict[str, str]] | None,
    *,
    temperature: float,
    top_p: float | None,
    on_token: Any,
) -> tuple[str, GenerationMetrics]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    path = _require_mlx_path(spec)
    model, tokenizer = _load_model(path)
    messages = _build_chat_messages(query, context, history)

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "has_chat_template", True):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"

    kwargs: dict[str, Any] = {
        "max_tokens": MLX_MAX_TOKENS,
        "sampler": make_sampler(temp=temperature, top_p=top_p or 0.0),
    }

    started = time.perf_counter()
    parts: list[str] = []
    for response in stream_generate(model, tokenizer, prompt=prompt, **kwargs):
        # mlx-lm yields incremental segments in response.text, not cumulative text.
        segment = str(response.text)
        if segment:
            parts.append(segment)
            on_token(segment)

    answer = "".join(parts).strip()
    if not answer:
        raise RuntimeError("MLX model returned an empty response.")

    generation_ms = int((time.perf_counter() - started) * 1000)
    prompt_tokens = _count_tokens(tokenizer, prompt)
    completion_tokens = _count_tokens(tokenizer, answer)
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_second = None
    if completion_tokens > 0 and generation_ms > 0:
        tokens_per_second = completion_tokens / (generation_ms / 1000)

    metrics = GenerationMetrics(
        generation_ms=generation_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tokens_per_second=tokens_per_second,
    )
    return answer, metrics


class MLXLLMClient:
    """Generate chat completions with mlx-lm on Apple Silicon."""

    async def health(self) -> dict[str, Any]:
        from api.model_registry import list_catalog

        if not mlx_runtime_available():
            return {
                "available": False,
                "error": "mlx-lm is not installed. Run: pip install mlx-lm",
                "models": [],
                "loaded_paths": [],
            }

        catalog = [spec for spec in list_catalog() if spec.is_mlx]
        return {
            "available": True,
            "models": [spec.id for spec in catalog],
            "loaded_paths": list(_model_cache.keys()),
            "catalog": [
                {"id": spec.id, "path": spec.mlx_path, "loaded": spec.mlx_path in _model_cache}
                for spec in catalog
            ],
        }

    async def generate_answer(
        self,
        query: str,
        context: str,
        history: list[dict[str, str]] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, GenerationMetrics]:
        spec = resolve_model(model)
        if not spec.is_mlx:
            raise ValueError(f"Model `{spec.id}` is not configured for MLX.")

        temp = 0.2 if temperature is None else temperature
        return await asyncio.to_thread(
            _generate_sync,
            spec,
            query,
            context,
            history,
            temperature=temp,
            top_p=top_p,
        )

    async def stream_generate_answer(
        self,
        query: str,
        context: str,
        history: list[dict[str, str]] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> AsyncIterator[tuple[str, GenerationMetrics | None]]:
        spec = resolve_model(model)
        if not spec.is_mlx:
            raise ValueError(f"Model `{spec.id}` is not configured for MLX.")

        temp = 0.2 if temperature is None else temperature
        queue: asyncio.Queue[tuple[str, GenerationMetrics | None] | Exception | None] = (
            asyncio.Queue()
        )
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:

                def on_token(delta: str) -> None:
                    loop.call_soon_threadsafe(queue.put_nowait, (delta, None))

                _, metrics = _stream_sync(
                    spec,
                    query,
                    context,
                    history,
                    temperature=temp,
                    top_p=top_p,
                    on_token=on_token,
                )
                loop.call_soon_threadsafe(queue.put_nowait, ("", metrics))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            delta, maybe_metrics = item
            if delta:
                yield delta, None
            if maybe_metrics is not None:
                yield "", maybe_metrics
