"""OpenAI-compatible local LLM client (Ollama)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from api.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS
from api.metrics import GenerationMetrics
from api.chat_history import normalize_conversation_history
from api.prompts import SYSTEM_PROMPT


class LLMClient:
    """Thin wrapper around Ollama's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.base_url.replace('/v1', '')}/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = [item.get("name") for item in payload.get("models", [])]
            return {
                "reachable": True,
                "available": True,
                "configured_model": self.model,
                "available_models": models,
                "model_ready": self.model in models or any(
                    self.model.split(":")[0] in (name or "") for name in models
                ),
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
        messages = self._build_messages(query, context, history)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0.2 if temperature is None else temperature,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        generation_ms = int((time.perf_counter() - started) * 1000)

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM returned no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("LLM returned an empty message.")

        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        tokens_per_second = None
        if completion_tokens and generation_ms > 0:
            tokens_per_second = completion_tokens / (generation_ms / 1000)

        metrics = GenerationMetrics(
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
        )
        return content.strip(), metrics

    def _build_messages(
        self,
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
        messages = self._build_messages(query, context, history)
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0.2 if temperature is None else temperature,
            "stream": True,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        started = time.perf_counter()
        parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        parts.append(text)
                        yield text, None

                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")

        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("LLM returned an empty message.")

        generation_ms = int((time.perf_counter() - started) * 1000)
        if completion_tokens is None:
            completion_tokens = len(answer.split())
        tokens_per_second = None
        if completion_tokens and generation_ms > 0:
            tokens_per_second = completion_tokens / (generation_ms / 1000)

        metrics = GenerationMetrics(
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
        )
        yield "", metrics
