"""Normalize chat history for strict chat templates (e.g. Gemma)."""

from __future__ import annotations


def normalize_conversation_history(
    history: list[dict[str, str]] | None,
    query: str,
) -> list[dict[str, str]]:
    """Merge same-role turns, drop duplicate current user, start with user."""
    turns: list[dict[str, str]] = []
    for item in history or []:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] = f"{turns[-1]['content']}\n\n{content}"
        else:
            turns.append({"role": role, "content": content})

    query_stripped = query.strip()
    while turns and turns[-1]["role"] == "user" and turns[-1]["content"].strip() == query_stripped:
        turns.pop()

    while turns and turns[0]["role"] == "assistant":
        turns.pop(0)

    return turns
