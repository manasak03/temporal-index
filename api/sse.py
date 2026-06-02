"""Server-Sent Events helpers for streaming chat."""

from __future__ import annotations

import json
from typing import Any


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
