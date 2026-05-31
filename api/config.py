"""Application configuration for Temporal Index."""

from __future__ import annotations

import os
from pathlib import Path

from paths import DEFAULT_COLLECTION, DEFAULT_MILVUS_DB

MILVUS_DB_PATH = Path(os.environ.get("MILVUS_DB_PATH", DEFAULT_MILVUS_DB))
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", DEFAULT_COLLECTION)
RETRIEVAL_LIMIT = int(os.environ.get("RETRIEVAL_LIMIT", "5"))
RETRIEVAL_FUSION = os.environ.get("RETRIEVAL_FUSION", "rrf")
RETRIEVAL_ALPHA = float(os.environ.get("RETRIEVAL_ALPHA", "0.7"))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
MLX_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "1024"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
