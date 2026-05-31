"""Repository paths and defaults for Temporal Index."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
DB_DIR = PROJECT_ROOT / "db"

DEFAULT_MODELS_CSV = PROJECT_ROOT / "scraper" / "catalog_models.csv"
DEFAULT_WAITLIST_CSV = PROJECT_ROOT / "scraper" / "waitlist.csv"
DEFAULT_CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"
DEFAULT_MILVUS_DB = DB_DIR / "milvus_local.db"
DEFAULT_VECTORIZER = DB_DIR / "hybrid_tfidf_vectorizer.joblib"
DEFAULT_COLLECTION = "temporal_index"
