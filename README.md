# Temporal Index

Hybrid retrieval-augmented chat over structured catalog and waitlist data. Dense Qwen3 embeddings plus sparse TF-IDF in Milvus Lite, with grounded answers from local models (Ollama or Apple MLX).

## Layout

```
api/           # FastAPI chat server
chunking/      # CSV → Markdown chunks
embedding/     # Hybrid embed + Milvus ingestion
retrieval/     # Hybrid search
scraper/       # Data acquisition
data/chunks/   # Exported chunk JSONL
db/            # Milvus Lite + TF-IDF vectorizer
frontend/      # React chat UI
paths.py       # Shared paths and defaults
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m embedding.hybrid
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Stop the API server before re-running ingestion — Milvus Lite locks `db/milvus_local.db` to one process at a time.

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

## Environment

```bash
export OLLAMA_MODEL=llama3.1:latest
export MILVUS_COLLECTION=temporal_index   # default
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Milvus + Ollama + MLX status |
| `GET /api/models` | Model catalog |
| `POST /api/chat` | RAG chat |

## CLI

```bash
python -m chunking.semantic
python -m retrieval.hybrid "wait time for Submariner"
```
