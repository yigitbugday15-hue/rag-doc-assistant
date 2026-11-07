# RAG Document Assistant

![Python](https://img.shields.io/badge/python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-2.x-green) ![RAG](https://img.shields.io/badge/RAG-retrieval-orange) ![License](https://img.shields.io/badge/license-MIT-lightblue)

**FastAPI + Retrieval-Augmented Generation** — Ingest documents, ask questions, get grounded answers with sources.

A practical RAG pipeline with zero heavy dependencies: chunking, TF-IDF retrieval, and answer synthesis. Designed as a drop-in backend you can extend with embeddings + LLM APIs.

## ✨ Features (v2)
- 📥 Document ingestion via JSON **or file upload** (`.txt .md .pdf .py .json .csv`, max 2MB)
- ✂️ Smart chunking with overlap + document IDs
- 🔎 TF-IDF retrieval with normalized scoring + `/search` endpoint
- 📚 Grounded answers with source chunk IDs (no hallucinated citations)
- 📊 `/documents` list + `/stats` (vocab, top terms)
- ✅ Tested, Docker-ready, CI

## 🚀 Quick Start
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 📡 Usage
```bash
# 1. Ingest text
curl -X POST localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"FastAPI is a modern Python framework for building APIs. It supports async endpoints and automatic OpenAPI docs."}'

# 2. Upload file
curl -X POST localhost:8000/upload -F "file=@notes.txt"

# 3. Ask
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI?","top_k":2}'

# 4. Raw search + stats
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"question":"async API"}'
curl localhost:8000/stats
```

## 🔌 Extending to LLM + Vector DB
Replace `RagIndex` with e.g. `pgvector` + OpenAI embeddings, and rewrite `answer()` to call your LLM with retrieved context. The API contract stays the same.

## 🧪 Tests
```bash
pytest -q   # 3 tests incl. upload
```

## 🐳 Docker
```bash
docker build -t rag-doc-assistant .
docker run -p 8000:8000 rag-doc-assistant
```

## 🛠 Tech Stack
Python · FastAPI · Pydantic · PyPDF2 · python-multipart · Docker · GitHub Actions

## 👤 Author
**Yiğit Buğday** — AI Automation Engineer (Remote, EU/US timezones)
