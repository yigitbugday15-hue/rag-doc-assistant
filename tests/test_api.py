from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_ingest_ask_search():
    r = client.post("/ingest", json={"text": "FastAPI is a modern Python framework for building APIs with async support and automatic OpenAPI docs.", "source": "docs"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1
    assert client.post("/ask", json={"question": "What is FastAPI?", "top_k": 1}).json()["sources"]
    assert client.post("/search", json={"question": "async API", "top_k": 1}).status_code == 200
    assert client.get("/documents").status_code == 200
    assert client.get("/stats").status_code == 200

def test_upload_txt():
    files = {"file": ("notes.txt", b"Async endpoints run faster with Uvicorn.", "text/plain")}
    assert client.post("/upload", files=files).status_code == 200

def test_health():
    assert client.get("/health").status_code == 200
