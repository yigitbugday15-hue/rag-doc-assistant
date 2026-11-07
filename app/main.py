from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from .retriever import RagIndex

app = FastAPI(title="RAG Document Assistant", version="2.0.0")
index = RagIndex()


class IngestIn(BaseModel):
    text: str = Field(min_length=10, max_length=200_000)
    source: str = "manual"


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "chunks": len(index.chunks)}


@app.post("/ingest")
def ingest(payload: IngestIn):
    return index.add_document(payload.text, payload.source)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".txt", ".md", ".pdf", ".py", ".json", ".csv")):
        raise HTTPException(400, "Supported: .txt .md .pdf .py .json .csv")
    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(400, "File too large (max 2MB)")
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(400, "Could not decode file")
    if file.filename.lower().endswith(".pdf"):
        try:
            from io import BytesIO
            import PyPDF2
            reader = PyPDF2.PdfReader(BytesIO(raw))
            text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception as e:
            raise HTTPException(400, f"PDF parse error: {e}")
    return index.add_document(text, file.filename)


@app.post("/ask")
def ask(payload: AskIn):
    if not index.chunks:
        raise HTTPException(400, "No documents ingested yet. POST /ingest or /upload first.")
    return index.answer(payload.question, top_k=payload.top_k)


@app.post("/search")
def search(payload: AskIn):
    if not index.chunks:
        raise HTTPException(400, "No documents ingested yet.")
    return {"query": payload.question, "hits": index.search(payload.question, top_k=payload.top_k)}


@app.get("/documents")
def documents():
    return {"documents": index.documents}


@app.get("/stats")
def stats():
    return index.stats()
