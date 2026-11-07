"""Lightweight RAG retriever: chunking + TF-IDF (no vector DB required)."""
from __future__ import annotations
import math, re, hashlib
from collections import Counter

STOP = set("the a an and or of to in is are was were for on with as by at from this that it be have has had do does did not but as if you your we our can could should would i me my".split())

def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]

class RagIndex:
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[dict] = []  # {id, text, source}
        self.doc_freq: Counter = Counter()
        self.doc_count = 0
        self.documents: list[dict] = []

    def chunk(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        out, step = [], max(1, self.chunk_size - self.overlap)
        for i in range(0, len(text), step):
            out.append(text[i:i + self.chunk_size])
            if i + self.chunk_size >= len(text):
                break
        return out

    def add_document(self, text: str, source: str = "manual") -> dict:
        doc_id = hashlib.md5(f"{source}|{text[:200]}".encode()).hexdigest()[:8]
        chunks = self.chunk(text)
        added = 0
        for c in chunks:
            cid = f"{doc_id}-{added}"
            self.chunks.append({"id": cid, "text": c, "source": source})
            for tok in set(tokenize(c)):
                self.doc_freq[tok] += 1
            self.doc_count += 1
            added += 1
        self.documents.append({"doc_id": doc_id, "source": source, "chunks": added, "chars": len(text)})
        return {"doc_id": doc_id, "chunks_added": added, "total_chunks": len(self.chunks)}

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.doc_count + 1) / (df + 1)) + 1.0

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []
        results = []
        for item in self.chunks:
            tf = Counter(tokenize(item["text"]))
            score = sum(tf.get(t, 0) * self._idf(t) for t in q_tokens)
            norm = max(1, len(tf))
            results.append({"chunk_id": item["id"], "source": item["source"], "score": round(score / norm, 4), "text": item["text"][:600]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return [r for r in results[:top_k] if r["score"] > 0]

    def answer(self, query: str, top_k: int = 3) -> dict:
        hits = self.search(query, top_k)
        if not hits:
            return {"answer": "No relevant context found in the indexed documents.", "sources": []}
        context = "\n\n".join([f"[{h['chunk_id']} from {h['source']}] {h['text']}" for h in hits])
        answer = (
            f"Based on retrieved context ({len(hits)} snippet(s)):\n\n{context[:1500]}\n\n"
            f"— Extractive answer generated from indexed documents. "
            f"Connect an LLM provider to rewrite this into a concise summary."
        )
        return {"answer": answer, "sources": [{"chunk_id": h["chunk_id"], "score": h["score"], "source": h["source"]} for h in hits]}

    def stats(self) -> dict:
        top_terms = [w for w, _ in self.doc_freq.most_common(15)]
        return {"documents": len(self.documents), "chunks": len(self.chunks), "vocab_size": len(self.doc_freq), "top_terms": top_terms}
