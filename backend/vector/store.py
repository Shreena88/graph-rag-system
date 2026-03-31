import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class VectorMatch:
    chunk_id: str
    text: str
    score: float
    doc_id: str = ""
    page_number: int = 0


class VectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        self._model = None
        self._model_name = embedding_model
        self._index = None
        self._metadata: List[dict] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)
        return self._index

    def add(self, chunk_id: str, text: str, doc_id: str = "", page_number: int = 0):
        model = self._get_model()
        embedding = model.encode([text], normalize_embeddings=True)
        self._get_index().add(embedding.astype(np.float32))
        self._metadata.append({"chunk_id": chunk_id, "text": text, "doc_id": doc_id, "page_number": page_number})

    def add_batch(self, chunks: List[dict]):
        model = self._get_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
        self._get_index().add(embeddings.astype(np.float32))
        self._metadata.extend(chunks)

    def search(self, query: str, top_k: int = 10) -> List[VectorMatch]:
        if self._get_index().ntotal == 0:
            return []
        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self._get_index().search(q_emb, min(top_k, self._get_index().ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            results.append(VectorMatch(score=float(score), **self._metadata[idx]))
        return results


# In-memory only — resets on every restart
vector_store = VectorStore()
