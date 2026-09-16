import os
import json
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
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.save_dir = os.path.join(base_dir, "vector_store")
        self.load()

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

    def load(self):
        import faiss
        os.makedirs(self.save_dir, exist_ok=True)
        index_path = os.path.join(self.save_dir, "index.faiss")
        meta_path = os.path.join(self.save_dir, "metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self._index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                print(f"[VectorStore] Loaded {len(self._metadata)} vectors from disk.")
            except Exception as e:
                print(f"[VectorStore] Error loading index: {e}")
                self._index = faiss.IndexFlatIP(self.dim)
                self._metadata = []
        else:
            self._index = faiss.IndexFlatIP(self.dim)
            self._metadata = []

    def save(self):
        import faiss
        os.makedirs(self.save_dir, exist_ok=True)
        index_path = os.path.join(self.save_dir, "index.faiss")
        meta_path = os.path.join(self.save_dir, "metadata.json")
        
        faiss.write_index(self._get_index(), index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f)

    def add(self, chunk_id: str, text: str, doc_id: str = "", page_number: int = 0):
        model = self._get_model()
        embedding = model.encode([text], normalize_embeddings=True)
        self._get_index().add(embedding.astype(np.float32))
        self._metadata.append({"chunk_id": chunk_id, "text": text, "doc_id": doc_id, "page_number": page_number})
        self.save()

    def add_batch(self, chunks: List[dict]):
        if not chunks:
            return
        model = self._get_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
        self._get_index().add(embeddings.astype(np.float32))
        self._metadata.extend(chunks)
        self.save()

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


# Persisted to disk on every change
vector_store = VectorStore()
