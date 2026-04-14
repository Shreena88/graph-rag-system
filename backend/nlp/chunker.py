from typing import List
from dataclasses import dataclass
import numpy as np
from backend.models.document import PageContent


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    token_count: int
    section_id: str = ""


class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_tokens: int = 512):
        self.max_tokens = max_tokens
        self._model = None
        self._model_name = model_name

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def chunk(self, pages: List[PageContent]) -> List[Chunk]:
        import uuid
        sentences = self._split_sentences(pages)
        if not sentences:
            return []

        model = self._get_model()
        texts = [s["text"] for s in sentences]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
        boundaries = self._detect_boundaries(embeddings, threshold=0.75)
        return self._merge_into_chunks(sentences, boundaries, uuid)

    def _split_sentences(self, pages: List[PageContent]) -> List[dict]:
        import re
        sentences = []
        for page in pages:
            # Split on sentence-ending punctuation followed by whitespace or end of string
            raw = re.split(r'(?<=[.!?])\s+', page.text)
            for sent in raw:
                sent = sent.strip()
                if len(sent) > 20:
                    sentences.append({"text": sent, "page_number": page.page_number})
        return sentences

    def _detect_boundaries(self, embeddings: np.ndarray, threshold: float) -> List[int]:
        boundaries = [0]
        for i in range(1, len(embeddings)):
            norm_a = np.linalg.norm(embeddings[i - 1])
            norm_b = np.linalg.norm(embeddings[i])
            if norm_a == 0 or norm_b == 0:
                continue
            sim = np.dot(embeddings[i - 1], embeddings[i]) / (norm_a * norm_b)
            if sim < threshold:
                boundaries.append(i)
        boundaries.append(len(embeddings))
        return boundaries

    def _merge_into_chunks(self, sentences: List[dict], boundaries: List[int], uuid) -> List[Chunk]:
        chunks = []
        for i in range(len(boundaries) - 1):
            segment = sentences[boundaries[i]:boundaries[i + 1]]
            text = " ".join(s["text"] for s in segment)
            token_count = len(text.split())
            # Split further if exceeds max_tokens
            if token_count > self.max_tokens:
                words = text.split()
                for j in range(0, len(words), self.max_tokens):
                    sub_text = " ".join(words[j:j + self.max_tokens])
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=sub_text,
                        page_number=segment[0]["page_number"],
                        token_count=len(sub_text.split()),
                    ))
            else:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    page_number=segment[0]["page_number"],
                    token_count=token_count,
                ))
        return chunks
