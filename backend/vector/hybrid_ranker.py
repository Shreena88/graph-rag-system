from typing import List
from dataclasses import dataclass
from backend.graph.traversal import TraversalNode
from backend.vector.store import VectorMatch


@dataclass
class RankedChunk:
    chunk_id: str
    text: str
    rrf_score: float
    doc_id: str = ""
    page_number: int = 0


class HybridRanker:
    """Reciprocal Rank Fusion (RRF) merges graph + vector results."""
    K = 60

    def rank(
        self,
        graph_results: List[TraversalNode],
        vector_results: List[VectorMatch],
        top_k: int = 10,
    ) -> List[RankedChunk]:
        scores: dict[str, float] = {}

        for rank, node in enumerate(graph_results):
            scores[node.chunk_id] = scores.get(node.chunk_id, 0) + 1 / (self.K + rank + 1)

        for rank, match in enumerate(vector_results):
            scores[match.chunk_id] = scores.get(match.chunk_id, 0) + 1 / (self.K + rank + 1)

        # Build lookup
        all_chunks: dict[str, dict] = {}
        for n in graph_results:
            all_chunks[n.chunk_id] = {"chunk_id": n.chunk_id, "text": n.text}
        for m in vector_results:
            all_chunks[m.chunk_id] = {
                "chunk_id": m.chunk_id, "text": m.text,
                "doc_id": m.doc_id, "page_number": m.page_number,
            }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RankedChunk(rrf_score=score, **all_chunks[cid])
            for cid, score in ranked[:top_k]
            if cid in all_chunks
        ]
