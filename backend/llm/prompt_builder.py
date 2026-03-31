from typing import List
from backend.vector.hybrid_ranker import RankedChunk
from backend.models.query import ParsedQuery, GraphStep

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. Answer the user's question clearly and naturally using the provided context.

Rules:
- Write in plain, readable prose — no chunk IDs, no technical references
- Be direct and specific — extract the actual facts from the context
- If the context doesn't contain enough information, say so briefly
- Do not mention "the document" or "the context" — just answer the question directly
- Keep the answer concise but complete"""


def build_rag_prompt(
    query: ParsedQuery,
    ranked_chunks: List[RankedChunk],
    reasoning_path: List[GraphStep] = None,
) -> str:
    context_blocks = [rc.text for rc in ranked_chunks if rc.text.strip()]
    context = "\n\n".join(context_blocks)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query.original}\n\n"
        f"Answer:"
    )
