"""
Query service — thin wrapper around the LangGraph RAG pipeline.

The pipeline handles all orchestration:
  parse_query → graph_search → vector_search → hybrid_rank → generate
"""

from typing import AsyncIterator

from backend.models.query import QueryRequest
from backend.pipeline import rag_graph


async def answer_query(request: QueryRequest) -> AsyncIterator[str]:
    """Run the RAG pipeline and stream answer tokens back to the caller."""
    initial_state = {
        "request": request,
        "parsed": None,
        "graph_chunks": [],
        "vector_results": [],
        "ranked_chunks": [],
        "prompt": "",
        "answer_tokens": [],
    }

    final_state = await rag_graph.ainvoke(initial_state)

    for token in final_state["answer_tokens"]:
        yield token
