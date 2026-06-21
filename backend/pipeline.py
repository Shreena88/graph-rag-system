"""
LangGraph-based RAG pipeline.

Nodes:
  parse_query   → parse + entity extract from raw question
  graph_search  → multi-hop Neo4j traversal
  vector_search → FAISS similarity search
  hybrid_rank   → RRF merge of graph + vector results
  generate      → stream answer from LLM

State flows linearly through all nodes.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, TypedDict

from langgraph.graph import StateGraph, END

from backend.nlp.query_parser import QueryParser
from backend.nlp.entity_extractor import EntityExtractor
from backend.graph.traversal import GraphTraversal, TraversalNode
from backend.graph.neo4j_client import neo4j_client
from backend.vector.store import vector_store, VectorMatch
from backend.vector.hybrid_ranker import HybridRanker, RankedChunk
from backend.llm.router import LLMRouter
from backend.llm.prompt_builder import build_rag_prompt
from backend.models.query import QueryRequest, ParsedQuery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared singletons (same as before, just moved here)
# ---------------------------------------------------------------------------
_extractor = EntityExtractor()
_parser = QueryParser(_extractor)
_traversal = GraphTraversal(neo4j_client)
_ranker = HybridRanker()
_llm = LLMRouter()


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    """Mutable state that flows through every node in the pipeline."""
    request: QueryRequest
    parsed: ParsedQuery | None
    graph_chunks: List[TraversalNode]
    vector_results: List[VectorMatch]
    ranked_chunks: List[RankedChunk]
    prompt: str
    # tokens are yielded externally; we store them for visibility / testing
    answer_tokens: List[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
async def parse_query(state: RAGState) -> RAGState:
    """Node 1 — parse raw question into entities + intent."""
    parsed = _parser.parse(state["request"].question)
    logger.debug("[parse_query] intent=%s entities=%s", parsed.intent, parsed.entities)
    return {**state, "parsed": parsed}


async def graph_search(state: RAGState) -> RAGState:
    """Node 2 — multi-hop graph traversal (graceful fallback on failure)."""
    chunks: List[TraversalNode] = []
    try:
        result = await _traversal.multi_hop(state["parsed"])
        chunks = result["bfs_chunks"]
        logger.debug("[graph_search] retrieved %d graph chunks", len(chunks))
    except Exception as exc:
        logger.warning("[graph_search] skipped — %s", exc)
    return {**state, "graph_chunks": chunks}


async def vector_search(state: RAGState) -> RAGState:
    """Node 3 — FAISS vector similarity search."""
    results = vector_store.search(
        state["request"].question,
        top_k=state["request"].top_k,
    )
    logger.debug("[vector_search] retrieved %d vector results", len(results))
    return {**state, "vector_results": results}


async def hybrid_rank(state: RAGState) -> RAGState:
    """Node 4 — RRF merge of graph + vector results."""
    ranked = _ranker.rank(
        state["graph_chunks"],
        state["vector_results"],
        top_k=state["request"].top_k,
    )
    logger.debug("[hybrid_rank] top ranked chunk score=%.4f", ranked[0].rrf_score if ranked else 0)
    prompt = build_rag_prompt(state["parsed"], ranked)
    return {**state, "ranked_chunks": ranked, "prompt": prompt}


async def generate(state: RAGState) -> RAGState:
    """Node 5 — collect all tokens (streaming happens in the API layer)."""
    tokens: List[str] = []
    async for token in _llm.generate(state["prompt"]):
        tokens.append(token)
    return {**state, "answer_tokens": tokens}


# ---------------------------------------------------------------------------
# Build the compiled graph (done once at import time)
# ---------------------------------------------------------------------------
def _build_graph():
    g = StateGraph(RAGState)

    g.add_node("parse_query", parse_query)
    g.add_node("graph_search", graph_search)
    g.add_node("vector_search", vector_search)
    g.add_node("hybrid_rank", hybrid_rank)
    g.add_node("generate", generate)

    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "graph_search")
    g.add_edge("graph_search", "vector_search")
    g.add_edge("vector_search", "hybrid_rank")
    g.add_edge("hybrid_rank", "generate")
    g.add_edge("generate", END)

    return g.compile()


rag_graph = _build_graph()
