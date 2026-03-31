from typing import AsyncIterator, List
from backend.nlp.entity_extractor import EntityExtractor
from backend.nlp.query_parser import QueryParser
from backend.graph.traversal import GraphTraversal
from backend.graph.neo4j_client import neo4j_client
from backend.vector.store import vector_store
from backend.vector.hybrid_ranker import HybridRanker
from backend.llm.router import LLMRouter
from backend.llm.prompt_builder import build_rag_prompt
from backend.models.query import QueryRequest, ParsedQuery

_extractor = EntityExtractor()
_parser = QueryParser(_extractor)
_traversal = GraphTraversal(neo4j_client)
_ranker = HybridRanker()
_llm = LLMRouter()


async def answer_query(request: QueryRequest) -> AsyncIterator[str]:
    # 1. Understand query
    parsed: ParsedQuery = _parser.parse(request.question)

    # 2. Graph traversal (graceful fallback if Neo4j unavailable)
    graph_chunks = []
    try:
        traversal_result = await _traversal.multi_hop(parsed)
        graph_chunks = traversal_result["bfs_chunks"]
    except Exception:
        pass  # continue with vector-only search

    # 3. Vector search
    vector_results = vector_store.search(request.question, top_k=request.top_k)

    # 4. Hybrid rank
    ranked = _ranker.rank(graph_chunks, vector_results, top_k=request.top_k)

    # 5. Build prompt + stream answer
    prompt = build_rag_prompt(parsed, ranked)
    async for token in _llm.generate(prompt):
        yield token
