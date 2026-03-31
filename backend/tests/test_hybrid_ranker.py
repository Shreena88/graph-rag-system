import pytest
from hypothesis import given, strategies as st
from backend.vector.hybrid_ranker import HybridRanker, RankedChunk
from backend.graph.traversal import TraversalNode
from backend.vector.store import VectorMatch


@given(st.integers(min_value=1, max_value=100))
def test_rrf_scores_always_positive(n_results):
    """RRF scores must always be positive regardless of ranking position."""
    ranker = HybridRanker()
    nodes = [TraversalNode(chunk_id=f"c{i}", text=f"text {i}", hop_distance=i) for i in range(n_results)]
    result = ranker.rank(nodes, [], top_k=n_results)
    assert all(r.rrf_score > 0 for r in result)


@given(st.integers(min_value=1, max_value=50))
def test_rank_respects_top_k(top_k):
    """Result count must never exceed top_k."""
    ranker = HybridRanker()
    nodes = [TraversalNode(chunk_id=f"c{i}", text=f"text {i}", hop_distance=i) for i in range(100)]
    result = ranker.rank(nodes, [], top_k=top_k)
    assert len(result) <= top_k


def test_deduplication():
    """Same chunk_id appearing in both graph and vector results should appear once."""
    ranker = HybridRanker()
    nodes = [TraversalNode(chunk_id="shared", text="hello", hop_distance=1)]
    vectors = [VectorMatch(chunk_id="shared", text="hello", score=0.9)]
    result = ranker.rank(nodes, vectors, top_k=10)
    ids = [r.chunk_id for r in result]
    assert ids.count("shared") == 1


def test_graph_results_ranked_higher_than_distant_vector():
    """Chunk at hop_distance=1 should outscore a low-similarity vector match."""
    ranker = HybridRanker()
    nodes = [TraversalNode(chunk_id="graph_chunk", text="relevant", hop_distance=1)]
    vectors = [VectorMatch(chunk_id="vec_chunk", text="less relevant", score=0.1)]
    result = ranker.rank(nodes, vectors, top_k=10)
    graph_score = next(r.rrf_score for r in result if r.chunk_id == "graph_chunk")
    vec_score = next(r.rrf_score for r in result if r.chunk_id == "vec_chunk")
    assert graph_score >= vec_score
