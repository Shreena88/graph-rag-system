"""
Tests for the LangChain LCEL pipeline wrappers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, strategies as st

from backend.graph.traversal import TraversalNode
from backend.models.query import EntityRef, ParsedQuery
from backend.services.lcel_pipeline import GraphTraversalTool


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

entity_ref_strategy = st.builds(
    EntityRef,
    name=st.text(min_size=1, max_size=50),
    type=st.sampled_from(["PERSON", "ORG", "LOC", "CONCEPT"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)

parsed_query_strategy = st.builds(
    ParsedQuery,
    original=st.text(min_size=1, max_size=200),
    intent=st.sampled_from(["factual", "comparative", "causal", "procedural"]),
    entities=st.lists(entity_ref_strategy, min_size=0, max_size=5),
    keywords=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=10),
    sub_questions=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=3),
)

traversal_node_strategy = st.builds(
    TraversalNode,
    chunk_id=st.text(min_size=1, max_size=50),
    text=st.text(min_size=1, max_size=500),
    hop_distance=st.integers(min_value=0, max_value=10),
)


# ---------------------------------------------------------------------------
# Property 3: GraphTraversalTool delegation
# Feature: langchain-integration, Property 3: GraphTraversalTool delegation
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

@given(
    query=parsed_query_strategy,
    bfs_chunks=st.lists(traversal_node_strategy, min_size=0, max_size=20),
)
@settings(max_examples=100)
def test_traversal_tool_delegation(query: ParsedQuery, bfs_chunks: list[TraversalNode]):
    """
    For any ParsedQuery, GraphTraversalTool._arun should return the same list of
    TraversalNode objects as the bfs_chunks from GraphTraversal.multi_hop.

    # Feature: langchain-integration, Property 3: GraphTraversalTool delegation
    # Validates: Requirements 2.2
    """
    mock_traversal = MagicMock()
    mock_traversal.multi_hop = AsyncMock(return_value={"bfs_chunks": bfs_chunks})

    tool = GraphTraversalTool(traversal=mock_traversal)

    result = asyncio.get_event_loop().run_until_complete(tool._arun(query))

    mock_traversal.multi_hop.assert_called_once_with(query)
    assert result == bfs_chunks
