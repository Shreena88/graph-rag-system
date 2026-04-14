import pytest
from hypothesis import given, strategies as st
from unittest.mock import patch, MagicMock
import numpy as np
from backend.nlp.chunker import SemanticChunker
from backend.models.document import PageContent


def make_chunker(max_tokens=512):
    chunker = SemanticChunker(max_tokens=max_tokens)
    # Mock the sentence-transformers model
    mock_model = MagicMock()
    mock_model.encode = lambda texts, **kwargs: np.random.rand(len(texts), 384)
    chunker._model = mock_model
    return chunker


@given(st.integers(min_value=64, max_value=512))
def test_no_chunk_exceeds_max_tokens(max_tokens):
    """Every chunk must respect the max_tokens limit."""
    chunker = make_chunker(max_tokens=max_tokens)
    long_text = " ".join([f"word{i}" for i in range(2000)])
    pages = [PageContent(page_number=1, text=long_text)]
    chunks = chunker.chunk(pages)
    assert all(c.token_count <= max_tokens for c in chunks)


def test_empty_input_returns_empty():
    chunker = make_chunker()
    assert chunker.chunk([]) == []


def test_short_text_produces_at_least_one_chunk():
    chunker = make_chunker()
    pages = [PageContent(page_number=1, text="This is a short sentence about something important.")]
    chunks = chunker.chunk(pages)
    assert len(chunks) >= 1


def test_chunk_ids_are_unique():
    chunker = make_chunker()
    text = ". ".join([f"Sentence number {i} contains some content" for i in range(50)])
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
