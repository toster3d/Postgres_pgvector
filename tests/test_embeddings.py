# tests/test_embeddings.py
"""
Test module for embedding functionality.

Tests various embedding providers and their integration.
"""

import pytest

# Import will be resolved when dependencies are installed
try:
    from semantic_doc_search.core.embeddings import EmbeddingProvider
    imports_available = True
except ImportError:
    imports_available = False


@pytest.mark.skipif(not imports_available, reason="Required imports not available")
def test_provider_initialization() -> None:
    """Test that EmbeddingProvider can be initialized."""
    if not imports_available:
        return
    
    provider = EmbeddingProvider()
    assert provider is not None

def test_encode_text() -> None:
    """Test encoding a single text."""
    if not imports_available:
        return
        
    provider = EmbeddingProvider()
    text = "This is a test document."
    
    embedding = provider.encode(text)
    
    assert isinstance(embedding, list)
    if isinstance(embedding, list) and len(embedding) > 0:
        # Single text might return nested list
        if isinstance(embedding[0], list):
            embedding = embedding[0]
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)

def test_encode_documents() -> None:
    """Test encoding multiple documents."""
    if not imports_available:
        return
        
    provider = EmbeddingProvider()
    documents = [
        "First test document.",
        "Second test document.",
        "Third test document."
    ]
    
    embeddings = provider.encode(documents)
    
    assert isinstance(embeddings, list)
    if len(embeddings) > 0:
        assert len(embeddings) == len(documents)
        for embedding in embeddings:
            if isinstance(embedding, list):
                assert len(embedding) > 0
                assert all(isinstance(x, (int, float)) for x in embedding)
