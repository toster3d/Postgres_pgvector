"""Tests for search functionality."""

import pytest
from typing import List, Dict, Any

from semantic_doc_search.core.search import SemanticSearchEngine


class TestSemanticSearchEngine:
    """Test cases for SemanticSearchEngine class."""
    
    def test_search_engine_initialization(self) -> None:
        """Test that search engine initializes correctly."""
        engine = SemanticSearchEngine()
        assert engine is not None
    
    def test_search_documents(self) -> None:
        """Test document search functionality."""
        engine = SemanticSearchEngine()
        query = "test query"
        
        results = engine.search(query, limit=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_search_with_filters(self) -> None:
        """Test search with additional filters."""
        engine = SemanticSearchEngine()
        query = "test query"
        filters = {"file_type": "pdf"}
        
        results = engine.search(query, filters=filters, limit=10)
        
        assert isinstance(results, list)
        assert len(results) <= 10 