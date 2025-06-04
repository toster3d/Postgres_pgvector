"""Tests for database functionality."""

import pytest
from typing import List, Dict, Any

from semantic_doc_search.core.database import DatabaseManager
from semantic_doc_search.core.models import Document


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    def test_database_manager_initialization(self) -> None:
        """Test that database manager initializes correctly."""
        db_manager = DatabaseManager()
        assert db_manager is not None
    
    def test_create_document(self) -> None:
        """Test document creation."""
        db_manager = DatabaseManager()
        document_data = {
            "title": "Test Document",
            "content": "This is test content.",
            "file_path": "/path/to/test.txt",
            "file_type": "txt"
        }
        
        document_id = db_manager.create_document(**document_data)
        
        assert isinstance(document_id, int)
        assert document_id > 0
    
    def test_get_document(self) -> None:
        """Test document retrieval."""
        db_manager = DatabaseManager()
        document_id = 1
        
        document = db_manager.get_document(document_id)
        
        assert document is not None
        assert isinstance(document, Document) 