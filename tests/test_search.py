"""
Testy jednostkowe dla modułu wyszukiwania.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Dodaj katalog główny projektu do ścieżki Pythona
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.search import SemanticSearch
from src.embeddings.embeddings import EmbeddingGenerator


class MockEmbeddingGenerator(EmbeddingGenerator):
    """Mock generatora embeddings do testów."""
    
    def __init__(self):
        self._model_name = "mock-model"
        self._dimension = 4
    
    def get_embedding(self, text):
        """Zwraca prosty embedding dla tekstu."""
        # Zawsze zwracaj ten sam wektor dla uproszczenia
        return [0.1, 0.2, 0.3, 0.4]
    
    @property
    def dimension(self):
        return self._dimension
    
    @property
    def model_name(self):
        return self._model_name


class TestSemanticSearch(unittest.TestCase):
    """Testy jednostkowe dla SemanticSearch."""
    
    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Mock sesji bazodanowej
        self.db_mock = MagicMock()
        
        # Mock zapytań SQL
        self.db_mock.execute.return_value = []
        
        # Użyj mocka generatora embeddings
        self.embedding_generator = MockEmbeddingGenerator()
        
        # Patch get_embedding_generator, aby zwracał nasz mock
        self.patcher = patch('src.search.search.get_embedding_generator', return_value=self.embedding_generator)
        self.mock_get_embedding_generator = self.patcher.start()
        
        # Inicjalizuj klasę SemanticSearch z mockami
        self.search = SemanticSearch(db=self.db_mock)
    
    def tearDown(self):
        """Czyszczenie po każdym teście."""
        self.patcher.stop()
    
    def test_init(self):
        """Test inicjalizacji klasy SemanticSearch."""
        self.assertEqual(self.search.db, self.db_mock)
        self.assertEqual(self.search.embedding_generator, self.embedding_generator)
    
    def test_search_semantic(self):
        """Test wyszukiwania semantycznego."""
        # Przygotuj dane testowe
        mock_results = [
            MagicMock(id=1, title="Test Document", content="This is a test", 
                     category_id=1, metadata={}, created_at=None, updated_at=None, similarity=0.2)
        ]
        self.db_mock.execute.return_value = mock_results
        
        # Wywołaj metodę search_semantic
        results = self.search.search_semantic("test query", limit=10)
        
        # Sprawdź, czy execute został wywołany z odpowiednimi parametrami
        self.db_mock.execute.assert_called_once()
        
        # Sprawdź wyniki
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["title"], "Test Document")
        self.assertEqual(results[0]["similarity"], 0.8)  # 1.0 - 0.2
    
    def test_search_fulltext(self):
        """Test wyszukiwania pełnotekstowego."""
        # Przygotuj dane testowe
        mock_results = [
            MagicMock(id=1, title="Test Document", content="This is a test", 
                     category_id=1, metadata={}, created_at=None, updated_at=None, rank=0.8)
        ]
        self.db_mock.execute.return_value = mock_results
        
        # Wywołaj metodę search_fulltext
        results = self.search.search_fulltext("test query", limit=10)
        
        # Sprawdź, czy execute został wywołany z odpowiednimi parametrami
        self.db_mock.execute.assert_called_once()
        
        # Sprawdź wyniki
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["title"], "Test Document")
        self.assertEqual(results[0]["rank"], 0.8)
    
    def test_search_hybrid(self):
        """Test wyszukiwania hybrydowego."""
        # Przygotuj dane testowe
        mock_results = [
            MagicMock(id=1, title="Test Document", content="This is a test", 
                     category_id=1, metadata={}, created_at=None, updated_at=None, hybrid_score=0.75)
        ]
        self.db_mock.execute.return_value = mock_results
        
        # Wywołaj metodę search_hybrid
        results = self.search.search_hybrid("test query", limit=10, semantic_weight=0.6)
        
        # Sprawdź, czy execute został wywołany z odpowiednimi parametrami
        self.db_mock.execute.assert_called_once()
        
        # Sprawdź wyniki
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["title"], "Test Document")
        self.assertEqual(results[0]["score"], 0.75)
    
    def test_search_hybrid_invalid_weight(self):
        """Test wyszukiwania hybrydowego z nieprawidłową wagą."""
        # Testuj z wagą poza zakresem 0-1
        with self.assertRaises(ValueError):
            self.search.search_hybrid("test query", semantic_weight=1.5)
        
        with self.assertRaises(ValueError):
            self.search.search_hybrid("test query", semantic_weight=-0.1)


if __name__ == '__main__':
    unittest.main() 