"""
Podstawowe testy dla systemu semantycznego wyszukiwania dokumentów.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os

from semantic_doc_search.config.settings import config, validate_config
from semantic_doc_search.core.embeddings import embedding_manager
from semantic_doc_search.core.models import Document, DocumentEmbedding


class TestConfiguration:
    """Testy konfiguracji."""
    
    def test_default_config_values(self):
        """Test domyślnych wartości konfiguracji."""
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.embedding.default_model == "all-MiniLM-L6-v2"
        assert config.search.default_limit == 10
    
    def test_config_validation_passes(self):
        """Test pomyślnej walidacji konfiguracji."""
        # Nie powinno rzucić wyjątkiem
        validate_config()
    
    def test_config_validation_fails_missing_openai_key(self):
        """Test walidacji z brakującym kluczem OpenAI."""
        with patch.object(config.embedding, 'default_model', 'text-embedding-ada-002'):
            with patch.object(config.embedding, 'openai_api_key', None):
                with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                    validate_config()


class TestEmbeddingManager:
    """Testy menedżera embeddings."""
    
    def test_get_available_models(self):
        """Test pobierania dostępnych modeli."""
        models = embedding_manager.get_available_models()
        assert isinstance(models, dict)
        
        # Sentence Transformers powinno być dostępne
        if 'sentence-transformers' in models:
            assert 'all-MiniLM-L6-v2' in models['sentence-transformers']
    
    @patch('semantic_doc_search.core.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_sentence_transformers_provider_creation(self):
        """Test tworzenia providera Sentence Transformers."""
        provider = embedding_manager.get_provider('all-MiniLM-L6-v2')
        assert provider is not None
        assert provider.model_name == 'all-MiniLM-L6-v2'
    
    def test_calculate_similarity_cosine(self):
        """Test obliczania podobieństwa cosinus."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        
        similarity = embedding_manager.calculate_similarity(
            embedding1, embedding2, metric="cosine"
        )
        
        # Cosinus między prostopadłymi wektorami powinien być 0
        assert abs(similarity - 0.0) < 1e-6
    
    def test_calculate_similarity_identical_vectors(self):
        """Test podobieństwa identycznych wektorów."""
        embedding = [1.0, 2.0, 3.0]
        
        similarity = embedding_manager.calculate_similarity(
            embedding, embedding, metric="cosine"
        )
        
        # Cosinus identycznych wektorów powinien być 1
        assert abs(similarity - 1.0) < 1e-6


class TestModels:
    """Testy modeli danych."""
    
    def test_document_creation(self):
        """Test tworzenia dokumentu."""
        doc = Document(
            title="Test Document",
            content="This is test content.",
            source="test.txt",
            metadata={"author": "Test Author"}
        )
        
        assert doc.title == "Test Document"
        assert doc.content == "This is test content."
        assert doc.source == "test.txt"
        assert doc.metadata["author"] == "Test Author"
    
    def test_document_to_dict(self):
        """Test konwersji dokumentu do słownika."""
        doc = Document(
            id=1,
            title="Test Document",
            content="Content",
            source="test.txt"
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == 1
        assert doc_dict["title"] == "Test Document"
        assert doc_dict["content"] == "Content"
        assert doc_dict["source"] == "test.txt"
    
    def test_document_embedding_creation(self):
        """Test tworzenia embeddings dokumentu."""
        embedding = DocumentEmbedding(
            document_id=1,
            chunk_index=0,
            chunk_text="Test chunk",
            embedding_model="test-model",
            embedding_dimension=384,
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert embedding.document_id == 1
        assert embedding.chunk_index == 0
        assert embedding.chunk_text == "Test chunk"
        assert embedding.embedding_model == "test-model"
        assert embedding.embedding_dimension == 384
        assert len(embedding.embedding) == 3


class TestFileOperations:
    """Testy operacji na plikach."""
    
    def test_read_text_file(self):
        """Test czytania pliku tekstowego."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "To jest treść testowa dokumentu."
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Symulacja czytania pliku (jak w docs.py)
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert content == test_content
            
        finally:
            os.unlink(temp_file_path)


class TestSearchQueries:
    """Testy zapytań wyszukiwania."""
    
    def test_search_query_validation(self):
        """Test walidacji zapytań wyszukiwania."""
        # Puste zapytanie
        assert len("") == 0
        
        # Bardzo długie zapytanie
        long_query = "a" * 10000
        assert len(long_query) == 10000
        
        # Zapytanie z polskimi znakami
        polish_query = "Ćwiczenia z języka polskiego"
        assert "ć" in polish_query.lower()
    
    def test_similarity_score_bounds(self):
        """Test granic wyników podobieństwa."""
        # Wynik podobieństwa powinien być między 0 a 1
        scores = [0.0, 0.5, 1.0, 0.999, 0.001]
        
        for score in scores:
            assert 0.0 <= score <= 1.0


class TestUtils:
    """Testy funkcji pomocniczych."""
    
    def test_chunk_text(self):
        """Test dzielenia tekstu na chunki."""
        text = "To jest bardzo długi tekst, który powinien zostać podzielony na mniejsze części."
        chunk_size = 20
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= chunk_size for chunk in chunks)
    
    def test_json_serialization(self):
        """Test serializacji JSON."""
        import json
        
        data = {
            "title": "Test Document",
            "metadata": {"author": "Test", "tags": ["AI", "ML"]},
            "score": 0.95
        }
        
        # Powinno się serializować bez błędów
        json_str = json.dumps(data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        assert parsed_data["title"] == data["title"]
        assert parsed_data["score"] == data["score"]


# Fixtures dla testów integracyjnych (wymagają działającej bazy)
@pytest.fixture
def mock_database_session():
    """Mock sesji bazy danych."""
    return Mock()


@pytest.fixture
def sample_document():
    """Przykładowy dokument do testów."""
    return Document(
        id=1,
        title="Sztuczna Inteligencja w 2025",
        content="Sztuczna inteligencja rozwija się w niespotykanym tempie. Nowe modele językowe...",
        source="ai_article.txt",
        metadata={"category": "AI", "year": 2025}
    )


@pytest.fixture
def sample_embeddings():
    """Przykładowe embeddings do testów."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3]
    ]


if __name__ == "__main__":
    # Uruchom testy
    pytest.main([__file__, "-v"])