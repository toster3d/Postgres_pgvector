"""
Testy jednostkowe dla modułu embeddings.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Dodaj katalog główny projektu do ścieżki Pythona
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.embeddings import (
    EmbeddingGenerator,
    SentenceTransformerEmbeddingGenerator,
    get_embedding_generator
)


class TestEmbeddingGenerators(unittest.TestCase):
    """Testy jednostkowe dla generatorów embeddings."""
    
    @patch('src.embeddings.embeddings.SentenceTransformer')
    def test_sentence_transformer_embedding_generator(self, mock_st):
        """Test SentenceTransformerEmbeddingGenerator."""
        # Skonfiguruj mock
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4]
        
        mock_st.return_value = mock_model
        
        # Inicjalizuj generator embeddings
        generator = SentenceTransformerEmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        
        # Sprawdź właściwości
        self.assertEqual(generator.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(generator.dimension, 384)
        
        # Sprawdź generowanie embeddingu
        text = "To jest przykładowy tekst do embeddings"
        embedding = generator.get_embedding(text)
        
        # Sprawdź czy model.encode został wywołany
        mock_model.encode.assert_called_once()
        
        # Sprawdź embedding
        self.assertEqual(embedding, [0.1, 0.2, 0.3, 0.4])
    
    @patch('src.embeddings.embeddings.EMBEDDING_MODEL', "all-MiniLM-L6-v2")
    @patch('src.embeddings.embeddings.SentenceTransformerEmbeddingGenerator')
    def test_get_embedding_generator_sentence_transformers(self, mock_st_generator):
        """Test get_embedding_generator dla modelu Sentence Transformers."""
        # Skonfiguruj mock
        mock_generator = MagicMock()
        mock_st_generator.return_value = mock_generator
        
        # Wywołaj funkcję
        generator = get_embedding_generator()
        
        # Sprawdź, czy odpowiedni generator został użyty
        mock_st_generator.assert_called_once_with("all-MiniLM-L6-v2")
        self.assertEqual(generator, mock_generator)
    
    @patch('src.embeddings.embeddings.EMBEDDING_MODEL', "text-embedding-ada-002")
    @patch('src.embeddings.embeddings.OpenAIEmbeddingGenerator')
    def test_get_embedding_generator_openai(self, mock_openai_generator):
        """Test get_embedding_generator dla modelu OpenAI."""
        # Skonfiguruj mock
        mock_generator = MagicMock()
        mock_openai_generator.return_value = mock_generator
        
        # Wywołaj funkcję
        generator = get_embedding_generator()
        
        # Sprawdź, czy odpowiedni generator został użyty
        mock_openai_generator.assert_called_once_with("text-embedding-ada-002")
        self.assertEqual(generator, mock_generator)
    
    @patch('src.embeddings.embeddings.EMBEDDING_MODEL', "all-MiniLM-L6-v2")
    @patch('src.embeddings.embeddings.SentenceTransformerEmbeddingGenerator')
    def test_get_embedding_generator_custom_model(self, mock_st_generator):
        """Test get_embedding_generator z niestandardowym modelem."""
        # Skonfiguruj mock
        mock_generator = MagicMock()
        mock_st_generator.return_value = mock_generator
        
        # Wywołaj funkcję z niestandardowym modelem
        custom_model = "paraphrase-MiniLM-L3-v2"
        generator = get_embedding_generator(model_name=custom_model)
        
        # Sprawdź, czy generator został utworzony z niestandardowym modelem
        mock_st_generator.assert_called_once_with(custom_model)
        self.assertEqual(generator, mock_generator)


if __name__ == '__main__':
    unittest.main() 