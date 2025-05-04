"""
Klasa do generowania wektorów embeddings dla dokumentów.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

from .config import EMBEDDING_MODEL, OPENAI_API_KEY


class EmbeddingGenerator(ABC):
    """Abstrakcyjna klasa bazowa dla generatorów embeddings."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generuje embedding dla podanego tekstu."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Zwraca wymiar wektorów embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Zwraca nazwę modelu."""
        pass


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Generator embeddings z użyciem OpenAI API."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """Inicjalizuje generator embeddings OpenAI."""
        import openai
        
        # Ustaw klucz API OpenAI
        if not OPENAI_API_KEY:
            raise ValueError(
                "Brak klucza API OpenAI. Ustaw zmienną środowiskową OPENAI_API_KEY."
            )
        openai.api_key = OPENAI_API_KEY
        
        self._model_name = model_name
        self._client = openai.OpenAI()
        
        # Ustaw wymiar wektorów (dla różnych modeli OpenAI)
        if model_name == "text-embedding-ada-002":
            self._dimension = 1536
        else:
            # Dla innych modeli (np. text-embedding-3-small, text-embedding-3-large)
            # Po wywołaniu API możemy zweryfikować faktyczny wymiar
            self._dimension = 1536
    
    def get_embedding(self, text: str) -> List[float]:
        """Generuje embedding dla podanego tekstu przy użyciu OpenAI API."""
        # Przygotuj tekst
        text = text.replace("\n", " ")
        
        # Wywołaj OpenAI API
        response = self._client.embeddings.create(
            input=text,
            model=self._model_name
        )
        
        # Pobierz wektor embedding
        embedding = response.data[0].embedding
        
        # Aktualizuj wymiar, jeśli jest inny niż oczekiwano
        if len(embedding) != self._dimension:
            self._dimension = len(embedding)
            
        return embedding
    
    @property
    def dimension(self) -> int:
        """Zwraca wymiar wektorów embeddings."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Zwraca nazwę modelu."""
        return self._model_name


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """Generator embeddings z użyciem Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Inicjalizuje generator embeddings Sentence Transformers."""
        from sentence_transformers import SentenceTransformer
        
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    def get_embedding(self, text: str) -> List[float]:
        """Generuje embedding dla podanego tekstu przy użyciu Sentence Transformers."""
        # Przygotuj tekst
        text = text.replace("\n", " ")
        
        # Wygeneruj embedding
        embedding = self._model.encode(text)
        
        # Konwertuj do listy
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        """Zwraca wymiar wektorów embeddings."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Zwraca nazwę modelu."""
        return self._model_name


def get_embedding_generator(model_name: str = None) -> EmbeddingGenerator:
    """Zwraca odpowiedni generator embeddings na podstawie nazwy modelu."""
    # Jeśli nie podano nazwy modelu, użyj domyślnej z konfiguracji
    if model_name is None:
        model_name = EMBEDDING_MODEL
    
    # Wybierz generator na podstawie nazwy modelu
    if model_name.startswith("text-embedding"):
        return OpenAIEmbeddingGenerator(model_name)
    else:
        return SentenceTransformerEmbeddingGenerator(model_name) 