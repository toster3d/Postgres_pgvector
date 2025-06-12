"""
Moduł do generowania i zarządzania embeddings dokumentów.
Obsługuje różne modele: Sentence Transformers, OpenAI, scikit-learn.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
import time
import numpy as np
from dataclasses import dataclass

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config.settings import config, AVAILABLE_EMBEDDING_MODELS

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Wynik generowania embeddings."""
    
    embeddings: List[List[float]]
    model_name: str
    model_type: str
    dimension: int
    processing_time: float
    chunk_texts: List[str]
    metadata: Dict[str, Any]


class BaseEmbeddingProvider(ABC):
    """Bazowa klasa dla providerów embeddings."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_info = self._get_model_info()
        self._model = None
    
    @abstractmethod
    def _get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu."""
        pass
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Ładuje model."""
        pass
    
    @abstractmethod
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generuje embeddings dla listy tekstów."""
        pass
    
    def ensure_model_loaded(self) -> None:
        """Zapewnia, że model jest załadowany."""
        if self._model is None:
            logger.info(f"Loading {self.model_name} model...")
            self._model = self._load_model()
            logger.info(f"✓ {self.model_name} model loaded")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """Generuje embeddings z mierzeniem czasu."""
        start_time = time.time()
        
        self.ensure_model_loaded()
        
        if batch_size is None:
            batch_size = config.embedding.batch_size
        
        # Generuj embeddings w batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.model_name,
            model_type=self.model_info.get('type', 'unknown'),
            dimension=self.model_info.get('dimension', len(all_embeddings[0]) if all_embeddings else 0),
            processing_time=processing_time,
            chunk_texts=texts,
            metadata={
                'batch_size': batch_size,
                'total_texts': len(texts),
                'avg_time_per_text': processing_time / len(texts) if texts else 0
            }
        )


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Provider dla modeli Sentence Transformers."""
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu ST."""
        for model_name, info in AVAILABLE_EMBEDDING_MODELS.get('sentence-transformers', {}).items():
            if model_name == self.model_name:
                return {**info, 'type': 'sentence-transformers'}
        
        # Fallback dla nieznanych modeli
        return {'type': 'sentence-transformers', 'dimension': 384}
    
    def _load_model(self) -> Any:
        """Ładuje model Sentence Transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package is not installed")
        
        return SentenceTransformer(
            self.model_name,
            cache_folder=config.embedding.sentence_transformers_cache_dir,
            device=config.embedding.sentence_transformers_device
        )
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generuje embeddings używając Sentence Transformers."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()


class OpenAIProvider(BaseEmbeddingProvider):
    """Provider dla modeli OpenAI."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = None
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu OpenAI."""
        for model_name, info in AVAILABLE_EMBEDDING_MODELS.get('openai', {}).items():
            if model_name == self.model_name:
                return {**info, 'type': 'openai'}
        
        # Fallback
        return {'type': 'openai', 'dimension': 1536}
    
    def _load_model(self) -> Any:
        """Ładuje klienta OpenAI."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed")
        
        if not config.embedding.openai_api_key:
            raise ValueError("OpenAI API key is not configured")
        
        from openai import OpenAI  # Import lokalny dla bezpieczeństwa
        return OpenAI(
            api_key=config.embedding.openai_api_key,
            timeout=config.embedding.openai_timeout
        )
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generuje embeddings używając OpenAI API."""
        try:
            response = self._model.embeddings.create(
                input=texts,
                model=self.model_name,
                encoding_format="float"
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class SklearnProvider(BaseEmbeddingProvider):
    """Provider dla modeli scikit-learn (demonstracyjny)."""
    
    def __init__(self, model_name: str = "tfidf-vectorizer"):
        super().__init__(model_name)
        self._fitted = False
        self._vocabulary = None
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu sklearn."""
        return AVAILABLE_EMBEDDING_MODELS.get('sklearn', {}).get(
            self.model_name, 
            {'type': 'sklearn', 'dimension': 1000}
        )
    
    def _load_model(self) -> Any:
        """Ładuje model TF-IDF."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn package is not installed")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=self.model_info.get('max_features', 10000),
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generuje embeddings używając TF-IDF."""
        if not self._fitted:
            # Trenuj na wszystkich tekstach
            self._model.fit(texts)
            self._fitted = True
            self._vocabulary = self._model.vocabulary_
        
        # Generuj embeddings
        tfidf_matrix = self._model.transform(texts)
        return tfidf_matrix.toarray().tolist()


class EmbeddingManager:
    """Główny menedżer embeddings."""
    
    def __init__(self):
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
    
    def get_provider(self, model_name: str) -> BaseEmbeddingProvider:
        """Zwraca provider dla danego modelu."""
        if model_name not in self._providers:
            self._providers[model_name] = self._create_provider(model_name)
        
        return self._providers[model_name]
    
    def _create_provider(self, model_name: str) -> BaseEmbeddingProvider:
        """Tworzy odpowiedni provider dla modelu."""
        # Sprawdź typ modelu
        for model_type, models in AVAILABLE_EMBEDDING_MODELS.items():
            if model_name in models:
                if model_type == 'sentence-transformers':
                    return SentenceTransformersProvider(model_name)
                elif model_type == 'openai':
                    return OpenAIProvider(model_name)
                elif model_type == 'sklearn':
                    return SklearnProvider(model_name)
        
        # Fallback - spróbuj jako Sentence Transformers
        logger.warning(f"Unknown model {model_name}, trying as Sentence Transformers")
        return SentenceTransformersProvider(model_name)
    
    def generate_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> EmbeddingResult:
        """Generuje embeddings dla listy tekstów."""
        if model_name is None:
            model_name = config.embedding.default_model
        
        # Chunking tekstów jeśli potrzebne
        if chunk_size is not None:
            texts = self._chunk_texts(texts, chunk_size, chunk_overlap or 0)
        
        provider = self.get_provider(model_name)
        return provider.generate_embeddings(texts)
    
    def _chunk_texts(
        self, 
        texts: List[str], 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Dzieli teksty na chunki."""
        chunked_texts = []
        
        for text in texts:
            if len(text) <= chunk_size:
                chunked_texts.append(text)
                continue
            
            # Podziel tekst na chunki
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():  # Pomija puste chunki
                    chunked_texts.append(chunk)
        
        return chunked_texts
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Zwraca listę dostępnych modeli."""
        available = {}
        
        for model_type, models in AVAILABLE_EMBEDDING_MODELS.items():
            if model_type == 'sentence-transformers' and SENTENCE_TRANSFORMERS_AVAILABLE:
                available[model_type] = list(models.keys())
            elif model_type == 'openai' and OPENAI_AVAILABLE and config.embedding.openai_api_key:
                available[model_type] = list(models.keys())
            elif model_type == 'sklearn' and SKLEARN_AVAILABLE:
                available[model_type] = list(models.keys())
        
        return available
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """Oblicza podobieństwo między dwoma embeddings."""
        arr1 = np.array(embedding1).reshape(1, -1)
        arr2 = np.array(embedding2).reshape(1, -1)
        
        if metric == "cosine":
            if SKLEARN_AVAILABLE:
                return float(cosine_similarity(arr1, arr2)[0][0])
            else:
                # Implementacja własna
                dot_product = np.dot(arr1, arr2.T)[0][0]
                norm1 = np.linalg.norm(arr1)
                norm2 = np.linalg.norm(arr2)
                return float(dot_product / (norm1 * norm2))
        
        elif metric == "euclidean":
            return float(1 / (1 + np.linalg.norm(arr1 - arr2)))
        
        elif metric == "dot_product":
            return float(np.dot(arr1, arr2.T)[0][0])
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")


# Globalna instancja menedżera
embedding_manager = EmbeddingManager()


# Funkcje pomocnicze
def generate_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    **kwargs
) -> EmbeddingResult:
    """Shortcut do generowania embeddings."""
    return embedding_manager.generate_embeddings(texts, model_name, **kwargs)


def get_available_models() -> Dict[str, List[str]]:
    """Shortcut do pobrania dostępnych modeli."""
    return embedding_manager.get_available_models()


def calculate_similarity(
    embedding1: List[float],
    embedding2: List[float],
    metric: str = "cosine"
) -> float:
    """Shortcut do obliczania podobieństwa."""
    return embedding_manager.calculate_similarity(embedding1, embedding2, metric)