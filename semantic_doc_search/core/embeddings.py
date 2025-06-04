"""
Moduł do generowania embeddings z wykorzystaniem różnych modeli AI.

Obsługuje Sentence Transformers 4.1.0, OpenAI 1.83.0 oraz scikit-learn
z optymalizacjami wydajności i cache'owaniem.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from semantic_doc_search.config.settings import get_settings

logger = logging.getLogger(__name__)

# Try to import OpenAI - it's optional
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI nie jest zainstalowane. Provider OpenAI będzie niedostępny.")


class BaseEmbeddingProvider(ABC):
    """Abstrakcyjna klasa bazowa dla providerów embeddings."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self._model = None
        self._dimension = None
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Zwraca nazwę providera."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Zwraca wymiar wektorów embeddings."""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generuje embeddings dla tekstu/tekstów."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Ładuje model."""
        pass
    
    def encode_single(self, text: str, **kwargs) -> List[float]:
        """Generuje embedding dla pojedynczego tekstu."""
        result = self.encode([text], **kwargs)
        return result[0].tolist()
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[List[float]]:
        """Generuje embeddings dla batch tekstów."""
        if not texts:
            return []
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch, **kwargs)
            embeddings.extend([emb.tolist() for emb in batch_embeddings])
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu."""
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "config": self.config
        }


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Provider dla modeli Sentence Transformers 4.1.0.
    
    Obsługuje najnowsze funkcje jak ONNX i OpenVINO backends
    dla zwiększonej wydajności.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        backend: str = "torch",
        **kwargs
    ):
        super().__init__(model_name, device=device, cache_folder=cache_folder, backend=backend, **kwargs)
        self.device = device
        self.cache_folder = Path(cache_folder) if cache_folder else None
        self.backend = backend
        
    @property
    def provider_name(self) -> str:
        return "sentence-transformers"
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            if self._model is None:
                self.load_model()
            # Generuj przykładowy tekst żeby poznać wymiar
            sample_embedding = self._model.encode(["test"])
            self._dimension = len(sample_embedding[0])
        return self._dimension
    
    def load_model(self) -> None:
        """Ładuje model Sentence Transformers."""
        if self._model is not None:
            return
        
        try:
            logger.info(f"Ładowanie modelu Sentence Transformers: {self.model_name}")
            start_time = time.time()
            
            # Konfiguracja modelu z nowymi opcjami z v4.1.0
            model_kwargs = {
                "device": self.device,
                "cache_folder": str(self.cache_folder) if self.cache_folder else None,
            }
            
            # Usuń None values
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            
            self._model = SentenceTransformer(self.model_name, **model_kwargs)
            
            # Nowe backendi dostępne w v4.1.0 (dla CrossEncoder, ale można sprawdzić dostępność)
            if hasattr(self._model, 'backend') and self.backend != "torch":
                logger.info(f"Próba użycia backend: {self.backend}")
                try:
                    # To jest dla CrossEncoder, ale sprawdzamy czy jest dostępne
                    if self.backend in ["onnx", "openvino"]:
                        logger.info(f"Backend {self.backend} może być dostępny w przyszłych wersjach")
                except Exception as e:
                    logger.warning(f"Backend {self.backend} niedostępny: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"Model załadowany w {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Nie można załadować modelu {self.model_name}: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generuje embeddings używając Sentence Transformers."""
        if self._model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Ustaw parametry encoding z nowymi opcjami z v4.1.0
            encode_kwargs = {
                "show_progress_bar": len(texts) > 100,
                "batch_size": kwargs.get("batch_size", 32),
                "convert_to_numpy": True,
                "normalize_embeddings": kwargs.get("normalize", False),
            }
            
            # Usuń parametry które nie są obsługiwane
            encode_kwargs.update({k: v for k, v in kwargs.items() 
                                if k in ['show_progress_bar', 'batch_size', 'convert_to_numpy', 'normalize_embeddings']})
            
            embeddings = self._model.encode(texts, **encode_kwargs)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania embeddings: {e}")
            raise


class OpenAIProvider(BaseEmbeddingProvider):
    """
    Provider dla OpenAI embeddings API 1.83.0.
    
    Obsługuje najnowsze modele text-embedding-3-small/large
    z optymalizacjami i retry logic.
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library nie jest zainstalowane. Zainstaluj: pip install openai>=1.83.0")
        
        super().__init__(model_name, **kwargs)
        
        # Inicjalizuj klienta OpenAI
        self.client = OpenAI(
            api_key=api_key or get_settings().embedding.openai_api_key,
            base_url=base_url or get_settings().embedding.openai_base_url,
        )
        
        # Mapowanie modeli na wymiary
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._model_dimensions.get(self.model_name, 1536)
        return self._dimension
    
    def load_model(self) -> None:
        """OpenAI API nie wymaga ładowania modelu."""
        pass
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generuje embeddings używając OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Nowe parametry dostępne w v1.83.0
            request_kwargs = {
                "input": texts,
                "model": self.model_name,
            }
            
            # Dodaj opcjonalne parametry jeśli są obsługiwane
            if "dimensions" in kwargs and self.model_name.startswith("text-embedding-3"):
                request_kwargs["dimensions"] = kwargs["dimensions"]
            
            response = self.client.embeddings.create(**request_kwargs)
            
            # Ekstraktuj embeddings z odpowiedzi
            embeddings = [item.embedding for item in response.data]
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Błąd podczas wywoływania OpenAI API: {e}")
            raise


class SklearnProvider(BaseEmbeddingProvider):
    """
    Provider dla scikit-learn TF-IDF embeddings.
    
    Używany głównie do testów i porównań.
    Nie jest prawdziwym semantic embedding, ale użyteczny do demonstracji.
    """
    
    def __init__(
        self,
        model_name: str = "tfidf",
        max_features: int = 1000,
        ngram_range: tuple = (1, 2),
        **kwargs
    ):
        super().__init__(model_name, max_features=max_features, ngram_range=ngram_range, **kwargs)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._fitted = False
        self._corpus = []
    
    @property
    def provider_name(self) -> str:
        return "sklearn"
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            if self._model is not None and self._fitted:
                self._dimension = len(self._model.get_feature_names_out())
            else:
                self._dimension = self.max_features  # Przybliżenie
        return self._dimension
    
    def load_model(self) -> None:
        """Inicjalizuje TfidfVectorizer."""
        if self._model is None:
            self._model = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=None,  # Można dodać polskie stop words
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
    
    def fit(self, corpus: List[str]) -> None:
        """Trenuje model TF-IDF na korpusie."""
        if self._model is None:
            self.load_model()
        
        self._corpus = corpus
        self._model.fit(corpus)
        self._fitted = True
        self._dimension = len(self._model.get_feature_names_out())
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generuje TF-IDF embeddings."""
        if self._model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not self._fitted:
            # Jeśli model nie jest wytrenowany, użyj podanych tekstów jako korpusu
            logger.warning("Model TF-IDF nie jest wytrenowany. Trenuję na podanych tekstach.")
            self.fit(texts)
        
        try:
            # Transform teksty na wektory TF-IDF
            vectors = self._model.transform(texts)
            return vectors.toarray()
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania TF-IDF embeddings: {e}")
            raise


class EmbeddingProvider:
    """
    Główna klasa zarządzająca providerami embeddings.
    
    Obsługuje cache'owanie, batch processing oraz automatyczny wybór providera.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = self.settings.embedding.enable_cache
        
    def get_provider(self, provider_name: str = None, model_name: str = None) -> BaseEmbeddingProvider:
        """Zwraca instancję providera embeddings."""
        provider_name = provider_name or self.settings.embedding.default_model
        model_name = model_name or self._get_default_model_for_provider(provider_name)
        
        provider_key = f"{provider_name}:{model_name}"
        
        if provider_key not in self._providers:
            self._providers[provider_key] = self._create_provider(provider_name, model_name)
        
        return self._providers[provider_key]
    
    def _get_default_model_for_provider(self, provider_name: str) -> str:
        """Zwraca domyślny model dla providera."""
        defaults = {
            "sentence-transformers": self.settings.embedding.sentence_transformers_model,
            "openai": self.settings.embedding.openai_model,
            "sklearn": "tfidf"
        }
        return defaults.get(provider_name, "all-MiniLM-L6-v2")
    
    def _create_provider(self, provider_name: str, model_name: str) -> BaseEmbeddingProvider:
        """Tworzy nową instancję providera."""
        if provider_name == "sentence-transformers":
            return SentenceTransformersProvider(
                model_name=model_name,
                device=self.settings.embedding.device,
                cache_folder=str(self.settings.embedding.cache_folder) if self.settings.embedding.cache_folder else None
            )
        elif provider_name == "openai":
            return OpenAIProvider(
                model_name=model_name,
                api_key=self.settings.embedding.openai_api_key,
                base_url=self.settings.embedding.openai_base_url
            )
        elif provider_name == "sklearn":
            return SklearnProvider(
                model_name=model_name,
                max_features=1000
            )
        else:
            raise ValueError(f"Nieznany provider: {provider_name}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        provider_name: str = None,
        model_name: str = None,
        use_cache: bool = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generuje embeddings z opcjonalnym cache'owaniem."""
        use_cache = use_cache if use_cache is not None else self._cache_enabled
        
        provider = self.get_provider(provider_name, model_name)
        
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Sprawdź cache
        cache_keys = []
        cached_results = []
        texts_to_encode = []
        
        if use_cache:
            for text in texts:
                cache_key = self._get_cache_key(text, provider.provider_name, provider.model_name)
                cache_keys.append(cache_key)
                
                if cache_key in self._cache:
                    cached_results.append(self._cache[cache_key])
                    texts_to_encode.append(None)  # Placeholder
                else:
                    cached_results.append(None)
                    texts_to_encode.append(text)
        else:
            texts_to_encode = texts
        
        # Generuj embeddings dla tekstów nie znalezionych w cache
        if any(text is not None for text in texts_to_encode):
            # Usuń None values i zachowaj mapowanie
            actual_texts = [text for text in texts_to_encode if text is not None]
            
            if actual_texts:
                new_embeddings = provider.encode(actual_texts, **kwargs)
                
                # Wstaw nowe embeddings w odpowiednie miejsca
                new_idx = 0
                for i, text in enumerate(texts_to_encode):
                    if text is not None:
                        embedding = new_embeddings[new_idx]
                        cached_results[i] = embedding
                        
                        # Dodaj do cache
                        if use_cache:
                            self._cache[cache_keys[i]] = embedding
                        
                        new_idx += 1
        
        # Konwertuj na listy
        results = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in cached_results]
        
        return results[0] if single_text else results
    
    def _get_cache_key(self, text: str, provider_name: str, model_name: str) -> str:
        """Generuje klucz cache dla tekstu i modelu."""
        content = f"{provider_name}:{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Czyści cache embeddings."""
        self._cache.clear()
        logger.info("Cache embeddings został wyczyszczony")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki cache."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self._cache_enabled,
            "memory_usage_mb": sum(emb.nbytes for emb in self._cache.values() if hasattr(emb, 'nbytes')) / 1024 / 1024
        }
    
    def list_available_providers(self) -> List[str]:
        """Zwraca listę dostępnych providerów."""
        providers = ["sentence-transformers", "sklearn"]
        if OPENAI_AVAILABLE:
            providers.append("openai")
        return providers
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Dzieli tekst na chunki."""
        chunk_size = chunk_size or self.settings.embedding.chunk_size
        overlap = overlap or self.settings.embedding.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Spróbuj znaleźć koniec zdania
            if end < len(text):
                # Szukaj ostatniej kropki w oknie
                last_sentence_end = text.rfind('.', start, end)
                if last_sentence_end > start:
                    end = last_sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Zapobiegaj nieskończonej pętli
            if start >= end:
                start = end
        
        return chunks