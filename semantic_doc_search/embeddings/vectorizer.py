"""
Document vectorization module for generating embeddings
"""
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
import psycopg2

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
except ImportError:
    OPENAI_AVAILABLE = False

from semantic_doc_search.database.connection import get_cursor

logger = logging.getLogger(__name__)

class VectorizationError(Exception):
    """Exception raised for errors during document vectorization."""
    pass

class Vectorizer:
    """Base class for text vectorization"""
    
    def __init__(self, model_name: str, dimension: int):
        """
        Initialize the vectorizer.
        
        Args:
            model_name (str): Name of the embedding model
            dimension (int): Dimension of the embeddings
        """
        self.model_name = model_name
        self.dimension = dimension
    
    def vectorize(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.
        Must be implemented by subclasses.
        
        Args:
            text (str): Text to vectorize
            
        Returns:
            List[float]: Embedding vector
        """
        raise NotImplementedError("Subclasses must implement vectorize method")
    
    def save_embedding(self, document_id: int, embedding: List[float]) -> bool:
        """
        Save embedding vector to the database.
        
        Args:
            document_id (int): Document ID
            embedding (List[float]): Embedding vector
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            with get_cursor(commit=True) as cursor:
                cursor.execute(
                    """
                    INSERT INTO doc_search.embeddings (document_id, model_name, embedding_vector)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (document_id, self.model_name, embedding)
                )
                embedding_id = cursor.fetchone()[0]
                logger.info(f"Saved embedding (ID: {embedding_id}) for document ID: {document_id}")
                return True
        except psycopg2.Error as e:
            logger.error(f"Error saving embedding: {e}")
            return False

class SentenceTransformerVectorizer(Vectorizer):
    """Vectorizer using Sentence Transformers models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Sentence Transformers vectorizer.
        
        Args:
            model_name (str): Name of the Sentence Transformers model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers not available. Install with 'pip install sentence-transformers'")
        
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_sentence_embedding_dimension()
        super().__init__(model_name, dimension)
        
    def vectorize(self, text: str) -> List[float]:
        """
        Generate embedding vector using Sentence Transformers.
        
        Args:
            text (str): Text to vectorize
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error vectorizing with Sentence Transformers: {e}")
            raise VectorizationError(str(e))

class OpenAIVectorizer(Vectorizer):
    """Vectorizer using OpenAI embedding models"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI vectorizer.
        
        Args:
            model_name (str): Name of the OpenAI embedding model
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with 'pip install openai'")
        
        # Dimensions for OpenAI models
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if model_name not in dimensions:
            logger.warning(f"Unknown model: {model_name}. Using default dimension 1536.")
            dimension = 1536
        else:
            dimension = dimensions[model_name]
            
        super().__init__(model_name, dimension)
    
    def vectorize(self, text: str) -> List[float]:
        """
        Generate embedding vector using OpenAI API.
        
        Args:
            text (str): Text to vectorize
            
        Returns:
            List[float]: Embedding vector
        """
        if not openai.api_key:
            raise VectorizationError("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            logger.error(f"Error vectorizing with OpenAI: {e}")
            raise VectorizationError(str(e))

class SklearnVectorizer(Vectorizer):
    """Simple vectorizer using scikit-learn for demo purposes"""
    
    def __init__(self, model_name: str = "tfidf"):
        """
        Initialize the sklearn vectorizer.
        
        Args:
            model_name (str): Type of sklearn vectorization (tfidf or count)
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            raise ImportError("scikit-learn not available. Install with 'pip install scikit-learn'")
        
        self.model_type = model_name.lower()
        if self.model_type == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=384)
        elif self.model_type == "count":
            self.vectorizer = CountVectorizer(max_features=384)
        else:
            raise ValueError(f"Unknown model type: {model_name}. Use 'tfidf' or 'count'.")
        
        # We'll set it to fixed dimension for vector database compatibility
        super().__init__(f"sklearn-{model_name}", 384)
        
        # Flag to check if the vectorizer has been fitted
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """
        Fit the vectorizer on a collection of texts.
        
        Args:
            texts (List[str]): List of texts to fit on
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def vectorize(self, text: str) -> List[float]:
        """
        Generate embedding vector using scikit-learn.
        
        Args:
            text (str): Text to vectorize
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.is_fitted:
            # If not fitted, fit on this single document (not ideal, but works as fallback)
            self.fit([text])
        
        try:
            # Transform the text to a sparse vector
            sparse_vector = self.vectorizer.transform([text])
            # Convert to dense array and normalize
            dense_vector = sparse_vector.toarray()[0]
            # Ensure dimension is correct (pad or truncate)
            if len(dense_vector) < self.dimension:
                padded = np.zeros(self.dimension)
                padded[:len(dense_vector)] = dense_vector
                dense_vector = padded
            elif len(dense_vector) > self.dimension:
                dense_vector = dense_vector[:self.dimension]
            
            # Normalize
            norm = np.linalg.norm(dense_vector)
            if norm > 0:
                dense_vector = dense_vector / norm
                
            return dense_vector.tolist()
        except Exception as e:
            logger.error(f"Error vectorizing with scikit-learn: {e}")
            raise VectorizationError(str(e))

def get_vectorizer(model: str = "sklearn") -> Vectorizer:
    """
    Factory function to get the appropriate vectorizer.
    
    Args:
        model (str): Model type to use (openai, sentence-transformer, or sklearn)
        
    Returns:
        Vectorizer: Appropriate vectorizer instance
    """
    model = model.lower()
    
    if model.startswith("openai"):
        try:
            return OpenAIVectorizer()
        except ImportError:
            logger.warning("OpenAI not available, falling back to sklearn")
    
    if model.startswith("sentence") or model.startswith("st-"):
        try:
            return SentenceTransformerVectorizer()
        except ImportError:
            logger.warning("Sentence Transformers not available, falling back to sklearn")
    
    # Default fallback
    return SklearnVectorizer() 