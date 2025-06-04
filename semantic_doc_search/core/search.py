"""
Semantic search engine for document retrieval.

Provides semantic, full-text, and hybrid search capabilities using pgvector and PostgreSQL.
"""

import logging
from typing import Any

from semantic_doc_search.config.settings import AppSettings, get_settings
from semantic_doc_search.core.database import Database
from semantic_doc_search.core.embeddings import EmbeddingProvider
from semantic_doc_search.core.models import Document

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result with metadata."""
    
    def __init__(
        self,
        document: Document,
        score: float,
        search_type: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        self.document = document
        self.score = score
        self.search_type = search_type
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document.id,
            "title": self.document.title,
            "content": self.document.content,
            "source": self.document.source,
            "author": self.document.author,
            "category": self.document.category,
            "score": self.score,
            "search_type": self.search_type,
            "metadata": self.metadata
        }


class SemanticSearchEngine:
    """
    Main search engine class providing semantic, full-text, and hybrid search.
    """
    
    def __init__(self, database: Database | None = None, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.database = database or Database(settings=self.settings)
        self.embedding_provider = EmbeddingProvider(settings=self.settings)
    
    def search(
        self,
        query: str,
        search_type: str = "semantic",
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> list[SearchResult]:
        """
        Universal search method supporting different search types.
        
        Args:
            query: Search query text
            search_type: Type of search ("semantic", "fulltext", "hybrid")
            limit: Maximum number of results
            filters: Optional filters for documents
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        limit = limit or self.settings.search.default_limit
        limit = min(limit, self.settings.search.max_limit)
        
        if search_type == "semantic":
            return self.semantic_search(query, limit=limit, filters=filters, **kwargs)
        elif search_type == "fulltext":
            return self.fulltext_search(query, limit=limit, filters=filters, **kwargs)
        elif search_type == "hybrid":
            return self.hybrid_search(query, limit=limit, filters=filters, **kwargs)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    
    def semantic_search(
        self,
        query: str,
        model_name: str | None = None,
        provider_name: str | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
        metric: str | None = None,
        filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query text
            model_name: Embedding model name
            provider_name: Embedding provider name
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            metric: Distance metric ("cosine", "l2", "inner_product")
            filters: Document filters
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        try:
            # Use defaults if not provided
            provider_name = provider_name or self.settings.embedding.default_model
            model_name = model_name or self.settings.embedding.sentence_transformers_model
            
            query_embedding = self.embedding_provider.encode(
                query,
                provider_name=provider_name,
                model_name=model_name
            )
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]  # Single text returns nested list
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []
        
        # Perform database search
        try:
            metric = metric or self.settings.search.default_similarity_metric
            similarity_threshold = similarity_threshold or self.settings.search.min_similarity_score
            
            results = self.database.similarity_search(
                query_embedding=query_embedding,
                model_name=model_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
                metric=metric
            )
            
            # Convert to SearchResult objects
            search_results: list[SearchResult] = []
            for result in results:
                document = self.database.get_document(result["document_id"])
                if document and self._apply_filters(document, filters):
                    search_result = SearchResult(
                        document=document,
                        score=result["similarity"],
                        search_type="semantic",
                        metadata={
                            "model_name": model_name,
                            "provider": provider_name,
                            "metric": metric
                        }
                    )
                    search_results.append(search_result)
            
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def fulltext_search(
        self,
        query: str,
        limit: int = 10,
        language: str | None = None,
        filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Perform full-text search using PostgreSQL's text search.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            language: Language for text search
            filters: Document filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            language = language or self.settings.search.fulltext_language
            
            results = self.database.fulltext_search(
                query_text=query,
                limit=limit,
                filters=filters
            )
            
            # Convert to SearchResult objects
            search_results: list[SearchResult] = []
            for result in results:
                document = self.database.get_document(result["document_id"])
                if document:
                    search_result = SearchResult(
                        document=document,
                        score=result["score"],
                        search_type="fulltext",
                        metadata={
                            "language": language,
                            "headline": result.get("headline")
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error during full-text search: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        model_name: str | None = None,
        provider_name: str | None = None,
        semantic_weight: float | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic and full-text search.
        
        Args:
            query: Search query text
            model_name: Embedding model name
            provider_name: Embedding provider name
            semantic_weight: Weight for semantic vs full-text (0.0-1.0)
            limit: Maximum number of results
            filters: Document filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Use defaults if not provided
            provider_name = provider_name or self.settings.embedding.default_model
            model_name = model_name or self.settings.embedding.sentence_transformers_model
            semantic_weight = semantic_weight or self.settings.search.default_semantic_weight
            
            # Generate query embedding
            query_embedding = self.embedding_provider.encode(
                query,
                provider_name=provider_name,
                model_name=model_name
            )
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
            
            results = self.database.hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                model_name=model_name,
                semantic_weight=semantic_weight,
                limit=limit
            )
            
            # Convert to SearchResult objects
            search_results: list[SearchResult] = []
            for result in results:
                document = self.database.get_document(result["document_id"])
                if document and self._apply_filters(document, filters):
                    search_result = SearchResult(
                        document=document,
                        score=result["hybrid_score"],
                        search_type="hybrid",
                        metadata={
                            "model_name": model_name,
                            "semantic_score": result.get("semantic_distance", 0),
                            "fulltext_score": result.get("fulltext_score", 0),
                            "semantic_weight": semantic_weight
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return []
    
    def find_similar_documents(
        self,
        document_id: int,
        model_name: str | None = None,
        provider_name: str | None = None,
        limit: int = 5,
        similarity_threshold: float | None = None
    ) -> list[SearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: Source document ID
            model_name: Embedding model name
            provider_name: Embedding provider name
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Get document embedding
            embeddings = self.database.get_embeddings_by_document(document_id)
            if not embeddings:
                logger.warning(f"No embeddings found for document {document_id}")
                return []
            
            # Use first embedding (chunk_index=0 or first available)
            embedding = embeddings[0]
            query_embedding = embedding.embedding
            
            # Use defaults if not provided
            model_name = model_name or embedding.model_name
            similarity_threshold = similarity_threshold or self.settings.search.min_similarity_score
            
            results = self.database.similarity_search(
                query_embedding=query_embedding,
                model_name=model_name,
                limit=limit + 1,  # +1 because source document will be included
                similarity_threshold=similarity_threshold,
                metric="cosine"
            )
            
            # Filter out the source document and convert to SearchResult
            search_results: list[SearchResult] = []
            for result in results:
                if result["document_id"] != document_id:
                    document = self.database.get_document(result["document_id"])
                    if document:
                        search_result = SearchResult(
                            document=document,
                            score=result["similarity"],
                            search_type="similar",
                            metadata={
                                "source_document_id": document_id,
                                "model_name": model_name
                            }
                        )
                        search_results.append(search_result)
            
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    def _apply_filters(self, document: Document, filters: dict[str, Any] | None) -> bool:
        """Apply filters to document."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if hasattr(document, key):
                doc_value = getattr(document, key)
                if doc_value != value:
                    return False
        
        return True 