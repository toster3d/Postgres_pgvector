"""
Modele SQLAlchemy dla systemu semantycznego wyszukiwania dokumentów.

Definiuje tabele dla dokumentów, embeddings i powiązanych metadanych
z wykorzystaniem najnowszej wersji SQLAlchemy 2.0.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Bazowa klasa dla wszystkich modeli."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje model na słownik."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }


class Document(Base):
    """
    Model dokumentu tekstowego.
    
    Przechowuje metadane dokumentu oraz jego treść z obsługą
    full-text search PostgreSQL.
    """
    
    __tablename__ = "documents"
    
    # Primary fields
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Metadata fields
    source: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    author: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    language: Mapped[str] = mapped_column(String(10), default="pl", nullable=False)
    
    # Custom metadata as JSON
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Full-text search
    search_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)
    
    # File information
    file_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Status fields
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Relationships
    embeddings: Mapped[List["DocumentEmbedding"]] = relationship(
        "DocumentEmbedding",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_documents_title", "title"),
        Index("idx_documents_created_at", "created_at"),
        Index("idx_documents_search_vector", "search_vector", postgresql_using="gin"),
        Index("idx_documents_active", "is_active"),
        Index("idx_documents_category_source", "category", "source"),
        Index("idx_documents_file_hash", "file_hash"),
    )
    
    @hybrid_property
    def content_length(self) -> int:
        """Zwraca długość treści dokumentu."""
        return len(self.content) if self.content else 0
    
    @hybrid_property
    def word_count(self) -> int:
        """Przybliżona liczba słów w dokumencie."""
        return len(self.content.split()) if self.content else 0
    
    def has_embeddings(self) -> bool:
        """Sprawdza czy dokument ma wygenerowane embeddings."""
        return len(self.embeddings) > 0
    
    def get_embedding_by_model(self, model_name: str) -> Optional["DocumentEmbedding"]:
        """Zwraca embedding dla konkretnego modelu."""
        for embedding in self.embeddings:
            if embedding.model_name == model_name:
                return embedding
        return None
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title[:50]}...')>"


class DocumentEmbedding(Base):
    """
    Model embeddings dokumentu.
    
    Przechowuje wektorowe reprezentacje dokumentów z metadanymi
    o modelu użytym do generowania.
    """
    
    __tablename__ = "document_embeddings"
    
    # Primary fields
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Embedding vector - dynamiczny rozmiar w zależności od modelu
    embedding: Mapped[List[float]] = mapped_column(Vector(None), nullable=False)
    
    # Model metadata
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Embedding metadata
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Processing metadata
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Quality metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Relationships
    document: Mapped[Document] = relationship(
        "Document",
        back_populates="embeddings"
    )
    
    # Indexes dla wydajnego wyszukiwania wektorowego
    __table_args__ = (
        Index("idx_embeddings_document_id", "document_id"),
        Index("idx_embeddings_model", "model_name", "provider"),
        Index("idx_embeddings_dimension", "dimension"),
        Index("idx_embeddings_chunk", "document_id", "chunk_index"),
        Index("idx_embeddings_created_at", "created_at"),
        
        # Vector indexes - tworzone dynamicznie w zależności od rozmiaru
        Index(
            "idx_embeddings_vector_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        Index(
            "idx_embeddings_vector_l2", 
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_l2_ops"}
        ),
        Index(
            "idx_embeddings_vector_ip",
            "embedding", 
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_ip_ops"}
        ),
        
        # Composite indexes
        Index("idx_embeddings_model_dimension", "model_name", "dimension"),
        Index("idx_embeddings_document_model", "document_id", "model_name"),
    )
    
    @hybrid_property
    def vector_norm(self) -> Optional[float]:
        """Zwraca normę wektora embeddings."""
        if self.embedding:
            return sum(x * x for x in self.embedding) ** 0.5
        return None
    
    def similarity_cosine(self, other_embedding: List[float]) -> float:
        """Oblicza podobieństwo cosinusowe z innym wektorem."""
        if not self.embedding or not other_embedding:
            return 0.0
        
        # Cosine similarity = dot product / (norm_a * norm_b)
        dot_product = sum(a * b for a, b in zip(self.embedding, other_embedding))
        norm_a = sum(a * a for a in self.embedding) ** 0.5
        norm_b = sum(b * b for b in other_embedding) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def similarity_l2(self, other_embedding: List[float]) -> float:
        """Oblicza odległość L2 z innym wektorem."""
        if not self.embedding or not other_embedding:
            return float('inf')
        
        return sum((a - b) ** 2 for a, b in zip(self.embedding, other_embedding)) ** 0.5
    
    def __repr__(self) -> str:
        return (
            f"<DocumentEmbedding(id={self.id}, document_id={self.document_id}, "
            f"model='{self.model_name}', dim={self.dimension})>"
        )


class EmbeddingModel(Base):
    """
    Model konfiguracji modeli embeddings.
    
    Przechowuje metadane o dostępnych modelach embeddings
    oraz ich konfigurację.
    """
    
    __tablename__ = "embedding_models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Model configuration
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    max_sequence_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    __table_args__ = (
        Index("idx_embedding_models_provider", "provider"),
        Index("idx_embedding_models_active", "is_active"),
        Index("idx_embedding_models_default", "is_default"),
    )
    
    def __repr__(self) -> str:
        return f"<EmbeddingModel(name='{self.name}', provider='{self.provider}', dim={self.dimension})>"


class SearchHistory(Base):
    """
    Model historii wyszukiwań.
    
    Przechowuje informacje o wykonanych wyszukiwaniach
    dla celów analitycznych i optymalizacji.
    """
    
    __tablename__ = "search_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Query details
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Search parameters
    limit: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    semantic_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    filters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Results
    results_count: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_time: Mapped[float] = mapped_column(Float, nullable=False)
    
    # User context
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    __table_args__ = (
        Index("idx_search_history_type", "search_type"),
        Index("idx_search_history_created_at", "created_at"),
        Index("idx_search_history_user", "user_id"),
        Index("idx_search_history_execution_time", "execution_time"),
    )
    
    def __repr__(self) -> str:
        return (
            f"<SearchHistory(id={self.id}, type='{self.search_type}', "
            f"results={self.results_count}, time={self.execution_time:.3f}s)>"
        )