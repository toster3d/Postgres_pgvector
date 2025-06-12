"""
Modele danych dla systemu semantycznego wyszukiwania dokumentów.
Zawiera modele SQLAlchemy do interakcji z bazą danych PostgreSQL.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from sqlalchemy import String, Text, Integer, Boolean, DateTime, JSON, ForeignKey, Index, func
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel

Base = declarative_base()


class Document(Base):
    """Model dokumentu."""
    
    __tablename__ = "documents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    doc_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Full-text search vector (generated column w PostgreSQL)
    content_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR)
    
    # Relacje
    embeddings: Mapped[List["DocumentEmbedding"]] = relationship(
        "DocumentEmbedding",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title[:50]}...')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje model do słownika."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "metadata": self.doc_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentEmbedding(Base):
    """Model embeddings dokumentów."""
    
    __tablename__ = "document_embeddings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(
        String(100), 
        nullable=False, 
        default="sentence-transformers"
    )
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False, default=384)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relacje
    document: Mapped["Document"] = relationship(
        "Document", 
        back_populates="embeddings"
    )
    
    __table_args__ = (
        Index(
            "uix_document_chunk_model",
            "document_id", "chunk_index", "embedding_model",
            unique=True
        ),
        Index("idx_embeddings_document_id", "document_id"),
        Index("idx_embeddings_model", "embedding_model"),
    )
    
    def __repr__(self) -> str:
        return (f"<DocumentEmbedding(id={self.id}, document_id={self.document_id}, "
                f"model='{self.embedding_model}')>")
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje model do słownika."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EmbeddingModel(Base):
    """Model konfiguracji modeli embeddings."""
    
    __tablename__ = "embedding_models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    model_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    def __repr__(self) -> str:
        return (f"<EmbeddingModel(name='{self.model_name}', "
                f"type='{self.model_type}', dim={self.embedding_dimension})>")
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje model do słownika."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "embedding_dimension": self.embedding_dimension,
            "model_config": self.model_config,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SearchHistory(Base):
    """Model historii wyszukiwań."""
    
    __tablename__ = "search_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    search_type: Mapped[str] = mapped_column(String(20), nullable=False)
    model_used: Mapped[Optional[str]] = mapped_column(String(100))
    results_count: Mapped[Optional[int]] = mapped_column(Integer)
    search_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    def __repr__(self) -> str:
        return (f"<SearchHistory(id={self.id}, type='{self.search_type}', "
                f"query='{self.query[:30]}...')>")
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje model do słownika."""
        return {
            "id": self.id,
            "query": self.query,
            "search_type": self.search_type,
            "model_used": self.model_used,
            "results_count": self.results_count,
            "search_time_ms": self.search_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Indeksy dla lepszej wydajności (dodatkowe)
Index("idx_documents_title_gin", Document.title, postgresql_using="gin", postgresql_ops={"title": "gin_trgm_ops"})
Index("idx_documents_content_vector_gin", Document.content_vector, postgresql_using="gin")
Index("idx_search_history_created_at", SearchHistory.created_at)
Index("idx_search_history_search_type", SearchHistory.search_type)


# Pomocnicze typy dla wyników wyszukiwania
class SearchResult:
    """Klasa reprezentująca wynik wyszukiwania."""
    
    def __init__(
        self,
        document: Document,
        score: float,
        search_type: str,
        snippet: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.document = document
        self.score = score
        self.search_type = search_type
        self.snippet = snippet
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje wynik do słownika."""
        result = self.document.to_dict()
        result.update({
            "score": self.score,
            "search_type": self.search_type,
            "snippet": self.snippet,
            "search_metadata": self.metadata
        })
        return result
    
    def __repr__(self) -> str:
        return (f"<SearchResult(doc_id={self.document.id}, "
                f"score={self.score:.4f}, type='{self.search_type}')>")


class HybridSearchResult:
    """Klasa reprezentująca wynik hybrydowego wyszukiwania."""
    
    def __init__(
        self,
        document: Document,
        semantic_score: float,
        text_score: float,
        combined_score: float,
        snippet: Optional[str] = None
    ):
        self.document = document
        self.semantic_score = semantic_score
        self.text_score = text_score
        self.combined_score = combined_score
        self.snippet = snippet
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje wynik do słownika."""
        result = self.document.to_dict()
        result.update({
            "semantic_score": self.semantic_score,
            "text_score": self.text_score,
            "combined_score": self.combined_score,
            "search_type": "hybrid",
            "snippet": self.snippet
        })
        return result
    
    def __repr__(self) -> str:
        return (f"<HybridSearchResult(doc_id={self.document.id}, "
                f"combined_score={self.combined_score:.4f})>")


# Pydantic modele dla walidacji danych wejściowych
class DocumentCreate(BaseModel):
    """Model do tworzenia nowych dokumentów."""
    
    title: str
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        """Konfiguracja Pydantic."""
        extra = "forbid"  # Nie pozwala na dodatkowe pola
        validate_assignment = True