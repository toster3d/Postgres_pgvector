"""
Modele SQLAlchemy dla bazy danych.
"""
from datetime import datetime
import json
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Category(Base):
    """Model kategorii dokumentów."""
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now())

    documents = relationship("Document", back_populates="category")

    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj obiekt do słownika."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Document(Base):
    """Model dokumentu."""
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    category_id = Column(Integer, ForeignKey('categories.id'))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    category = relationship("Category", back_populates="documents")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:30]}...')>"

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj obiekt do słownika."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category_id": self.category_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "category": self.category.to_dict() if self.category else None
        }


class DocumentEmbedding(Base):
    """Model wektora embedding dla dokumentu."""
    __tablename__ = 'document_embeddings'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    embedding = Column('embedding', String)  # Przechowujemy embedding jako JSON
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())

    document = relationship("Document", back_populates="embeddings")

    def __repr__(self):
        return f"<DocumentEmbedding(id={self.id}, document_id={self.document_id}, model='{self.model_name}')>"

    def get_embedding_vector(self) -> List[float]:
        """Konwertuj przechowywany ciąg znaków embeddings do listy."""
        return json.loads(self.embedding)

    def set_embedding_vector(self, vector: List[float]) -> None:
        """Konwertuj listę do formatu ciągu znaków do przechowywania."""
        self.embedding = json.dumps(vector) 