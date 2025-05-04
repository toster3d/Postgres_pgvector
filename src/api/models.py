"""
Modele Pydantic dla API.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CategoryBase(BaseModel):
    """Model bazowy kategorii."""
    name: str
    description: Optional[str] = None


class CategoryCreate(CategoryBase):
    """Model do tworzenia kategorii."""
    pass


class CategoryResponse(CategoryBase):
    """Model odpowiedzi dla kategorii."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    """Model bazowy dokumentu."""
    title: str
    content: str
    category_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    """Model do tworzenia dokumentu."""
    pass


class DocumentUpdate(BaseModel):
    """Model do aktualizacji dokumentu."""
    title: Optional[str] = None
    content: Optional[str] = None
    category_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    """Model odpowiedzi dla dokumentu."""
    id: int
    created_at: datetime
    updated_at: datetime
    category: Optional[CategoryResponse] = None

    class Config:
        from_attributes = True


class SearchQuery(BaseModel):
    """Model zapytania wyszukiwania."""
    query: str
    limit: int = 10
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Model wyniku wyszukiwania."""
    id: int
    title: str
    content: str
    category_id: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity: Optional[float] = None  # dla wyszukiwania semantycznego
    rank: Optional[float] = None  # dla wyszukiwania pełnotekstowego
    score: Optional[float] = None  # dla wyszukiwania hybrydowego
    created_at: datetime
    updated_at: datetime 