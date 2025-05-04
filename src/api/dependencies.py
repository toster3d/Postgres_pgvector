"""
Zależności dla API.
"""
from typing import Generator
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.embeddings.embeddings import get_embedding_generator, EmbeddingGenerator
from src.search.search import SemanticSearch


def get_embedding_model() -> EmbeddingGenerator:
    """Zwraca generator embeddings."""
    try:
        return get_embedding_generator()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Nie można zainicjalizować modelu embeddings: {str(e)}",
        )


def get_search_engine(db: Session = Depends(get_db)) -> SemanticSearch:
    """Zwraca silnik wyszukiwania."""
    try:
        return SemanticSearch(db=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Nie można zainicjalizować silnika wyszukiwania: {str(e)}",
        ) 