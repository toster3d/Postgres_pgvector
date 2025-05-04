"""
Endpointy API dla wyszukiwania semantycznego.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.api.models import SearchQuery, SearchResult
from src.api.dependencies import get_search_engine
from src.search.search import SemanticSearch

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.post("/semantic", response_model=List[SearchResult])
def search_semantic(
    query: SearchQuery,
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> List[Dict[str, Any]]:
    """
    Wyszukuje dokumenty semantycznie podobne do zapytania.
    """
    try:
        results = search_engine.search_semantic(
            query=query.query,
            limit=query.limit
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas wyszukiwania semantycznego: {str(e)}"
        )


@router.post("/fulltext", response_model=List[SearchResult])
def search_fulltext(
    query: SearchQuery,
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> List[Dict[str, Any]]:
    """
    Wyszukuje dokumenty przy użyciu PostgreSQL full-text search.
    """
    try:
        results = search_engine.search_fulltext(
            query=query.query,
            limit=query.limit
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas wyszukiwania pełnotekstowego: {str(e)}"
        )


@router.post("/hybrid", response_model=List[SearchResult])
def search_hybrid(
    query: SearchQuery,
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> List[Dict[str, Any]]:
    """
    Wyszukuje dokumenty przy użyciu hybrydowego podejścia łączącego wyszukiwanie semantyczne i full-text.
    """
    try:
        results = search_engine.search_hybrid(
            query=query.query,
            limit=query.limit,
            semantic_weight=query.semantic_weight
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas wyszukiwania hybrydowego: {str(e)}"
        )


@router.get("/similar/{document_id}", response_model=List[SearchResult])
def get_similar_documents(
    document_id: int,
    limit: int = 5,
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> List[Dict[str, Any]]:
    """
    Znajduje dokumenty podobne do podanego dokumentu.
    """
    try:
        results = search_engine.get_similar_documents(
            document_id=document_id,
            limit=limit
        )
        return results
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas wyszukiwania podobnych dokumentów: {str(e)}"
        ) 