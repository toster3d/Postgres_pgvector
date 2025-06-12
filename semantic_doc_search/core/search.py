"""
Moduł wyszukiwania dokumentów - semantyczne, pełnotekstowe i hybrydowe.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import time
from sqlalchemy.orm import Session
from sqlalchemy import text, func, and_, or_
from sqlalchemy.sql import select

from .database import get_sync_session, get_sync_connection
from .models import Document, DocumentEmbedding, SearchHistory, SearchResult, HybridSearchResult
from .embeddings import embedding_manager
from ..config.settings import config

logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """Parametry wyszukiwania."""
    
    query: str
    limit: int = 10
    offset: int = 0
    semantic_weight: float = 0.7  # Dla hybrydowego wyszukiwania
    model_name: Optional[str] = None
    source_filter: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    min_score: float = 0.0


@dataclass
class SearchResultSet:
    """Zestaw wyników wyszukiwania."""
    
    results: List[Union[SearchResult, HybridSearchResult]]
    total_found: int
    search_type: str
    search_time: float
    query: str
    params: SearchParams
    metadata: Dict[str, Any]


class FullTextSearchEngine:
    """Silnik wyszukiwania pełnotekstowego."""
    
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        source_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """Wyszukiwanie pełnotekstowe używając PostgreSQL ts_search."""
        
        start_time = time.time()
        results = []
        
        with get_sync_session() as session:
            # Bazowe zapytanie
            query_obj = select(
                Document.id,
                Document.title,
                Document.content,
                Document.source,
                Document.doc_metadata,
                Document.created_at,
                func.ts_rank_cd(
                    Document.content_vector,
                    func.plainto_tsquery(config.search.fts_language, query)
                ).label('score'),
                func.ts_headline(
                    config.search.fts_language,
                    Document.content,
                    func.plainto_tsquery(config.search.fts_language, query),
                    'MaxWords=50, MinWords=10'
                ).label('snippet')
            ).where(
                Document.content_vector.op('@@')(
                    func.plainto_tsquery(config.search.fts_language, query)
                )
            )
            
            # Filtry
            if source_filter:
                query_obj = query_obj.where(Document.source == source_filter)
            
            if metadata_filter:
                for key, value in metadata_filter.items():
                    query_obj = query_obj.where(
                        Document.doc_metadata[key].astext == str(value)
                    )
            
            # Sortowanie i paginacja
            query_obj = query_obj.having(
                func.ts_rank_cd(
                    Document.content_vector,
                    func.plainto_tsquery(config.search.fts_language, query)
                ) >= min_score
            ).order_by(
                func.ts_rank_cd(
                    Document.content_vector,
                    func.plainto_tsquery(config.search.fts_language, query)
                ).desc()
            ).offset(offset).limit(limit)
            
            # Wykonanie zapytania
            rows = session.execute(query_obj).fetchall()
            
            # Przetwarzanie wyników
            for row in rows:
                doc = Document(
                    id=row.id,
                    title=row.title,
                    content=row.content,
                    source=row.source,
                    doc_metadata=row.doc_metadata,
                    created_at=row.created_at
                )
                
                result = SearchResult(
                    document=doc,
                    score=float(row.score),
                    search_type="fulltext",
                    snippet=row.snippet,
                    metadata={"fts_language": config.search.fts_language}
                )
                results.append(result)
        
        search_time = time.time() - start_time
        logger.info(f"Full-text search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results


class SemanticSearchEngine:
    """Silnik wyszukiwania semantycznego."""
    
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        model_name: Optional[str] = None,
        source_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        distance_metric: str = "cosine"
    ) -> List[SearchResult]:
        """Wyszukiwanie semantyczne używając embeddings."""
        
        start_time = time.time()
        
        if model_name is None:
            model_name = config.embedding.default_model
        
        # Generuj embedding dla zapytania
        embedding_result = embedding_manager.generate_embeddings([query], model_name)
        query_embedding = embedding_result.embeddings[0]
        
        results = []
        
        # Wybierz operator odległości
        distance_ops = {
            "cosine": "<=>",
            "l2": "<->", 
            "euclidean": "<->",
            "dot_product": "<#>",
            "inner_product": "<#>"
        }
        
        distance_op = distance_ops.get(distance_metric, "<=>")
        
        with get_sync_connection() as conn:
            # SQL query dla wyszukiwania wektorowego
            sql = f"""
            SELECT 
                d.id,
                d.title,
                d.content,
                d.source,
                d.metadata,
                d.created_at,
                de.chunk_text,
                de.chunk_index,
                (1 - (de.embedding {distance_op} %s::vector)) AS score
            FROM documents d
            JOIN document_embeddings de ON d.id = de.document_id
            WHERE de.embedding_model = %s
                AND de.embedding IS NOT NULL
            """
            
            params = [str(query_embedding), model_name]
            
            # Dodaj filtry
            if source_filter:
                sql += " AND d.source = %s"
                params.append(source_filter)
            
            if metadata_filter:
                for key, value in metadata_filter.items():
                    sql += f" AND d.metadata->%s = %s"
                    params.extend([key, str(value)])
            
            # Filtr minimalnego score
            if min_score > 0.0:
                sql += f" AND (1 - (de.embedding {distance_op} %s::vector)) >= %s"
                params.extend([str(query_embedding), min_score])
            
            # Sortowanie i paginacja
            sql += f" ORDER BY de.embedding {distance_op} %s::vector"
            sql += " OFFSET %s LIMIT %s"
            params.extend([str(query_embedding), offset, limit])
            
            # Wykonanie zapytania
            rows = conn.execute(sql, params).fetchall()
            
            # Przetwarzanie wyników
            seen_docs = set()
            for row in rows:
                if row[0] not in seen_docs:  # Deduplikacja dokumentów
                    doc = Document(
                        id=row[0],
                        title=row[1],
                        content=row[2],
                        source=row[3],
                        doc_metadata=row[4] or {},
                        created_at=row[5]
                    )
                    
                    result = SearchResult(
                        document=doc,
                        score=float(row[8]),
                        search_type="semantic",
                        snippet=row[6][:200] + "..." if len(row[6]) > 200 else row[6],
                        metadata={
                            "model_name": model_name,
                            "chunk_index": row[7],
                            "distance_metric": distance_metric
                        }
                    )
                    results.append(result)
                    seen_docs.add(row[0])
        
        search_time = time.time() - start_time
        logger.info(f"Semantic search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results


class HybridSearchEngine:
    """Silnik hybrydowego wyszukiwania (semantyczne + pełnotekstowe)."""
    
    def __init__(self):
        self.fulltext_engine = FullTextSearchEngine()
        self.semantic_engine = SemanticSearchEngine()
    
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.7,
        model_name: Optional[str] = None,
        source_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[HybridSearchResult]:
        """Hybrydowe wyszukiwanie łączące semantyczne i pełnotekstowe."""
        
        start_time = time.time()
        
        if model_name is None:
            model_name = config.embedding.default_model
        
        # Generuj embedding dla zapytania
        embedding_result = embedding_manager.generate_embeddings([query], model_name)
        query_embedding = embedding_result.embeddings[0]
        
        results = []
        
        with get_sync_connection() as conn:
            # Wywołaj funkcję hybrydowego wyszukiwania z bazy danych
            sql = """
            SELECT 
                doc_id,
                title,
                content,
                source,
                semantic_score,
                text_score,
                combined_score
            FROM hybrid_search(%s, %s::vector, %s, %s)
            """
            
            params = [query, str(query_embedding), semantic_weight, limit * 2]  # Pobierz więcej wyników
            
            rows = conn.execute(sql, params).fetchall()
            
            # Filtrowanie i przetwarzanie wyników
            for row in rows:
                if len(results) >= limit:
                    break
                
                # Sprawdź filtry
                if source_filter and row[3] != source_filter:
                    continue
                
                combined_score = float(row[6])
                if combined_score < min_score:
                    continue
                
                doc = Document(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    source=row[3]
                )
                
                result = HybridSearchResult(
                    document=doc,
                    semantic_score=float(row[4]),
                    text_score=float(row[5]),
                    combined_score=combined_score,
                    snippet=None  # Można dodać później
                )
                results.append(result)
        
        search_time = time.time() - start_time
        logger.info(f"Hybrid search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results


class SearchManager:
    """Główny menedżer wyszukiwania."""
    
    def __init__(self):
        self.fulltext_engine = FullTextSearchEngine()
        self.semantic_engine = SemanticSearchEngine()
        self.hybrid_engine = HybridSearchEngine()
    
    def search(
        self,
        query: str,
        search_type: str = "semantic",
        **kwargs
    ) -> SearchResultSet:
        """Uniwersalna metoda wyszukiwania."""
        
        start_time = time.time()
        
        # Przygotuj parametry
        params = SearchParams(
            query=query,
            limit=kwargs.get('limit', config.search.default_limit),
            offset=kwargs.get('offset', 0),
            semantic_weight=kwargs.get('semantic_weight', config.search.default_semantic_weight),
            model_name=kwargs.get('model_name'),
            source_filter=kwargs.get('source_filter'),
            metadata_filter=kwargs.get('metadata_filter'),
            min_score=kwargs.get('min_score', 0.0)
        )
        
        # Wykonaj wyszukiwanie
        if search_type == "fulltext" or search_type == "text":
            results = self.fulltext_engine.search(
                query=params.query,
                limit=params.limit,
                offset=params.offset,
                source_filter=params.source_filter,
                metadata_filter=params.metadata_filter,
                min_score=params.min_score
            )
        
        elif search_type == "semantic":
            results = self.semantic_engine.search(
                query=params.query,
                limit=params.limit,
                offset=params.offset,
                model_name=params.model_name,
                source_filter=params.source_filter,
                metadata_filter=params.metadata_filter,
                min_score=params.min_score,
                distance_metric=kwargs.get('distance_metric', 'cosine')
            )
        
        elif search_type == "hybrid":
            results = self.hybrid_engine.search(
                query=params.query,
                limit=params.limit,
                offset=params.offset,
                semantic_weight=params.semantic_weight,
                model_name=params.model_name,
                source_filter=params.source_filter,
                metadata_filter=params.metadata_filter,
                min_score=params.min_score
            )
        
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        search_time = time.time() - start_time
        
        # Zapisz w historii
        self._save_search_history(
            query=params.query,
            search_type=search_type,
            model_used=params.model_name,
            results_count=len(results),
            search_time_ms=int(search_time * 1000)
        )
        
        return SearchResultSet(
            results=results,
            total_found=len(results),  # W prawdziwej implementacji byłby to count(*)
            search_type=search_type,
            search_time=search_time,
            query=params.query,
            params=params,
            metadata={
                "engine_type": search_type,
                "performance": {
                    "search_time_ms": int(search_time * 1000),
                    "results_per_second": len(results) / search_time if search_time > 0 else 0
                }
            }
        )
    
    def recommend_similar(
        self,
        document_id: int,
        limit: int = 10,
        model_name: Optional[str] = None
    ) -> List[SearchResult]:
        """Znajdź dokumenty podobne do danego dokumentu."""
        
        if model_name is None:
            model_name = config.embedding.default_model
        
        results = []
        
        with get_sync_connection() as conn:
            # Pobierz embedding dokumentu referencyjnego
            sql_ref = """
            SELECT embedding FROM document_embeddings 
            WHERE document_id = %s AND embedding_model = %s
            LIMIT 1
            """
            
            ref_row = conn.execute(sql_ref, [document_id, model_name]).fetchone()
            if not ref_row:
                logger.warning(f"No embedding found for document {document_id} with model {model_name}")
                return results
            
            ref_embedding = ref_row[0]
            
            # Znajdź podobne dokumenty
            sql = """
            SELECT 
                d.id,
                d.title,
                d.content,
                d.source,
                d.doc_metadata,
                d.created_at,
                de.chunk_text,
                (1 - (de.embedding <=> %s::vector)) AS score
            FROM documents d
            JOIN document_embeddings de ON d.id = de.document_id
            WHERE de.embedding_model = %s
                AND d.id != %s
                AND de.embedding IS NOT NULL
            ORDER BY de.embedding <=> %s::vector
            LIMIT %s
            """
            
            rows = conn.execute(sql, [
                str(ref_embedding), model_name, document_id, 
                str(ref_embedding), limit
            ]).fetchall()
            
            # Przetwarzanie wyników
            for row in rows:
                doc = Document(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    source=row[3],
                    doc_metadata=row[4] or {},
                    created_at=row[5]
                )
                
                result = SearchResult(
                    document=doc,
                    score=float(row[7]),
                    search_type="recommendation",
                    snippet=row[6][:200] + "..." if len(row[6]) > 200 else row[6],
                    metadata={
                        "reference_document_id": document_id,
                        "model_name": model_name
                    }
                )
                results.append(result)
        
        logger.info(f"Recommendation search found {len(results)} similar documents")
        return results
    
    def _save_search_history(
        self,
        query: str,
        search_type: str,
        model_used: Optional[str],
        results_count: int,
        search_time_ms: int
    ) -> None:
        """Zapisuje historię wyszukiwania."""
        try:
            with get_sync_session() as session:
                history = SearchHistory(
                    query=query,
                    search_type=search_type,
                    model_used=model_used,
                    results_count=results_count,
                    search_time_ms=search_time_ms
                )
                session.add(history)
                session.commit()
        except Exception as e:
            logger.warning(f"Failed to save search history: {e}")


# Globalna instancja menedżera wyszukiwania
search_manager = SearchManager()


# Funkcje pomocnicze
def search(query: str, search_type: str = "semantic", **kwargs) -> SearchResultSet:
    """Shortcut do wyszukiwania."""
    return search_manager.search(query, search_type, **kwargs)


def recommend_similar(document_id: int, limit: int = 10, model_name: Optional[str] = None) -> List[SearchResult]:
    """Shortcut do rekomendacji."""
    return search_manager.recommend_similar(document_id, limit, model_name)