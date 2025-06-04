"""
Database connection and operations management.

Handles PostgreSQL connections with pgvector support using SQLAlchemy 2.0 and psycopg3.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from semantic_doc_search.config.settings import get_settings, AppSettings
from semantic_doc_search.core.models import Base, Document, DocumentEmbedding

logger = logging.getLogger(__name__)


class Database:
    """
    Klasa zarządzania bazą danych PostgreSQL z pgvector.
    
    Obsługuje zarówno synchroniczne jak i asynchroniczne połączenia
    z wykorzystaniem najnowszych wersji psycopg3 i SQLAlchemy 2.0.
    """
    
    def __init__(self, settings: AppSettings | None = None):
        """Inicjalizuje połączenie z bazą danych."""
        self.settings = settings or get_settings()
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
        self._pool = None
        
    @property
    def engine(self):
        """Lazy loading silnika SQLAlchemy."""
        if self._engine is None:
            self._engine = create_engine(
                self.settings.database.url,
                pool_size=self.settings.database.min_pool_size,
                max_overflow=self.settings.database.max_pool_size - self.settings.database.min_pool_size,
                pool_timeout=self.settings.database.pool_timeout,
                echo=self.settings.debug,
                future=True
            )
        return self._engine
    
    @property
    def async_engine(self):
        """Lazy loading asynchronicznego silnika SQLAlchemy."""
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                self.settings.database.async_url,
                pool_size=self.settings.database.min_pool_size,
                max_overflow=self.settings.database.max_pool_size - self.settings.database.min_pool_size,
                pool_timeout=self.settings.database.pool_timeout,
                echo=self.settings.debug,
                future=True
            )
        return self._async_engine
    
    @property
    def session_factory(self):
        """Factory dla sesji SQLAlchemy."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                class_=Session,
                expire_on_commit=False
            )
        return self._session_factory
    
    @property
    def async_session_factory(self):
        """Factory dla asynchronicznych sesji SQLAlchemy."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        return self._async_session_factory
    
    @property
    def pool(self):
        """Connection pool psycopg3."""
        if self._pool is None:
            self._pool = ConnectionPool(
                self.settings.database.url,
                min_size=self.settings.database.min_pool_size,
                max_size=self.settings.database.max_pool_size,
                timeout=self.settings.database.pool_timeout,
                open=True
            )
        return self._pool
    
    def initialize(self) -> None:
        """Inicjalizuje bazę danych i tworzy tabele."""
        logger.info("Inicjalizacja bazy danych...")
        
        # Sprawdź czy można połączyć się z bazą
        self.test_connection()
        
        # Włącz rozszerzenie pgvector
        self.enable_pgvector()
        
        # Utwórz tabele
        self.create_tables()
        
        logger.info("Baza danych została zainicjalizowana")
    
    def test_connection(self) -> bool:
        """Testuje połączenie z bazą danych."""
        try:
            with self.get_connection() as conn:
                result = conn.execute("SELECT version()").fetchone()
                logger.info(f"Połączono z PostgreSQL: {result[0]}")
                return True
        except Exception as e:
            logger.error(f"Nie można połączyć się z bazą danych: {e}")
            raise
    
    def enable_pgvector(self) -> None:
        """Włącza rozszerzenie pgvector."""
        try:
            with self.get_connection() as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                logger.info("Rozszerzenie pgvector zostało włączone")
        except Exception as e:
            logger.error(f"Nie można włączyć rozszerzenia pgvector: {e}")
            raise
    
    def create_tables(self) -> None:
        """Tworzy tabele w bazie danych."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Tabele zostały utworzone")
        except Exception as e:
            logger.error(f"Nie można utworzyć tabel: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Usuwa wszystkie tabele."""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Tabele zostały usunięte")
        except Exception as e:
            logger.error(f"Nie można usunąć tabel: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager dla połączenia psycopg3."""
        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            yield conn
    
    @contextmanager
    def get_session(self):
        """Context manager dla sesji SQLAlchemy."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self):
        """Context manager dla asynchronicznej sesji SQLAlchemy."""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    # Document operations
    def create_document(self, **kwargs: Any) -> Document:
        """Tworzy nowy dokument."""
        with self.get_session() as session:
            document = Document(**kwargs)
            session.add(document)
            session.flush()
            session.refresh(document)
            return document
    
    def get_document(self, document_id: int) -> Document | None:
        """Pobiera dokument po ID."""
        with self.get_session() as session:
            return session.get(Document, document_id)
    
    def get_documents(
        self, 
        limit: int = 100, 
        offset: int = 0,
        filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """Pobiera listę dokumentów z opcjonalnymi filtrami."""
        with self.get_session() as session:
            query = session.query(Document)
            
            if filters:
                if 'source' in filters:
                    query = query.filter(Document.source == filters['source'])
                if 'category' in filters:
                    query = query.filter(Document.category == filters['category'])
                if 'author' in filters:
                    query = query.filter(Document.author == filters['author'])
                if 'is_active' in filters:
                    query = query.filter(Document.is_active == filters['is_active'])
            
            return query.offset(offset).limit(limit).all()
    
    def update_document(self, document_id: int, **kwargs: Any) -> Document | None:
        """Aktualizuje dokument."""
        with self.get_session() as session:
            document = session.get(Document, document_id)
            if document:
                for key, value in kwargs.items():
                    if hasattr(document, key):
                        setattr(document, key, value)
                session.flush()
                session.refresh(document)
            return document
    
    def delete_document(self, document_id: int) -> bool:
        """Usuwa dokument."""
        with self.get_session() as session:
            document = session.get(Document, document_id)
            if document:
                session.delete(document)
                return True
            return False
    
    # Embedding operations
    def create_embedding(self, **kwargs: Any) -> DocumentEmbedding:
        """Tworzy nowy embedding."""
        with self.get_session() as session:
            embedding = DocumentEmbedding(**kwargs)
            session.add(embedding)
            session.flush()
            session.refresh(embedding)
            return embedding
    
    def get_embeddings_by_document(self, document_id: int) -> list[DocumentEmbedding]:
        """Pobiera wszystkie embeddings dla dokumentu."""
        with self.get_session() as session:
            return session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).all()
    
    def delete_embeddings_by_document(self, document_id: int) -> int:
        """Usuwa wszystkie embeddings dla dokumentu."""
        with self.get_session() as session:
            count = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).count()
            session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).delete()
            return count
    
    # Vector search operations
    def similarity_search(
        self,
        query_embedding: list[float],
        model_name: str,
        limit: int = 10,
        similarity_threshold: float | None = None,
        metric: str = "cosine"
    ) -> list[dict[str, Any]]:
        """Wykonuje wyszukiwanie podobieństwa wektorowego."""
        
        # Wybierz operator w zależności od metryki
        operator_map = {
            "cosine": "<=>",
            "l2": "<->", 
            "inner_product": "<#>"
        }
        operator = operator_map.get(metric, "<=>")
        
        query = f"""
        SELECT 
            de.document_id,
            de.embedding {operator} %s::vector AS similarity,
            de.chunk_index,
            de.chunk_text,
            d.title,
            d.content,
            d.source,
            d.author,
            d.category
        FROM document_embeddings de
        JOIN documents d ON de.document_id = d.id
        WHERE de.model_name = %s
            AND d.is_active = true
            {f"AND de.embedding {operator} %s::vector < %s" if similarity_threshold else ""}
        ORDER BY de.embedding {operator} %s::vector
        LIMIT %s
        """
        
        params: list[Any] = [query_embedding, model_name]
        if similarity_threshold:
            params.extend([query_embedding, similarity_threshold])
        params.extend([query_embedding, limit])
        
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
            return [dict(row) for row in result]
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        model_name: str,
        semantic_weight: float = 0.7,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Wykonuje hybrydowe wyszukiwanie (semantyczne + full-text)."""
        
        query = """
        WITH semantic_results AS (
            SELECT 
                de.document_id,
                de.embedding <=> %s::vector AS semantic_distance,
                ROW_NUMBER() OVER (ORDER BY de.embedding <=> %s::vector) AS semantic_rank
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.id
            WHERE de.model_name = %s AND d.is_active = true
            ORDER BY de.embedding <=> %s::vector
            LIMIT %s
        ),
        fulltext_results AS (
            SELECT 
                d.id as document_id,
                ts_rank(d.search_vector, plainto_tsquery('polish', %s)) AS fulltext_score,
                ROW_NUMBER() OVER (ORDER BY ts_rank(d.search_vector, plainto_tsquery('polish', %s)) DESC) AS fulltext_rank
            FROM documents d
            WHERE d.search_vector @@ plainto_tsquery('polish', %s)
                AND d.is_active = true
            ORDER BY ts_rank(d.search_vector, plainto_tsquery('polish', %s)) DESC
            LIMIT %s
        )
        SELECT 
            COALESCE(sr.document_id, fr.document_id) as document_id,
            COALESCE(sr.semantic_distance, 1.0) as semantic_distance,
            COALESCE(fr.fulltext_score, 0.0) as fulltext_score,
            COALESCE(sr.semantic_rank, %s) as semantic_rank,
            COALESCE(fr.fulltext_rank, %s) as fulltext_rank,
            (%s * (1.0 / COALESCE(sr.semantic_rank, %s)) + 
             %s * (1.0 / COALESCE(fr.fulltext_rank, %s))) as hybrid_score,
            d.title,
            d.content,
            d.source,
            d.author,
            d.category
        FROM semantic_results sr
        FULL OUTER JOIN fulltext_results fr ON sr.document_id = fr.document_id
        JOIN documents d ON COALESCE(sr.document_id, fr.document_id) = d.id
        ORDER BY hybrid_score DESC
        LIMIT %s
        """
        
        max_rank = limit * 2
        fulltext_weight = 1.0 - semantic_weight
        
        params: list[Any] = [
            query_embedding, query_embedding, model_name, query_embedding, limit,
            query_text, query_text, query_text, query_text, limit,
            max_rank, max_rank,
            semantic_weight, max_rank,
            fulltext_weight, max_rank,
            limit
        ]
        
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
            return [dict(row) for row in result]
    
    def fulltext_search(
        self,
        query_text: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Wykonuje wyszukiwanie pełnotekstowe."""
        
        where_conditions = ["d.search_vector @@ plainto_tsquery('polish', %s)", "d.is_active = true"]
        params: list[Any] = [query_text]
        
        if filters:
            if 'source' in filters:
                where_conditions.append("d.source = %s")
                params.append(filters['source'])
            if 'category' in filters:
                where_conditions.append("d.category = %s") 
                params.append(filters['category'])
            if 'author' in filters:
                where_conditions.append("d.author = %s")
                params.append(filters['author'])
        
        params.extend([query_text, limit])
        
        query = f"""
        SELECT 
            d.id as document_id,
            d.title,
            d.content,
            d.source,
            d.author,
            d.category,
            ts_rank(d.search_vector, plainto_tsquery('polish', %s)) AS score,
            ts_headline('polish', d.content, plainto_tsquery('polish', %s), 
                       'MaxWords=35, MinWords=15, ShortWord=3') AS headline
        FROM documents d
        WHERE {' AND '.join(where_conditions)}
        ORDER BY score DESC
        LIMIT %s
        """
        
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
            return [dict(row) for row in result]
    
    def update_search_vector(self, document_id: int) -> None:
        """Aktualizuje wektor wyszukiwania pełnotekstowego."""
        query = """
        UPDATE documents 
        SET search_vector = to_tsvector('polish', COALESCE(title, '') || ' ' || COALESCE(content, ''))
        WHERE id = %s
        """
        
        with self.get_connection() as conn:
            conn.execute(query, [document_id])
            conn.commit()
    
    def create_vector_indexes(self, dimension: int, lists: int | None = None) -> None:
        """Tworzy indeksy wektorowe dla konkretnego wymiaru."""
        if lists is None:
            lists = self.settings.database.ivfflat_lists
        
        # Sprawdź czy indeksy już istnieją
        index_check_query = """
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'document_embeddings' 
        AND indexname LIKE %s
        """
        
        with self.get_connection() as conn:
            existing_indexes = conn.execute(
                index_check_query, 
                [f"%_vector_{dimension}_%"]
            ).fetchall()
            
            if existing_indexes:
                logger.info(f"Indeksy wektorowe dla wymiaru {dimension} już istnieją")
                return
            
            # Utwórz indeksy IVFFlat
            indexes = [
                f"CREATE INDEX CONCURRENTLY idx_embeddings_vector_{dimension}_cosine ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists}) WHERE dimension = {dimension}",
                f"CREATE INDEX CONCURRENTLY idx_embeddings_vector_{dimension}_l2 ON document_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists}) WHERE dimension = {dimension}",
                f"CREATE INDEX CONCURRENTLY idx_embeddings_vector_{dimension}_ip ON document_embeddings USING ivfflat (embedding vector_ip_ops) WITH (lists = {lists}) WHERE dimension = {dimension}"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                    conn.commit()
                    logger.info(f"Utworzono indeks wektorowy: {index_sql.split()[2]}")
                except Exception as e:
                    logger.error(f"Nie można utworzyć indeksu: {e}")
                    conn.rollback()
    
    def get_stats(self) -> dict[str, Any]:
        """Zwraca statystyki bazy danych."""
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM documents WHERE is_active = true) as active_documents,
            (SELECT COUNT(*) FROM documents) as total_documents,
            (SELECT COUNT(*) FROM document_embeddings) as total_embeddings,
            (SELECT COUNT(DISTINCT model_name) FROM document_embeddings) as unique_models,
            (SELECT AVG(LENGTH(content)) FROM documents WHERE is_active = true) as avg_content_length
        """
        
        with self.get_connection() as conn:
            result = conn.execute(stats_query).fetchone()
            return dict(result) if result else {}
    
    def close(self) -> None:
        """Zamyka połączenia z bazą danych."""
        if self._pool:
            self._pool.close()
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            self._async_engine.dispose()
        
        logger.info("Połączenia z bazą danych zostały zamknięte")