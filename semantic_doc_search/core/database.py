# semantic_doc_search/core/database.py
# Poprawiony moduł dla psycopg3 z connection pooling

"""
Moduł zarządzania bazą danych PostgreSQL z pgvector.
Wykorzystuje psycopg3 z connection pooling dla optymalnej wydajności.
Obsługuje zarówno bezpośrednie operacje psycopg3 jak i SQLAlchemy ORM.
"""

import logging
from typing import Optional, List, Dict, Any, Sequence
from contextlib import contextmanager, asynccontextmanager
import asyncio
from types import TracebackType

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, AsyncConnectionPool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from semantic_doc_search.config.settings import DatabaseConfig, config as app_global_config

# Konfiguracja loggera
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Menedżer bazy danych z connection pooling dla psycopg3.
    Obsługuje synchroniczne i asynchroniczne operacje.
    """
    
    def __init__(self, db_config: DatabaseConfig):
        """
        Inicjalizacja menedżera bazy danych.
        
        Args:
            db_config: Konfiguracja bazy danych.
        """
        self.config = db_config
        self._sync_pool: Optional[ConnectionPool] = None
        self._async_pool: Optional[AsyncConnectionPool] = None
        self._initialized = False
        
        # Connection string dla psycopg3
        self.dsn = self._build_dsn()
        
        # SQLAlchemy engines i session makers
        self._engine = None
        self._async_engine = None
        self._session_maker = None
        self._async_session_maker = None
        
        logger.info(f"DatabaseManager initialized with DSN: {self._safe_dsn()}")
    
    def _build_dsn(self) -> str:
        """Buduje connection string dla psycopg3."""
        return (
            f"postgresql://{self.config.db_user}:{self.config.db_password}"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )
    
    def _safe_dsn(self) -> str:
        """Zwraca bezpieczną wersję DSN (bez hasła) do logowania."""
        return (
            f"postgresql://{self.config.db_user}:***"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )
    
    def initialize_pools(self) -> None:
        """
        Inicjalizuje connection pools.
        """
        if self._initialized:
            logger.warning("Database pools already initialized")
            return
        
        try:
            # Synchroniczny pool
            self._sync_pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                open=True,
                configure=self._configure_connection
            )
            
            # Asynchroniczny pool
            self._async_pool = AsyncConnectionPool(
                conninfo=self.dsn,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                open=True,
                configure=self._configure_connection
            )
            
            # SQLAlchemy engines
            self._engine = create_engine(
                self.dsn,
                pool_size=self.config.max_pool_size,
                max_overflow=0,
                pool_pre_ping=True,
                echo=app_global_config.debug
            )
            
            self._async_engine = create_async_engine(
                self.dsn.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=self.config.max_pool_size,
                max_overflow=0,
                pool_pre_ping=True,
                echo=app_global_config.debug
            )
            
            # Session makers
            self._session_maker = sessionmaker(bind=self._engine, expire_on_commit=False)
            self._async_session_maker = async_sessionmaker(
                bind=self._async_engine, 
                expire_on_commit=False
            )
            
            self._initialized = True
            logger.info("Database connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pools: {e}")
            raise
    
    def _configure_connection(self, conn: Any) -> None:
        """
        Konfiguruje nowe połączenie.
        
        Args:
            conn: Połączenie psycopg3
        """
        # Włączenie rozszerzenia pgvector
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("SET search_path TO public;")
        
        # Konfiguracja connection dla wydajności
        conn.autocommit = False
        
        logger.debug("Connection configured successfully")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager dla synchronicznego połączenia z bazy danych.
        
        Yields:
            psycopg.Connection: Połączenie do bazy danych
        """
        if not self._initialized:
            self.initialize_pools()
        
        if self._sync_pool is None:
            raise RuntimeError("Sync pool not initialized")
        
        conn = None
        try:
            conn = self._sync_pool.getconn()
            conn.row_factory = dict_row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.commit()
                    if self._sync_pool:
                        self._sync_pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
    
    @asynccontextmanager
    async def get_async_connection(self):
        """
        Async context manager dla asynchronicznego połączenia.
        
        Yields:
            psycopg.AsyncConnection: Asynchroniczne połączenie do bazy danych
        """
        if not self._initialized:
            self.initialize_pools()
        
        if self._async_pool is None:
            raise RuntimeError("Async pool not initialized")
        
        conn = None
        try:
            conn = await self._async_pool.getconn()
            conn.row_factory = dict_row
            yield conn
        except Exception as e:
            if conn:
                await conn.rollback()
            logger.error(f"Async database error: {e}")
            raise
        finally:
            if conn:
                try:
                    await conn.commit()
                    if self._async_pool:
                        await self._async_pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning async connection to pool: {e}")
    
    def execute_query(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        """
        Wykonuje zapytanie SELECT i zwraca wyniki.
        
        Args:
            query: Zapytanie SQL
            params: Parametry zapytania
            
        Returns:
            Lista słowników z wynikami
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
    
    def execute_command(self, command: str, params: Sequence[Any] = ()) -> int:
        """
        Wykonuje komendę SQL (INSERT, UPDATE, DELETE).
        
        Args:
            command: Komenda SQL
            params: Parametry komendy
            
        Returns:
            Liczba zmienionych wierszy
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(command, params)
                return cur.rowcount
    
    def execute_many(self, command: str, params_list: List[Sequence[Any]]) -> int:
        """
        Wykonuje komendę SQL dla wielu zestawów parametrów.
        
        Args:
            command: Komenda SQL
            params_list: Lista zestawów parametrów
            
        Returns:
            Liczba zmienionych wierszy
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(command, params_list)
                return cur.rowcount
    
    async def execute_query_async(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        """
        Asynchronicznie wykonuje zapytanie SELECT.
        
        Args:
            query: Zapytanie SQL
            params: Parametry zapytania
            
        Returns:
            Lista słowników z wynikami
        """
        async with self.get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchall()
    
    async def execute_command_async(self, command: str, params: Sequence[Any] = ()) -> int:
        """
        Asynchronicznie wykonuje komendę SQL.
        
        Args:
            command: Komenda SQL
            params: Parametry komendy
            
        Returns:
            Liczba zmienionych wierszy
        """
        async with self.get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(command, params)
                return cur.rowcount
    
    def test_connection(self) -> bool:
        """
        Testuje połączenie z bazą danych.
        
        Returns:
            True jeśli połączenie działa, False w przeciwnym przypadku
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 as test")
                    result = cur.fetchone()
                    if result and result['test'] == 1:
                        logger.info("Database connection test successful")
                        return True
            return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def test_pgvector(self) -> bool:
        """
        Testuje dostępność rozszerzenia pgvector.
        
        Returns:
            True jeśli pgvector jest dostępne, False w przeciwnym przypadku
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS(
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        ) as has_vector
                    """)
                    result = cur.fetchone()
                    if result and result['has_vector']:
                        logger.info("pgvector extension is available")
                        return True
            logger.warning("pgvector extension not found")
            return False
        except Exception as e:
            logger.error(f"pgvector test failed: {e}")
            return False
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Zwraca statystyki connection pool.
        
        Returns:
            Słownik ze statystykami pool
        """
        stats = {}
        
        if self._sync_pool:
            stats['sync_pool'] = {
                'size': self._sync_pool.get_stats().pool_size,
                'available': self._sync_pool.get_stats().pool_available,
                'waiting': self._sync_pool.get_stats().requests_waiting,
            }
        
        if self._async_pool:
            stats['async_pool'] = {
                'size': self._async_pool.get_stats().pool_size,
                'available': self._async_pool.get_stats().pool_available,
                'waiting': self._async_pool.get_stats().requests_waiting,
            }
        
        return stats
    
    def close(self) -> None:
        """
        Zamyka connection pools i zwalnia zasoby.
        """
        if self._sync_pool:
            self._sync_pool.close()
            self._sync_pool = None
        
        if self._async_pool:
            # Asynchroniczne zamknięcie
            asyncio.create_task(self._async_pool.close())
            self._async_pool = None
        
        if self._engine:
            self._engine.dispose()
            self._engine = None
            
        if self._async_engine:
            asyncio.create_task(self._async_engine.aclose())
            self._async_engine = None
        
        self._initialized = False
        logger.info("Database pools closed")
    
    def create_tables(self) -> None:
        """
        Tworzy tabele w bazie danych używając SQL skryptów.
        """
        from pathlib import Path
        
        # Ścieżka do pliku SQL
        sql_file = Path(__file__).parent.parent / "sql" / "create_tables.sql"
        
        if not sql_file.exists():
            logger.warning(f"SQL file not found: {sql_file}")
            return
        
        try:
            sql_content = sql_file.read_text(encoding='utf-8')
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_content)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Zwraca informacje o bazie danych.
        """
        info = {}
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Wersja PostgreSQL
                    cur.execute("SELECT version()")
                    version_result = cur.fetchone()
                    if version_result:
                        info['postgresql_version'] = version_result['version'].split()[1]
                    
                    # Wersja pgvector
                    cur.execute("""
                        SELECT extversion 
                        FROM pg_extension 
                        WHERE extname = 'vector'
                    """)
                    pgvector_result = cur.fetchone()
                    if pgvector_result:
                        info['pgvector_version'] = pgvector_result['extversion']
                    else:
                        info['pgvector_version'] = 'not installed'
                    
                    # Liczba dokumentów
                    cur.execute("SELECT COUNT(*) as count FROM documents")
                    docs_result = cur.fetchone()
                    info['documents_count'] = docs_result['count'] if docs_result else 0
                    
                    # Liczba embeddings
                    cur.execute("SELECT COUNT(*) as count FROM document_embeddings")
                    embeddings_result = cur.fetchone()
                    info['embeddings_count'] = embeddings_result['count'] if embeddings_result else 0
                    
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            info['error'] = str(e)
        
        return info
    
    def create_vector_indexes(self, force: bool = False) -> None:
        """
        Tworzy indeksy wektorowe w bazie danych.
        
        Args:
            force: Czy wymusić ponowne utworzenie indeksów
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if force:
                        # Usuń istniejące indeksy
                        index_names = [
                            'document_embeddings_vector_l2_idx',
                            'document_embeddings_vector_cosine_idx', 
                            'document_embeddings_vector_ip_idx'
                        ]
                        
                        for index_name in index_names:
                            cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                    
                    # Sprawdź czy tabela ma jakieś embeddings
                    cur.execute("SELECT COUNT(*) as count FROM document_embeddings WHERE embedding IS NOT NULL")
                    embedding_count = cur.fetchone()['count']
                    
                    if embedding_count == 0:
                        logger.warning("No embeddings found in database. Skipping vector index creation.")
                        return
                    
                    # Utwórz indeksy wektorowe
                    lists_param = max(embedding_count // 1000, 1)  # Dynamiczne listy
                    lists_param = min(lists_param, 1000)  # Maksymalnie 1000 list
                    
                    cur.execute(f"""
                        SELECT create_vector_indexes('document_embeddings', {lists_param})
                    """)
                    
            logger.info("Vector indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector indexes: {e}")
            raise
    
    def __enter__(self):
        """Support for context manager."""
        self.initialize_pools()
        return self
    
    def __exit__(self, 
                 exc_type: Optional[type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]
                 ) -> None:
        """Support for context manager."""
        self.close()


# Globalna instancja menedżera - używa konfiguracji z app_global_config
db_manager = DatabaseManager(db_config=app_global_config.database)


# Funkcje pomocnicze dla backward compatibility
def get_connection():
    """Zwraca connection context manager."""
    return db_manager.get_connection()


def get_sync_connection():
    """Zwraca synchroniczne połączenie psycopg (alias do get_connection)."""
    return db_manager.get_connection()


def execute_query(query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    """Wykonuje zapytanie SELECT."""
    return db_manager.execute_query(query, params)


def execute_command(command: str, params: Sequence[Any] = ()) -> int:
    """Wykonuje komendę SQL."""
    return db_manager.execute_command(command, params)


def test_connection() -> bool:
    """Testuje połączenie z bazą danych."""
    return db_manager.test_connection()


def test_pgvector() -> bool:
    """Testuje dostępność pgvector."""
    return db_manager.test_pgvector()


# SQLAlchemy session management functions
@contextmanager
def get_sync_session():
    """Context manager dla synchronicznej sesji SQLAlchemy."""
    if not db_manager._initialized:
        db_manager.initialize_pools()
    
    session = db_manager._session_maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_sync_session_raw() -> Session:
    """Zwraca synchroniczną sesję SQLAlchemy (bez context managera)."""
    if not db_manager._initialized:
        db_manager.initialize_pools()
    return db_manager._session_maker()


@contextmanager
def get_sync_session_context():
    """Context manager dla synchronicznej sesji SQLAlchemy."""
    session = get_sync_session_raw()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def get_async_session() -> AsyncSession:
    """Zwraca asynchroniczną sesję SQLAlchemy."""
    if not db_manager._initialized:
        db_manager.initialize_pools()
    return db_manager._async_session_maker()


@asynccontextmanager
async def get_async_session_context():
    """Context manager dla asynchronicznej sesji SQLAlchemy."""
    session = await get_async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()