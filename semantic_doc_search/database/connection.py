"""
Database connection management
"""
import os
import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    'dbname': os.environ.get('DB_NAME', 'semantic_docs'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
}

@contextmanager
def get_connection():
    """
    Context manager for database connections.
    
    Yields:
        psycopg2.connection: Database connection object
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            
@contextmanager
def get_cursor(commit=False):
    """
    Context manager for database cursors.
    
    Args:
        commit (bool): Whether to commit the transaction
        
    Yields:
        psycopg2.cursor: Database cursor object
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=DictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

def test_connection():
    """
    Test the database connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"Successfully connected to PostgreSQL. Version: {version[0]}")
                return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False 