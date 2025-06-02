"""
Document management operations
"""
import logging
from typing import Optional, List, Dict, Any, Union
import psycopg2

from semantic_doc_search.database.connection import get_cursor

logger = logging.getLogger(__name__)

def add_document(title: str, content: str, source: Optional[str] = None, author: Optional[str] = None) -> Optional[int]:
    """
    Add a new document to the database.
    
    Args:
        title (str): Document title
        content (str): Document content
        source (str, optional): Source of the document
        author (str, optional): Author of the document
        
    Returns:
        Optional[int]: ID of the newly added document, None if operation failed
    """
    try:
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO doc_search.documents (title, content, source, author)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (title, content, source, author)
            )
            document_id = cursor.fetchone()[0]
            logger.info(f"Added document '{title}' with ID: {document_id}")
            return document_id
    except psycopg2.Error as e:
        logger.error(f"Error adding document: {e}")
        return None

def get_document(document_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by ID.
    
    Args:
        document_id (int): Document ID
        
    Returns:
        Optional[Dict[str, Any]]: Document data as a dictionary
    """
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM doc_search.documents
                WHERE id = %s
                """,
                (document_id,)
            )
            document = cursor.fetchone()
            return dict(document) if document else None
    except psycopg2.Error as e:
        logger.error(f"Error retrieving document: {e}")
        return None

def update_document(document_id: int, **kwargs) -> bool:
    """
    Update a document's fields.
    
    Args:
        document_id (int): Document ID
        **kwargs: Fields to update (title, content, source, author)
        
    Returns:
        bool: True if update successful, False otherwise
    """
    allowed_fields = {'title', 'content', 'source', 'author'}
    update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
    
    if not update_fields:
        logger.warning("No valid fields provided for update")
        return False
    
    set_clause = ", ".join([f"{field} = %s" for field in update_fields])
    values = list(update_fields.values()) + [document_id]
    
    try:
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                f"""
                UPDATE doc_search.documents
                SET {set_clause}
                WHERE id = %s
                """,
                tuple(values)
            )
            affected_rows = cursor.rowcount
            return affected_rows > 0
    except psycopg2.Error as e:
        logger.error(f"Error updating document: {e}")
        return False

def delete_document(document_id: int) -> bool:
    """
    Delete a document from the database.
    
    Args:
        document_id (int): Document ID
        
    Returns:
        bool: True if deletion successful, False otherwise
    """
    try:
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                DELETE FROM doc_search.documents
                WHERE id = %s
                """,
                (document_id,)
            )
            affected_rows = cursor.rowcount
            return affected_rows > 0
    except psycopg2.Error as e:
        logger.error(f"Error deleting document: {e}")
        return False

def search_documents_by_text(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search documents using full-text search.
    
    Args:
        query (str): Search query
        limit (int, optional): Maximum number of results
        
    Returns:
        List[Dict[str, Any]]: List of matching documents
    """
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, content, source, author, created_at, updated_at,
                       ts_rank(to_tsvector('english', content), to_tsquery('english', %s)) as rank
                FROM doc_search.documents
                WHERE to_tsvector('english', content) @@ to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, limit)
            )
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except psycopg2.Error as e:
        logger.error(f"Error during text search: {e}")
        return []

def list_documents(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    List documents with pagination.
    
    Args:
        limit (int, optional): Maximum number of results
        offset (int, optional): Offset for pagination
        
    Returns:
        List[Dict[str, Any]]: List of documents
    """
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, source, author, created_at, updated_at
                FROM doc_search.documents
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except psycopg2.Error as e:
        logger.error(f"Error listing documents: {e}")
        return [] 