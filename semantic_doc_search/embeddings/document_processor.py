"""
Document processing and embedding management
"""
import logging
from typing import List, Dict, Any, Optional
import psycopg2

from semantic_doc_search.database.connection import get_cursor
from semantic_doc_search.database.documents import get_document
from semantic_doc_search.embeddings.vectorizer import get_vectorizer, Vectorizer, VectorizationError

logger = logging.getLogger(__name__)

def process_document(document_id: int, model: str = "sklearn") -> bool:
    """
    Process a document and generate its embedding vector.
    
    Args:
        document_id (int): ID of the document to process
        model (str): Name of the embedding model to use
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    # Get the document
    document = get_document(document_id)
    if not document:
        logger.error(f"Document with ID {document_id} not found")
        return False
    
    # Get the appropriate vectorizer
    try:
        vectorizer = get_vectorizer(model)
    except Exception as e:
        logger.error(f"Failed to initialize vectorizer: {e}")
        return False
    
    # Generate embedding from document content
    try:
        embedding = vectorizer.vectorize(document['content'])
    except VectorizationError as e:
        logger.error(f"Failed to vectorize document: {e}")
        return False
    
    # Save the embedding to the database
    return vectorizer.save_embedding(document_id, embedding)

def process_multiple_documents(document_ids: List[int], model: str = "sklearn") -> Dict[int, bool]:
    """
    Process multiple documents and generate their embedding vectors.
    
    Args:
        document_ids (List[int]): List of document IDs to process
        model (str): Name of the embedding model to use
        
    Returns:
        Dict[int, bool]: Dictionary mapping document IDs to success status
    """
    results = {}
    
    # Get the appropriate vectorizer
    try:
        vectorizer = get_vectorizer(model)
    except Exception as e:
        logger.error(f"Failed to initialize vectorizer: {e}")
        return {doc_id: False for doc_id in document_ids}
    
    for doc_id in document_ids:
        results[doc_id] = process_document(doc_id, model)
    
    return results

def delete_document_embeddings(document_id: int) -> bool:
    """
    Delete all embeddings for a specific document.
    
    Args:
        document_id (int): Document ID
        
    Returns:
        bool: True if deletion successful, False otherwise
    """
    try:
        with get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                DELETE FROM doc_search.embeddings
                WHERE document_id = %s
                """,
                (document_id,)
            )
            affected_rows = cursor.rowcount
            logger.info(f"Deleted {affected_rows} embeddings for document ID: {document_id}")
            return affected_rows > 0
    except psycopg2.Error as e:
        logger.error(f"Error deleting embeddings: {e}")
        return False

def list_document_embeddings(document_id: int) -> List[Dict[str, Any]]:
    """
    List all embeddings for a specific document.
    
    Args:
        document_id (int): Document ID
        
    Returns:
        List[Dict[str, Any]]: List of embeddings metadata
    """
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, model_name, created_at
                FROM doc_search.embeddings
                WHERE document_id = %s
                """,
                (document_id,)
            )
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except psycopg2.Error as e:
        logger.error(f"Error listing embeddings: {e}")
        return [] 