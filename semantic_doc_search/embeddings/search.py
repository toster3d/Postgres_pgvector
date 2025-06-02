"""
Semantic search functionality
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import psycopg2

from semantic_doc_search.database.connection import get_cursor
from semantic_doc_search.embeddings.vectorizer import get_vectorizer, VectorizationError

logger = logging.getLogger(__name__)

def semantic_search(query: str, model: str = "sklearn", limit: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Perform semantic search to find similar documents.
    
    Args:
        query (str): Search query
        model (str): Embedding model to use
        limit (int): Maximum number of results
        threshold (float): Similarity threshold (0-1)
        
    Returns:
        List[Dict[str, Any]]: List of similar documents with similarity scores
    """
    # Generate query embedding
    try:
        vectorizer = get_vectorizer(model)
        query_embedding = vectorizer.vectorize(query)
    except (ImportError, VectorizationError) as e:
        logger.error(f"Error generating query embedding: {e}")
        return []
    
    # Search for similar documents
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    d.id, 
                    d.title, 
                    d.content, 
                    d.source, 
                    d.author,
                    1 - (e.embedding_vector <=> %s) as similarity
                FROM 
                    doc_search.embeddings e
                JOIN 
                    doc_search.documents d ON e.document_id = d.id
                WHERE 
                    e.model_name = %s
                    AND 1 - (e.embedding_vector <=> %s) > %s
                ORDER BY 
                    similarity DESC
                LIMIT %s
                """,
                (query_embedding, vectorizer.model_name, query_embedding, threshold, limit)
            )
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except psycopg2.Error as e:
        logger.error(f"Error during semantic search: {e}")
        return []

def hybrid_search(query: str, model: str = "sklearn", limit: int = 10, 
                 semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and full-text search.
    
    Args:
        query (str): Search query
        model (str): Embedding model to use
        limit (int): Maximum number of results
        semantic_weight (float): Weight for semantic search (0-1)
        
    Returns:
        List[Dict[str, Any]]: List of similar documents with combined scores
    """
    text_weight = 1.0 - semantic_weight
    
    # Generate query embedding
    try:
        vectorizer = get_vectorizer(model)
        query_embedding = vectorizer.vectorize(query)
    except (ImportError, VectorizationError) as e:
        logger.error(f"Error generating query embedding: {e}")
        return []
    
    # Perform hybrid search
    try:
        with get_cursor() as cursor:
            # Convert the query to tsquery format (replace spaces with &)
            ts_query = ' & '.join(query.split())
            
            cursor.execute(
                """
                SELECT 
                    d.id, 
                    d.title, 
                    d.content, 
                    d.source, 
                    d.author,
                    (%s * (1 - (e.embedding_vector <=> %s)) + 
                     %s * ts_rank(to_tsvector('english', d.content), to_tsquery('english', %s))) as combined_score,
                    (1 - (e.embedding_vector <=> %s)) as semantic_score,
                    ts_rank(to_tsvector('english', d.content), to_tsquery('english', %s)) as text_score
                FROM 
                    doc_search.embeddings e
                JOIN 
                    doc_search.documents d ON e.document_id = d.id
                WHERE 
                    e.model_name = %s
                    AND (
                        (1 - (e.embedding_vector <=> %s)) > 0.3
                        OR to_tsvector('english', d.content) @@ to_tsquery('english', %s)
                    )
                ORDER BY 
                    combined_score DESC
                LIMIT %s
                """,
                (semantic_weight, query_embedding, text_weight, ts_query,
                 query_embedding, ts_query, vectorizer.model_name, query_embedding, ts_query, limit)
            )
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except psycopg2.Error as e:
        logger.error(f"Error during hybrid search: {e}")
        return []

def document_recommendations(document_id: int, model: str = "sklearn", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find similar documents to a given document.
    
    Args:
        document_id (int): Document ID
        model (str): Embedding model to use
        limit (int): Maximum number of results
        
    Returns:
        List[Dict[str, Any]]: List of similar documents
    """
    try:
        with get_cursor() as cursor:
            # Get the document's embedding
            cursor.execute(
                """
                SELECT embedding_vector
                FROM doc_search.embeddings
                WHERE document_id = %s AND model_name LIKE %s
                LIMIT 1
                """,
                (document_id, f"%{model}%")
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No embedding found for document ID {document_id}")
                return []
            
            document_embedding = result[0]
            
            # Find similar documents
            cursor.execute(
                """
                SELECT 
                    d.id, 
                    d.title, 
                    d.content, 
                    d.source, 
                    d.author,
                    1 - (e.embedding_vector <=> %s) as similarity
                FROM 
                    doc_search.embeddings e
                JOIN 
                    doc_search.documents d ON e.document_id = d.id
                WHERE 
                    e.document_id != %s
                    AND e.model_name LIKE %s
                ORDER BY 
                    e.embedding_vector <=> %s
                LIMIT %s
                """,
                (document_embedding, document_id, f"%{model}%", document_embedding, limit)
            )
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except psycopg2.Error as e:
        logger.error(f"Error finding similar documents: {e}")
        return [] 