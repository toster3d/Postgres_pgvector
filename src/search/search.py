"""
Moduł wyszukiwania semantycznego i hybrydowego.
"""
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.models import Document, DocumentEmbedding
from src.embeddings.embeddings import get_embedding_generator


class SemanticSearch:
    """Klasa implementująca wyszukiwanie semantyczne i hybrydowe."""
    
    def __init__(self, db: Session, model_name: str = None):
        """
        Inicjalizuje wyszukiwarkę semantyczną.
        
        Args:
            db: Sesja bazodanowa
            model_name: Nazwa modelu do generowania embeddings
        """
        self.db = db
        self.embedding_generator = get_embedding_generator(model_name)
    
    def search_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Wyszukuje dokumenty semantycznie podobne do zapytania.
        
        Args:
            query: Zapytanie użytkownika
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista dokumentów z ich podobieństwem
        """
        # Generuj embedding dla zapytania
        query_embedding = self.embedding_generator.get_embedding(query)
        
        # Utwórz zapytanie SQL z wykorzystaniem pgvector
        query_str = f"""
        SELECT 
            d.id, 
            d.title, 
            d.content,
            d.category_id, 
            d.metadata,
            d.created_at,
            d.updated_at,
            de.embedding <=> :query_embedding AS similarity
        FROM 
            documents d
        JOIN 
            document_embeddings de ON d.id = de.document_id
        WHERE
            de.model_name = :model_name
        ORDER BY 
            similarity ASC
        LIMIT :limit
        """
        
        # Wykonaj zapytanie
        results = self.db.execute(
            text(query_str),
            {
                "query_embedding": str(query_embedding),
                "model_name": self.embedding_generator.model_name,
                "limit": limit
            }
        )
        
        # Przetwórz wyniki
        documents = []
        for row in results:
            doc = {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "category_id": row.category_id,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "similarity": 1.0 - float(row.similarity)  # Konwertuj odległość na podobieństwo
            }
            documents.append(doc)
        
        return documents
    
    def search_fulltext(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Wyszukuje dokumenty przy użyciu PostgreSQL full-text search.
        
        Args:
            query: Zapytanie użytkownika
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista dokumentów z ich rankingiem
        """
        # Utwórz zapytanie SQL z wykorzystaniem full-text search PostgreSQL
        query_str = f"""
        SELECT 
            d.id, 
            d.title, 
            d.content,
            d.category_id, 
            d.metadata,
            d.created_at,
            d.updated_at,
            ts_rank(to_tsvector('english', d.title || ' ' || d.content), plainto_tsquery('english', :query)) AS rank
        FROM 
            documents d
        WHERE 
            to_tsvector('english', d.title || ' ' || d.content) @@ plainto_tsquery('english', :query)
        ORDER BY 
            rank DESC
        LIMIT :limit
        """
        
        # Wykonaj zapytanie
        results = self.db.execute(
            text(query_str),
            {
                "query": query,
                "limit": limit
            }
        )
        
        # Przetwórz wyniki
        documents = []
        for row in results:
            doc = {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "category_id": row.category_id,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "rank": float(row.rank)
            }
            documents.append(doc)
        
        return documents
    
    def search_hybrid(self, query: str, limit: int = 10, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Wyszukuje dokumenty przy użyciu hybrydowego podejścia łączącego wyszukiwanie semantyczne i full-text.
        
        Args:
            query: Zapytanie użytkownika
            limit: Maksymalna liczba wyników
            semantic_weight: Waga dla wyników wyszukiwania semantycznego (0.0-1.0)
            
        Returns:
            Lista dokumentów z ich złożonym rankingiem
        """
        # Sprawdź poprawność wagi
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight musi być wartością od 0.0 do 1.0")
        
        # Ustaw wagę dla full-text search
        fulltext_weight = 1.0 - semantic_weight
        
        # Utwórz embedding dla zapytania
        query_embedding = self.embedding_generator.get_embedding(query)
        
        # Utwórz zapytanie SQL łączące oba podejścia
        query_str = f"""
        SELECT 
            d.id, 
            d.title, 
            d.content,
            d.category_id, 
            d.metadata,
            d.created_at,
            d.updated_at,
            (
                :semantic_weight * (1.0 - (de.embedding <=> :query_embedding)) + 
                :fulltext_weight * ts_rank(to_tsvector('english', d.title || ' ' || d.content), plainto_tsquery('english', :query))
            ) AS hybrid_score
        FROM 
            documents d
        JOIN 
            document_embeddings de ON d.id = de.document_id
        WHERE
            de.model_name = :model_name
        ORDER BY 
            hybrid_score DESC
        LIMIT :limit
        """
        
        # Wykonaj zapytanie
        results = self.db.execute(
            text(query_str),
            {
                "query_embedding": str(query_embedding),
                "query": query,
                "model_name": self.embedding_generator.model_name,
                "semantic_weight": semantic_weight,
                "fulltext_weight": fulltext_weight,
                "limit": limit
            }
        )
        
        # Przetwórz wyniki
        documents = []
        for row in results:
            doc = {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "category_id": row.category_id,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "score": float(row.hybrid_score)
            }
            documents.append(doc)
        
        return documents
    
    def get_similar_documents(self, document_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Znajduje dokumenty podobne do podanego dokumentu.
        
        Args:
            document_id: ID dokumentu, dla którego szukamy podobnych
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista podobnych dokumentów z ich podobieństwem
        """
        # Sprawdź, czy dokument istnieje
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError(f"Dokument o ID {document_id} nie istnieje")
        
        # Sprawdź, czy dokument ma embedding
        embedding = self.db.query(DocumentEmbedding).filter(
            DocumentEmbedding.document_id == document_id,
            DocumentEmbedding.model_name == self.embedding_generator.model_name
        ).first()
        
        if not embedding:
            # Wygeneruj embedding dla dokumentu, jeśli nie istnieje
            text_content = f"{document.title} {document.content}"
            vector = self.embedding_generator.get_embedding(text_content)
            
            # Utwórz nowy embedding
            embedding = DocumentEmbedding(
                document_id=document_id,
                model_name=self.embedding_generator.model_name
            )
            embedding.set_embedding_vector(vector)
            
            # Zapisz embedding
            self.db.add(embedding)
            self.db.commit()
        
        # Pobierz wektor embedding dokumentu
        document_embedding = embedding.get_embedding_vector()
        
        # Utwórz zapytanie SQL wyszukujące podobne dokumenty
        query_str = f"""
        SELECT 
            d.id, 
            d.title, 
            d.content,
            d.category_id, 
            d.metadata,
            d.created_at,
            d.updated_at,
            de.embedding <=> :document_embedding AS similarity
        FROM 
            documents d
        JOIN 
            document_embeddings de ON d.id = de.document_id
        WHERE
            de.model_name = :model_name
            AND d.id != :document_id
        ORDER BY 
            similarity ASC
        LIMIT :limit
        """
        
        # Wykonaj zapytanie
        results = self.db.execute(
            text(query_str),
            {
                "document_embedding": str(document_embedding),
                "model_name": self.embedding_generator.model_name,
                "document_id": document_id,
                "limit": limit
            }
        )
        
        # Przetwórz wyniki
        documents = []
        for row in results:
            doc = {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "category_id": row.category_id,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "similarity": 1.0 - float(row.similarity)  # Konwertuj odległość na podobieństwo
            }
            documents.append(doc)
        
        return documents 