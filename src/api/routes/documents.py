"""
Endpointy API dla dokumentów.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.database.models import Document, DocumentEmbedding
from src.embeddings.embeddings import get_embedding_generator
from src.api.models import DocumentCreate, DocumentResponse, DocumentUpdate

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """
    Tworzy nowy dokument i generuje jego embedding.
    """
    # Utwórz nowy dokument
    db_document = Document(
        title=document.title,
        content=document.content,
        category_id=document.category_id,
        metadata=document.metadata or {}
    )
    
    # Dodaj dokument do bazy
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    # Generuj embedding dla dokumentu
    embedding_generator = get_embedding_generator()
    content_text = f"{db_document.title} {db_document.content}"
    embedding_vector = embedding_generator.get_embedding(content_text)
    
    # Utwórz nowy embedding
    db_embedding = DocumentEmbedding(
        document_id=db_document.id,
        model_name=embedding_generator.model_name
    )
    db_embedding.set_embedding_vector(embedding_vector)
    
    # Dodaj embedding do bazy
    db.add(db_embedding)
    db.commit()
    
    return db_document


@router.get("/", response_model=List[DocumentResponse])
def read_documents(
    skip: int = 0,
    limit: int = 100,
    category_id: Optional[int] = None,
    db: Session = Depends(get_db)
) -> List[DocumentResponse]:
    """
    Pobiera listę dokumentów z opcjonalnym filtrowaniem po kategorii.
    """
    query = db.query(Document)
    
    # Filtruj po kategorii, jeśli podano
    if category_id is not None:
        query = query.filter(Document.category_id == category_id)
    
    # Paginacja wyników
    documents = query.offset(skip).limit(limit).all()
    
    return documents


@router.get("/{document_id}", response_model=DocumentResponse)
def read_document(
    document_id: int,
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """
    Pobiera dokument po ID.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dokument o ID {document_id} nie został znaleziony"
        )
    
    return document


@router.put("/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: int,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """
    Aktualizuje dokument i jego embedding.
    """
    # Pobierz dokument
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dokument o ID {document_id} nie został znaleziony"
        )
    
    # Aktualizuj pola dokumentu
    update_data = document_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_document, key, value)
    
    # Zapisz zmiany
    db.commit()
    db.refresh(db_document)
    
    # Jeśli zmieniono treść lub tytuł, zaktualizuj embedding
    if "title" in update_data or "content" in update_data:
        # Generuj nowy embedding
        embedding_generator = get_embedding_generator()
        content_text = f"{db_document.title} {db_document.content}"
        embedding_vector = embedding_generator.get_embedding(content_text)
        
        # Znajdź istniejący embedding lub utwórz nowy
        db_embedding = db.query(DocumentEmbedding).filter(
            DocumentEmbedding.document_id == document_id,
            DocumentEmbedding.model_name == embedding_generator.model_name
        ).first()
        
        if db_embedding:
            # Aktualizuj istniejący embedding
            db_embedding.set_embedding_vector(embedding_vector)
        else:
            # Utwórz nowy embedding
            db_embedding = DocumentEmbedding(
                document_id=document_id,
                model_name=embedding_generator.model_name
            )
            db_embedding.set_embedding_vector(embedding_vector)
            db.add(db_embedding)
        
        # Zapisz zmiany
        db.commit()
    
    return db_document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
) -> None:
    """
    Usuwa dokument i jego embeddings.
    """
    # Pobierz dokument
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dokument o ID {document_id} nie został znaleziony"
        )
    
    # Usuń dokument (embeddings zostaną usunięte kaskadowo)
    db.delete(db_document)
    db.commit() 