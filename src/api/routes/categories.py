"""
Endpointy API dla kategorii.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.database.models import Category
from src.api.models import CategoryCreate, CategoryResponse

router = APIRouter(
    prefix="/categories",
    tags=["categories"],
)


@router.post("/", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
def create_category(
    category: CategoryCreate,
    db: Session = Depends(get_db)
) -> CategoryResponse:
    """
    Tworzy nową kategorię.
    """
    # Sprawdź, czy kategoria o podanej nazwie już istnieje
    db_category = db.query(Category).filter(Category.name == category.name).first()
    if db_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Kategoria o nazwie '{category.name}' już istnieje"
        )
    
    # Utwórz nową kategorię
    db_category = Category(
        name=category.name,
        description=category.description
    )
    
    # Dodaj kategorię do bazy
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    
    return db_category


@router.get("/", response_model=List[CategoryResponse])
def read_categories(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[CategoryResponse]:
    """
    Pobiera listę kategorii.
    """
    categories = db.query(Category).offset(skip).limit(limit).all()
    return categories


@router.get("/{category_id}", response_model=CategoryResponse)
def read_category(
    category_id: int,
    db: Session = Depends(get_db)
) -> CategoryResponse:
    """
    Pobiera kategorię po ID.
    """
    category = db.query(Category).filter(Category.id == category_id).first()
    if category is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Kategoria o ID {category_id} nie została znaleziona"
        )
    
    return category


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_category(
    category_id: int,
    db: Session = Depends(get_db)
) -> None:
    """
    Usuwa kategorię.
    """
    # Pobierz kategorię
    db_category = db.query(Category).filter(Category.id == category_id).first()
    if db_category is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Kategoria o ID {category_id} nie została znaleziona"
        )
    
    # Usuń kategorię
    db.delete(db_category)
    db.commit() 