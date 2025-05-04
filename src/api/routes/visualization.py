"""
Endpointy API dla wizualizacji.
"""
import os
import tempfile
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.visualization.visualization import EmbeddingVisualizer

router = APIRouter(
    prefix="/visualization",
    tags=["visualization"],
)


@router.get("/embeddings/matplotlib")
def visualize_embeddings_matplotlib(
    model_name: str,
    category_id: Optional[int] = None,
    method: str = "pca",
    db: Session = Depends(get_db)
):
    """
    Generuje i zwraca wizualizację embeddings w formacie PNG przy użyciu Matplotlib.
    """
    try:
        # Utwórz wizualizator
        visualizer = EmbeddingVisualizer(db=db)
        
        # Utwórz tymczasowy plik dla obrazu
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        
        # Generuj wizualizację
        visualizer.visualize_embeddings_matplotlib(
            model_name=model_name,
            category_id=category_id,
            method=method,
            save_path=temp_path
        )
        
        # Zwróć obraz jako odpowiedź
        return FileResponse(
            path=temp_path,
            media_type="image/png",
            filename=f"embeddings_{method}_{model_name}.png"
        )
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas generowania wizualizacji: {str(e)}"
        )


@router.get("/embeddings/plotly", response_class=HTMLResponse)
def visualize_embeddings_plotly(
    model_name: str,
    category_id: Optional[int] = None,
    method: str = "tsne",
    db: Session = Depends(get_db)
):
    """
    Generuje i zwraca interaktywną wizualizację embeddings w formacie HTML przy użyciu Plotly.
    """
    try:
        # Utwórz wizualizator
        visualizer = EmbeddingVisualizer(db=db)
        
        # Utwórz tymczasowy plik dla HTML
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            temp_path = tmp.name
        
        # Generuj wizualizację
        fig = visualizer.visualize_embeddings_plotly(
            model_name=model_name,
            category_id=category_id,
            method=method,
            save_path=temp_path
        )
        
        # Odczytaj zawartość HTML
        with open(temp_path, "r") as f:
            html_content = f.read()
        
        # Usuń plik tymczasowy
        os.unlink(temp_path)
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas generowania wizualizacji: {str(e)}"
        ) 