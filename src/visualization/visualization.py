"""
Moduł wizualizacji wektorów embeddings.
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session

from src.database.models import Document, DocumentEmbedding


class EmbeddingVisualizer:
    """Klasa do wizualizacji wektorów embeddings dokumentów."""
    
    def __init__(self, db: Session):
        """
        Inicjalizuje wizualizator embeddings.
        
        Args:
            db: Sesja bazodanowa
        """
        self.db = db
    
    def _get_embeddings_data(self, model_name: str, category_id: Optional[int] = None) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Pobiera dane embeddings i metadane dokumentów.
        
        Args:
            model_name: Nazwa modelu embeddings
            category_id: ID kategorii do filtrowania (opcjonalnie)
            
        Returns:
            Krotka (lista wektorów embeddings, lista metadanych dokumentów)
        """
        # Pobierz dokumenty i ich embeddings
        query = (
            self.db.query(Document, DocumentEmbedding)
            .join(DocumentEmbedding, Document.id == DocumentEmbedding.document_id)
            .filter(DocumentEmbedding.model_name == model_name)
        )
        
        # Opcjonalnie filtruj po kategorii
        if category_id is not None:
            query = query.filter(Document.category_id == category_id)
        
        # Wykonaj zapytanie
        results = query.all()
        
        # Przygotuj dane
        embeddings = []
        metadata = []
        
        for doc, emb in results:
            # Pobierz wektor embeddings
            vector = json.loads(emb.embedding) if isinstance(emb.embedding, str) else emb.get_embedding_vector()
            
            # Dodaj do list
            embeddings.append(vector)
            metadata.append({
                "id": doc.id,
                "title": doc.title,
                "category_id": doc.category_id,
                "category_name": doc.category.name if doc.category else None
            })
        
        return embeddings, metadata
    
    def reduce_dimensions_pca(self, embeddings: List[List[float]], n_components: int = 2) -> np.ndarray:
        """
        Redukuje wymiary wektorów embeddings przy użyciu PCA.
        
        Args:
            embeddings: Lista wektorów embeddings
            n_components: Liczba wymiarów wynikowych
            
        Returns:
            Tablica NumPy zredukowanych wektorów
        """
        # Konwertuj do NumPy
        embeddings_array = np.array(embeddings)
        
        # Sprawdź, czy mamy wystarczającą ilość danych
        if len(embeddings) < n_components:
            raise ValueError(f"Za mało próbek ({len(embeddings)}) dla {n_components} komponentów")
        
        # Redukuj wymiary za pomocą PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings_array)
        
        # Dodatkowe informacje
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"Wariancja wyjaśniona przez {n_components} komponenty: {explained_variance:.2%}")
        
        return reduced
    
    def reduce_dimensions_tsne(self, embeddings: List[List[float]], n_components: int = 2, perplexity: int = 30) -> np.ndarray:
        """
        Redukuje wymiary wektorów embeddings przy użyciu t-SNE.
        
        Args:
            embeddings: Lista wektorów embeddings
            n_components: Liczba wymiarów wynikowych
            perplexity: Parametr perplexity dla t-SNE
            
        Returns:
            Tablica NumPy zredukowanych wektorów
        """
        # Konwertuj do NumPy
        embeddings_array = np.array(embeddings)
        
        # Sprawdź, czy perplexity nie jest za duże
        if len(embeddings) < perplexity:
            perplexity = len(embeddings) - 1
            print(f"Dostosowano perplexity do {perplexity} ze względu na małą liczbę próbek")
        
        # Redukuj wymiary za pomocą t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings_array)
        
        return reduced
    
    def visualize_embeddings_matplotlib(
        self, 
        model_name: str, 
        category_id: Optional[int] = None,
        method: str = "pca",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wizualizuje embeddings przy użyciu Matplotlib.
        
        Args:
            model_name: Nazwa modelu embeddings
            category_id: ID kategorii do filtrowania (opcjonalnie)
            method: Metoda redukcji wymiarowości ('pca' lub 'tsne')
            save_path: Ścieżka do zapisania rysunku (opcjonalnie)
            
        Returns:
            Obiekt Figure Matplotlib
        """
        # Pobierz dane
        embeddings, metadata = self._get_embeddings_data(model_name, category_id)
        
        if not embeddings:
            raise ValueError("Brak danych embeddings do wizualizacji")
        
        # Redukuj wymiary
        if method.lower() == "pca":
            reduced = self.reduce_dimensions_pca(embeddings)
        elif method.lower() == "tsne":
            reduced = self.reduce_dimensions_tsne(embeddings)
        else:
            raise ValueError("Nieobsługiwana metoda redukcji wymiarowości. Użyj 'pca' lub 'tsne'")
        
        # Utwórz DataFrame dla łatwiejszej wizualizacji
        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "id": [m["id"] for m in metadata],
            "title": [m["title"] for m in metadata],
            "category_id": [m["category_id"] for m in metadata],
            "category_name": [m["category_name"] for m in metadata]
        })
        
        # Utwórz wykres
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Grupuj po kategorii
        categories = df["category_name"].unique()
        for category in categories:
            subset = df[df["category_name"] == category]
            ax.scatter(subset["x"], subset["y"], label=category, alpha=0.7)
        
        # Dodaj etykiety dla wybranych punktów
        for i, row in df.iterrows():
            ax.annotate(row["title"][:20], (row["x"], row["y"]), fontsize=8, alpha=0.7)
        
        # Dodaj legendę i tytuł
        ax.legend(title="Kategoria")
        ax.set_title(f"Wizualizacja embeddings dokumentów ({method.upper()})")
        ax.set_xlabel(f"{method.upper()} Komponent 1")
        ax.set_ylabel(f"{method.upper()} Komponent 2")
        
        # Dostosuj układ
        plt.tight_layout()
        
        # Zapisz rysunek, jeśli podano ścieżkę
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def visualize_embeddings_plotly(
        self, 
        model_name: str, 
        category_id: Optional[int] = None,
        method: str = "pca",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Wizualizuje embeddings przy użyciu Plotly.
        
        Args:
            model_name: Nazwa modelu embeddings
            category_id: ID kategorii do filtrowania (opcjonalnie)
            method: Metoda redukcji wymiarowości ('pca' lub 'tsne')
            save_path: Ścieżka do zapisania rysunku jako HTML (opcjonalnie)
            
        Returns:
            Obiekt Figure Plotly
        """
        # Pobierz dane
        embeddings, metadata = self._get_embeddings_data(model_name, category_id)
        
        if not embeddings:
            raise ValueError("Brak danych embeddings do wizualizacji")
        
        # Redukuj wymiary
        if method.lower() == "pca":
            reduced = self.reduce_dimensions_pca(embeddings)
        elif method.lower() == "tsne":
            reduced = self.reduce_dimensions_tsne(embeddings)
        else:
            raise ValueError("Nieobsługiwana metoda redukcji wymiarowości. Użyj 'pca' lub 'tsne'")
        
        # Przygotuj DataFrame
        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "id": [m["id"] for m in metadata],
            "title": [m["title"] for m in metadata],
            "category_id": [m["category_id"] for m in metadata],
            "category_name": [m["category_name"] if m["category_name"] else "Brak kategorii" for m in metadata]
        })
        
        # Utwórz wykres Plotly
        fig = px.scatter(
            df, 
            x="x", 
            y="y", 
            color="category_name",
            hover_data=["id", "title"],
            title=f"Wizualizacja embeddings dokumentów ({method.upper()})",
            labels={"x": f"{method.upper()} Komponent 1", "y": f"{method.upper()} Komponent 2", "category_name": "Kategoria"}
        )
        
        # Dodatkowe formatowanie
        fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode="markers"))
        
        # Zapisz jako HTML, jeśli podano ścieżkę
        if save_path:
            fig.write_html(save_path)
        
        return fig 