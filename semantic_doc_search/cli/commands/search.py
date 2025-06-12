"""
Komendy CLI do wyszukiwania dokumentów.
"""

import logging
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from semantic_doc_search.core.search import search_manager, SearchResultSet
from semantic_doc_search.core.models import SearchResult, HybridSearchResult
from semantic_doc_search.config.settings import config

console = Console()
logger = logging.getLogger(__name__)


@click.group(name="search")
def search_group():
    """Wyszukiwanie dokumentów w systemie."""
    pass


@search_group.command()
@click.argument("query", required=True)
@click.option(
    "--limit", "-l",
    default=None,
    type=int,
    help="Liczba wyników (domyślnie z konfiguracji)"
)
@click.option(
    "--min-score",
    default=0.0,
    type=float,
    help="Minimalny wynik podobieństwa"
)
@click.option(
    "--source",
    help="Filtruj po źródle dokumentów"
)
@click.option(
    "--show-content/--no-content",
    default=False,
    help="Czy pokazać fragmenty treści"
)
@click.option(
    "--export",
    type=click.Choice(['json', 'csv']),
    help="Eksportuj wyniki do formatu"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Plik wyjściowy dla eksportu"
)
def text(
    query: str,
    limit: Optional[int],
    min_score: float,
    source: Optional[str],
    show_content: bool,
    export: Optional[str],
    output: Optional[str]
):
    """Wyszukiwanie pełnotekstowe (full-text search)."""
    
    if limit is None:
        limit = config.search.default_limit
    
    console.print(f"🔍 Wyszukiwanie pełnotekstowe: '{query}'", style="bold blue")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🔍 Wyszukiwanie...", total=None)
            
            # Wykonaj wyszukiwanie
            result_set = search_manager.search(
                query=query,
                search_type="text",
                limit=limit,
                min_score=min_score,
                source_filter=source
            )
            
            progress.update(task, description="✅ Wyszukiwanie zakończone")
        
        # Wyświetl wyniki
        _display_search_results(result_set, show_content)
        
        # Eksport jeśli requested
        if export:
            _export_results(result_set, export, output)
            
    except Exception as e:
        console.print(f"❌ Błąd podczas wyszukiwania: {e}", style="red")
        logger.exception("Error in text search")


@search_group.command()
@click.argument("query", required=True)
@click.option(
    "--limit", "-l",
    default=None,
    type=int,
    help="Liczba wyników (domyślnie z konfiguracji)"
)
@click.option(
    "--model", "-m",
    help="Model embeddings do użycia"
)
@click.option(
    "--min-score",
    default=0.0,
    type=float,
    help="Minimalny wynik podobieństwa"
)
@click.option(
    "--distance-metric",
    type=click.Choice(['cosine', 'l2', 'euclidean', 'dot_product']),
    default='cosine',
    help="Metryka odległości"
)
@click.option(
    "--source",
    help="Filtruj po źródle dokumentów"
)
@click.option(
    "--show-content/--no-content",
    default=False,
    help="Czy pokazać fragmenty treści"
)
@click.option(
    "--export",
    type=click.Choice(['json', 'csv']),
    help="Eksportuj wyniki do formatu"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Plik wyjściowy dla eksportu"
)
def semantic(
    query: str,
    limit: Optional[int],
    model: Optional[str],
    min_score: float,
    distance_metric: str,
    source: Optional[str],
    show_content: bool,
    export: Optional[str],
    output: Optional[str]
):
    """Wyszukiwanie semantyczne używając embeddings."""
    
    if limit is None:
        limit = config.search.default_limit
    
    if model is None:
        model = config.embedding.default_model
    
    console.print(f"🧠 Wyszukiwanie semantyczne: '{query}'", style="bold blue")
    console.print(f"📚 Model: {model}")
    console.print(f"📏 Metryka: {distance_metric}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🧠 Generowanie embeddings...", total=None)
            
            # Wykonaj wyszukiwanie
            result_set = search_manager.search(
                query=query,
                search_type="semantic",
                limit=limit,
                model_name=model,
                min_score=min_score,
                distance_metric=distance_metric,
                source_filter=source
            )
            
            progress.update(task, description="✅ Wyszukiwanie zakończone")
        
        # Wyświetl wyniki
        _display_search_results(result_set, show_content)
        
        # Eksport jeśli requested
        if export:
            _export_results(result_set, export, output)
            
    except Exception as e:
        console.print(f"❌ Błąd podczas wyszukiwania: {e}", style="red")
        logger.exception("Error in semantic search")


@search_group.command()
@click.argument("query", required=True)
@click.option(
    "--limit", "-l",
    default=None,
    type=int,
    help="Liczba wyników (domyślnie z konfiguracji)"
)
@click.option(
    "--semantic-weight",
    default=None,
    type=float,
    help="Waga wyszukiwania semantycznego (0.0-1.0, domyślnie z konfiguracji)"
)
@click.option(
    "--model", "-m",
    help="Model embeddings do użycia"
)
@click.option(
    "--min-score",
    default=0.0,
    type=float,
    help="Minimalny wynik podobieństwa"
)
@click.option(
    "--source",
    help="Filtruj po źródle dokumentów"
)
@click.option(
    "--show-content/--no-content",
    default=False,
    help="Czy pokazać fragmenty treści"
)
@click.option(
    "--export",
    type=click.Choice(['json', 'csv']),
    help="Eksportuj wyniki do formatu"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Plik wyjściowy dla eksportu"
)
def hybrid(
    query: str,
    limit: Optional[int],
    semantic_weight: Optional[float],
    model: Optional[str],
    min_score: float,
    source: Optional[str],
    show_content: bool,
    export: Optional[str],
    output: Optional[str]
):
    """Wyszukiwanie hybrydowe (semantyczne + pełnotekstowe)."""
    
    if limit is None:
        limit = config.search.default_limit
    
    if semantic_weight is None:
        semantic_weight = config.search.default_semantic_weight
    
    if model is None:
        model = config.embedding.default_model
    
    console.print(f"🔄 Wyszukiwanie hybrydowe: '{query}'", style="bold blue")
    console.print(f"📚 Model: {model}")
    console.print(f"⚖️  Waga semantyczna: {semantic_weight:.1f}")
    console.print(f"📝 Waga pełnotekstowa: {1 - semantic_weight:.1f}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🔄 Hybrydowe wyszukiwanie...", total=None)
            
            # Wykonaj wyszukiwanie
            result_set = search_manager.search(
                query=query,
                search_type="hybrid",
                limit=limit,
                semantic_weight=semantic_weight,
                model_name=model,
                min_score=min_score,
                source_filter=source
            )
            
            progress.update(task, description="✅ Wyszukiwanie zakończone")
        
        # Wyświetl wyniki
        _display_hybrid_results(result_set, show_content)
        
        # Eksport jeśli requested
        if export:
            _export_results(result_set, export, output)
            
    except Exception as e:
        console.print(f"❌ Błąd podczas wyszukiwania: {e}", style="red")
        logger.exception("Error in hybrid search")


@search_group.command()
@click.argument("document_id", type=int)
@click.option(
    "--limit", "-l",
    default=10,
    type=int,
    help="Liczba podobnych dokumentów"
)
@click.option(
    "--model", "-m",
    help="Model embeddings do użycia"
)
@click.option(
    "--show-content/--no-content",
    default=False,
    help="Czy pokazać fragmenty treści"
)
@click.option(
    "--export",
    type=click.Choice(['json', 'csv']),
    help="Eksportuj wyniki do formatu"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Plik wyjściowy dla eksportu"
)
def recommend(
    document_id: int,
    limit: int,
    model: Optional[str],
    show_content: bool,
    export: Optional[str],
    output: Optional[str]
):
    """Znajdź dokumenty podobne do podanego dokumentu."""
    
    if model is None:
        model = config.embedding.default_model
    
    console.print(f"💡 Rekomendacje dla dokumentu ID: {document_id}", style="bold blue")
    console.print(f"📚 Model: {model}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("💡 Szukanie podobnych dokumentów...", total=None)
            
            # Wykonaj wyszukiwanie rekomendacji
            results = search_manager.recommend_similar(
                document_id=document_id,
                limit=limit,
                model_name=model
            )
            
            progress.update(task, description="✅ Rekomendacje gotowe")
        
        # Wyświetl wyniki
        if not results:
            console.print("📭 Nie znaleziono podobnych dokumentów", style="yellow")
            return
        
        console.print(f"\n🎯 Znaleziono {len(results)} podobnych dokumentów:")
        console.print(f"⏱️  Czas wyszukiwania: {0:.3f}s")  # TODO: dodać pomiar czasu
        console.print()
        
        # Tabela wyników
        table = Table()
        table.add_column("Pozycja", style="cyan", width=6)
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Tytuł", style="bold")
        table.add_column("Podobieństwo", style="green", width=12)
        
        if show_content:
            table.add_column("Fragment", style="dim")
        
        for i, result in enumerate(results, 1):
            row = [
                str(i),
                str(result.document.id),
                result.document.title[:60] + ("..." if len(result.document.title) > 60 else ""),
                f"{result.score:.3f}"
            ]
            
            if show_content and result.snippet:
                snippet = result.snippet[:100] + ("..." if len(result.snippet) > 100 else "")
                row.append(snippet)
            
            table.add_row(*row)
        
        console.print(table)
        
        # Eksport jeśli requested
        if export:
            # Stwórz sztuczny result_set dla eksportu
            from semantic_doc_search.core.search import SearchResultSet, SearchParams
            result_set = SearchResultSet(
                results=results,
                total_found=len(results),
                search_type="recommendation",
                search_time=0.0,
                query=f"recommend:{document_id}",
                params=SearchParams(query=f"recommend:{document_id}"),
                metadata={"reference_document_id": document_id}
            )
            _export_results(result_set, export, output)
            
    except Exception as e:
        console.print(f"❌ Błąd podczas wyszukiwania rekomendacji: {e}", style="red")
        logger.exception("Error in recommendation search")


def _display_search_results(result_set: SearchResultSet, show_content: bool) -> None:
    """Wyświetla wyniki wyszukiwania."""
    
    if not result_set.results:
        console.print("📭 Nie znaleziono dokumentów", style="yellow")
        return
    
    # Podsumowanie
    console.print(f"\n🎯 Znaleziono {result_set.total_found} dokumentów:")
    console.print(f"⏱️  Czas wyszukiwania: {result_set.search_time:.3f}s")
    console.print(f"🔧 Typ wyszukiwania: {result_set.search_type}")
    console.print()
    
    # Tabela wyników
    table = Table()
    table.add_column("Pozycja", style="cyan", width=6)
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Tytuł", style="bold")
    table.add_column("Wynik", style="green", width=10)
    
    if show_content:
        table.add_column("Fragment", style="dim")
    
    for i, result in enumerate(result_set.results, 1):
        row = [
            str(i),
            str(result.document.id),
            result.document.title[:60] + ("..." if len(result.document.title) > 60 else ""),
            f"{result.score:.3f}"
        ]
        
        if show_content and result.snippet:
            snippet = result.snippet[:100] + ("..." if len(result.snippet) > 100 else "")
            row.append(snippet)
        
        table.add_row(*row)
    
    console.print(table)


def _display_hybrid_results(result_set: SearchResultSet, show_content: bool) -> None:
    """Wyświetla wyniki hybrydowego wyszukiwania."""
    
    if not result_set.results:
        console.print("📭 Nie znaleziono dokumentów", style="yellow")
        return
    
    # Podsumowanie
    console.print(f"\n🎯 Znaleziono {result_set.total_found} dokumentów:")
    console.print(f"⏱️  Czas wyszukiwania: {result_set.search_time:.3f}s")
    console.print()
    
    # Tabela wyników z dodatkowymi kolumnami dla hybrydowego wyszukiwania
    table = Table()
    table.add_column("Poz.", style="cyan", width=4)
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Tytuł", style="bold")
    table.add_column("Semantyczny", style="blue", width=11)
    table.add_column("Tekstowy", style="magenta", width=10)
    table.add_column("Łączny", style="green", width=10)
    
    if show_content:
        table.add_column("Fragment", style="dim")
    
    for i, result in enumerate(result_set.results, 1):
        if isinstance(result, HybridSearchResult):
            row = [
                str(i),
                str(result.document.id),
                result.document.title[:40] + ("..." if len(result.document.title) > 40 else ""),
                f"{result.semantic_score:.3f}",
                f"{result.text_score:.3f}",
                f"{result.combined_score:.3f}"
            ]
            
            if show_content and result.snippet:
                snippet = result.snippet[:80] + ("..." if len(result.snippet) > 80 else "")
                row.append(snippet)
        else:
            # Fallback dla standardowych wyników
            row = [
                str(i),
                str(result.document.id),
                result.document.title[:40] + ("..." if len(result.document.title) > 40 else ""),
                "-",
                "-",
                f"{result.score:.3f}"
            ]
            
            if show_content and result.snippet:
                snippet = result.snippet[:80] + ("..." if len(result.snippet) > 80 else "")
                row.append(snippet)
        
        table.add_row(*row)
    
    console.print(table)


def _export_results(result_set: SearchResultSet, format: str, output: Optional[str]) -> None:
    """Eksportuje wyniki wyszukiwania."""
    
    if not result_set.results:
        console.print("⚠️  Brak wyników do eksportu", style="yellow")
        return
    
    # Przygotuj dane do eksportu
    data = []
    for i, result in enumerate(result_set.results, 1):
        if isinstance(result, HybridSearchResult):
            row = {
                "position": i,
                "document_id": result.document.id,
                "title": result.document.title,
                "source": result.document.source,
                "semantic_score": result.semantic_score,
                "text_score": result.text_score,
                "combined_score": result.combined_score,
                "snippet": result.snippet,
                "search_type": result_set.search_type,
                "query": result_set.query
            }
        else:
            row = {
                "position": i,
                "document_id": result.document.id,
                "title": result.document.title,
                "source": result.document.source,
                "score": result.score,
                "snippet": result.snippet,
                "search_type": result.search_type,
                "query": result_set.query
            }
        data.append(row)
    
    # Określ nazwę pliku
    if output is None:
        timestamp = result_set.metadata.get('timestamp', 'export')
        output = f"search_results_{timestamp}.{format}"
    
    output_path = Path(output)
    
    try:
        if format == 'json':
            # Eksport JSON
            export_data = {
                "search_metadata": {
                    "query": result_set.query,
                    "search_type": result_set.search_type,
                    "total_found": result_set.total_found,
                    "search_time": result_set.search_time,
                    "params": result_set.params.__dict__
                },
                "results": data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        elif format == 'csv':
            # Eksport CSV
            import csv
            
            if data:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        
        console.print(f"✅ Wyniki wyeksportowane do: {output_path}", style="green")
        
    except Exception as e:
        console.print(f"❌ Błąd podczas eksportu: {e}", style="red")
        logger.exception("Error exporting results")