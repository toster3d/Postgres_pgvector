# semantic_doc_search/cli/search_commands.py
"""
Moduł CLI do wyszukiwania dokumentów.
Obsługuje semantyczne, pełnotekstowe i hybrydowe wyszukiwanie.
"""

import click
import json
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt

from semantic_doc_search.core.database import DatabaseManager
from semantic_doc_search.core.embeddings import EmbeddingProvider
from semantic_doc_search.core.search import SearchEngine

console = Console()

@click.group()
def search_group():
    """🔍 Wyszukiwanie dokumentów w systemie"""
    pass

@search_group.command("text")
@click.argument("query", type=str)
@click.option("--limit", "-l", default=10, help="Liczba wyników")
@click.option("--min-score", default=0.1, help="Minimalny score podobieństwa")
@click.option("--show-content/--no-content", default=False, help="Pokaż fragmenty treści")
@click.option("--export", type=click.Choice(["json", "csv"]), help="Eksportuj wyniki")
@click.option("--output", "-o", help="Plik wyjściowy dla eksportu")
@click.pass_context
def text_search(ctx, query: str, limit: int, min_score: float, show_content: bool, 
                export: Optional[str], output: Optional[str]):
    """📝 Wyszukiwanie pełnotekstowe (PostgreSQL ts_search)"""
    
    db = ctx.obj['db']
    search_engine = SearchEngine(db)
    
    with console.status(f"[bold green]Wyszukiwanie: '{query}'..."):
        try:
            results = search_engine.text_search(
                query=query,
                limit=limit,
                min_score=min_score
            )
            
            if not results:
                console.print("[yellow]🔍 Nie znaleziono dokumentów pasujących do zapytania[/yellow]")
                return
            
            # Wyświetlenie wyników
            _display_search_results(results, query, "Wyszukiwanie Pełnotekstowe", show_content)
            
            # Eksport jeśli wymagany
            if export:
                _export_results(results, export, output, "text_search")
                
        except Exception as e:
            console.print(f"[red]❌ Błąd wyszukiwania: {e}[/red]")

@search_group.command("semantic")
@click.argument("query", type=str)
@click.option("--model", default="sentence-transformers", 
              type=click.Choice(["sentence-transformers", "openai", "sklearn"]),
              help="Model embeddings do użycia")
@click.option("--limit", "-l", default=10, help="Liczba wyników")
@click.option("--min-score", default=0.5, help="Minimalny score podobieństwa")
@click.option("--show-content/--no-content", default=False, help="Pokaż fragmenty treści")
@click.option("--export", type=click.Choice(["json", "csv"]), help="Eksportuj wyniki")
@click.option("--output", "-o", help="Plik wyjściowy dla eksportu")
@click.pass_context
def semantic_search(ctx, query: str, model: str, limit: int, min_score: float, 
                   show_content: bool, export: Optional[str], output: Optional[str]):
    """🧠 Wyszukiwanie semantyczne (podobieństwo wektorowe)"""
    
    db = ctx.obj['db']
    search_engine = SearchEngine(db)
    
    with console.status(f"[bold blue]Generowanie embeddings dla zapytania..."):
        try:
            # Generowanie embedding dla zapytania
            provider = EmbeddingProvider()
            query_embedding = provider.encode(query, provider_name=model)
            
        except Exception as e:
            console.print(f"[red]❌ Błąd generowania embeddings: {e}[/red]")
            raise click.Abort()
    
    with console.status(f"[bold green]Wyszukiwanie semantyczne..."):
        try:
            results = search_engine.semantic_search(
                query_embedding=query_embedding,
                model_name=model,
                limit=limit,
                min_score=min_score
            )
            
            if not results:
                console.print("[yellow]🔍 Nie znaleziono dokumentów semantycznie podobnych[/yellow]")
                return
            
            # Wyświetlenie wyników
            _display_search_results(results, query, f"Wyszukiwanie Semantyczne ({model})", show_content)
            
            # Eksport jeśli wymagany
            if export:
                _export_results(results, export, output, "semantic_search")
                
        except Exception as e:
            console.print(f"[red]❌ Błąd wyszukiwania semantycznego: {e}[/red]")

@search_group.command("hybrid")
@click.argument("query", type=str)
@click.option("--model", default="sentence-transformers", 
              type=click.Choice(["sentence-transformers", "openai", "sklearn"]),
              help="Model embeddings do użycia")
@click.option("--semantic-weight", default=0.7, help="Waga wyszukiwania semantycznego (0.0-1.0)")
@click.option("--limit", "-l", default=10, help="Liczba wyników")
@click.option("--min-score", default=0.3, help="Minimalny score podobieństwa")
@click.option("--show-content/--no-content", default=False, help="Pokaż fragmenty treści")
@click.option("--export", type=click.Choice(["json", "csv"]), help="Eksportuj wyniki")
@click.option("--output", "-o", help="Plik wyjściowy dla eksportu")
@click.pass_context
def hybrid_search(ctx, query: str, model: str, semantic_weight: float, limit: int, 
                 min_score: float, show_content: bool, export: Optional[str], output: Optional[str]):
    """🔄 Wyszukiwanie hybrydowe (semantyczne + pełnotekstowe)"""
    
    if not 0.0 <= semantic_weight <= 1.0:
        console.print("[red]❌ Waga semantyczna musi być między 0.0 a 1.0[/red]")
        raise click.Abort()
    
    db = ctx.obj['db']
    search_engine = SearchEngine(db)
    
    with console.status(f"[bold blue]Generowanie embeddings dla zapytania..."):
        try:
            # Generowanie embedding dla zapytania
            provider = EmbeddingProvider()
            query_embedding = provider.encode(query, provider_name=model)
            
        except Exception as e:
            console.print(f"[red]❌ Błąd generowania embeddings: {e}[/red]")
            raise click.Abort()
    
    with console.status(f"[bold green]Wyszukiwanie hybrydowe..."):
        try:
            results = search_engine.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                model_name=model,
                semantic_weight=semantic_weight,
                limit=limit,
                min_score=min_score
            )
            
            if not results:
                console.print("[yellow]🔍 Nie znaleziono dokumentów pasujących do zapytania[/yellow]")
                return
            
            # Wyświetlenie wyników
            title = f"Wyszukiwanie Hybrydowe (semantyczne: {semantic_weight:.1%}, tekstowe: {1-semantic_weight:.1%})"
            _display_search_results(results, query, title, show_content)
            
            # Eksport jeśli wymagany
            if export:
                _export_results(results, export, output, "hybrid_search")
                
        except Exception as e:
            console.print(f"[red]❌ Błąd wyszukiwania hybrydowego: {e}[/red]")

@search_group.command("recommend")
@click.argument("document_id", type=int)
@click.option("--model", default="sentence-transformers", 
              type=click.Choice(["sentence-transformers", "openai", "sklearn"]),
              help="Model embeddings do użycia")
@click.option("--limit", "-l", default=10, help="Liczba rekomendacji")
@click.option("--min-score", default=0.5, help="Minimalny score podobieństwa")
@click.option("--show-content/--no-content", default=False, help="Pokaż fragmenty treści")
@click.option("--export", type=click.Choice(["json", "csv"]), help="Eksportuj wyniki")
@click.option("--output", "-o", help="Plik wyjściowy dla eksportu")
@click.pass_context
def recommend_documents(ctx, document_id: int, model: str, limit: int, min_score: float,
                       show_content: bool, export: Optional[str], output: Optional[str]):
    """💡 Rekomenduje dokumenty podobne do danego dokumentu"""
    
    db = ctx.obj['db']
    search_engine = SearchEngine(db)
    
    with console.status(f"[bold green]Wyszukiwanie podobnych dokumentów..."):
        try:
            # Sprawdzenie czy dokument istnieje
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT title FROM documents WHERE id = %s", (document_id,))
                    row = cur.fetchone()
                    if not row:
                        console.print(f"[red]❌ Nie znaleziono dokumentu o ID: {document_id}[/red]")
                        raise click.Abort()
                    
                    source_title = row[0]
            
            results = search_engine.recommend_similar(
                document_id=document_id,
                model_name=model,
                limit=limit + 1,  # +1 bo usuniemy dokument źródłowy
                min_score=min_score
            )
            
            # Usunięcie dokumentu źródłowego z wyników
            results = [r for r in results if r['id'] != document_id][:limit]
            
            if not results:
                console.print(f"[yellow]🔍 Nie znaleziono dokumentów podobnych do '{source_title}'[/yellow]")
                return
            
            # Wyświetlenie wyników
            title = f"Rekomendacje dla: '{source_title}' (ID: {document_id})"
            _display_search_results(results, "", title, show_content)
            
            # Eksport jeśli wymagany
            if export:
                _export_results(results, export, output, "recommendations")
                
        except Exception as e:
            console.print(f"[red]❌ Błąd generowania rekomendacji: {e}[/red]")

@search_group.command("interactive")
@click.option("--model", default="sentence-transformers", 
              type=click.Choice(["sentence-transformers", "openai", "sklearn"]),
              help="Model embeddings do użycia")
@click.option("--search-type", default="hybrid",
              type=click.Choice(["text", "semantic", "hybrid"]),
              help="Typ wyszukiwania")
@click.pass_context
def interactive_search(ctx, model: str, search_type: str):
    """💬 Interaktywne wyszukiwanie dokumentów"""
    
    console.print(Panel.fit(
        "🔍 [bold]Interaktywne Wyszukiwanie Dokumentów[/bold]\n"
        "Wpisz zapytanie lub 'exit' aby zakończyć.\n"
        f"Tryb: [cyan]{search_type}[/cyan] | Model: [cyan]{model}[/cyan]",
        border_style="blue"
    ))
    
    db = ctx.obj['db']
    search_engine = SearchEngine(db)
    provider = EmbeddingProvider() if search_type in ["semantic", "hybrid"] else None
    
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Zapytanie[/bold cyan]")
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[green]👋 Do widzenia![/green]")
                break
            
            if not query.strip():
                continue
            
            with console.status(f"[bold green]Wyszukiwanie..."):
                if search_type == "text":
                    results = search_engine.text_search(query, limit=5)
                elif search_type == "semantic":
                    query_embedding = provider.encode(query, provider_name=model)
                    results = search_engine.semantic_search(query_embedding, model, limit=5)
                else:  # hybrid
                    query_embedding = provider.encode(query, provider_name=model)
                    results = search_engine.hybrid_search(query, query_embedding, model, limit=5)
            
            if results:
                _display_search_results(results, query, f"Wyniki ({search_type})", show_content=False)
            else:
                console.print("[yellow]🔍 Brak wyników[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[green]👋 Do widzenia![/green]")
            break
        except Exception as e:
            console.print(f"[red]❌ Błąd: {e}[/red]")

def _display_search_results(results: List[Dict[str, Any]], query: str, title: str, show_content: bool = False):
    """Wyświetla wyniki wyszukiwania w formacie tabeli"""
    
    # Główna tabela wyników
    table = Table(title=f"🔍 {title}")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Score", style="green", width=8)
    table.add_column("Tytuł", style="bold", width=35)
    table.add_column("Źródło", style="yellow", width=15)
    
    if show_content:
        table.add_column("Fragment", style="dim", width=40)
    
    for result in results:
        score_str = f"{result['score']:.3f}" if 'score' in result else "N/A"
        source = result.get('metadata', {}).get('source', '-') if result.get('metadata') else '-'
        
        row_data = [
            str(result['id']),
            score_str,
            result['title'][:35] + "..." if len(result['title']) > 35 else result['title'],
            source
        ]
        
        if show_content:
            content_preview = result.get('content', '')[:40] + "..." if result.get('content') else ""
            row_data.append(content_preview)
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Statystyki
    if results:
        avg_score = sum(r.get('score', 0) for r in results) / len(results)
        stats_text = f"Znaleziono: [bold]{len(results)}[/bold] dokumentów | "
        stats_text += f"Średni score: [bold]{avg_score:.3f}[/bold]"
        
        if query:
            stats_text += f" | Zapytanie: '[italic]{query}[/italic]'"
        
        console.print(f"\n[dim]{stats_text}[/dim]")

def _export_results(results: List[Dict[str, Any]], format_type: str, output: Optional[str], search_type: str):
    """Eksportuje wyniki do pliku JSON lub CSV"""
    
    if not output:
        output = f"{search_type}_results.{format_type}"
    
    try:
        if format_type == "json":
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        elif format_type == "csv":
            import csv
            with open(output, 'w', newline='', encoding='utf-8') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        
        console.print(f"[green]📁 Wyniki wyeksportowane do: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Błąd eksportu: {e}[/red]")

@search_group.command("history")
@click.option("--limit", "-l", default=20, help="Liczba ostatnich wyszukiwań")
@click.option("--clear", is_flag=True, help="Wyczyść historię wyszukiwań")
@click.pass_context
def search_history(ctx, limit: int, clear: bool):
    """📚 Historia wyszukiwań (jeśli zaimplementowana w bazie danych)"""
    
    if clear:
        console.print("[yellow]⚠️ Funkcja czyszczenia historii nie jest jeszcze zaimplementowana[/yellow]")
        return
    
    console.print("[yellow]💡 Historia wyszukiwań nie jest jeszcze zaimplementowana[/yellow]")
    console.print("[dim]Przyszłe wydanie będzie zawierać śledzenie historii zapytań z metrykami wydajności[/dim]")