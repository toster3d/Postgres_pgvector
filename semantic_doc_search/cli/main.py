# semantic_doc_search/cli/main.py
"""
Główny moduł CLI dla systemu semantycznego wyszukiwania dokumentów.
Wykorzystuje Click 8.2.0 z Rich dla kolorowego interfejsu.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

from semantic_doc_search.config.settings import get_settings
from semantic_doc_search.core.database import Database
from semantic_doc_search.cli.document_manager import document_group
from semantic_doc_search.cli.search_commands import search_group

console = Console()

@click.group()
@click.version_option(version="1.0.0", message="%(prog)s %(version)s")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    🔍 Semantic Document Search - System semantycznego wyszukiwania dokumentów
    
    Zaawansowany system wyszukiwania wykorzystujący PostgreSQL z pgvector
    do semantycznego wyszukiwania i rekomendacji dokumentów.
    """
    ctx.ensure_object(dict)
    
    # Inicjalizacja konfiguracji
    try:
        ctx.obj['settings'] = get_settings()
        ctx.obj['db'] = Database()
    except Exception as e:
        console.print(f"[red]❌ Błąd inicjalizacji: {e}[/red]")
        raise click.Abort()

@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """🚀 Inicjalizuje system i bazę danych"""
    settings = ctx.obj['settings']
    
    with console.status("[bold green]Inicjalizacja systemu..."):
        try:
            # Sprawdzenie połączenia z bazą
            db = ctx.obj['db']
            db.initialize()
            
            console.print("\n[bold green]✅ System został pomyślnie zainicjalizowany![/bold green]")
            
            # Wyświetlenie informacji o konfiguracji
            config_table = Table(title="🔧 Konfiguracja Systemu")
            config_table.add_column("Parametr", style="cyan")
            config_table.add_column("Wartość", style="green")
            
            config_table.add_row("Baza danych", f"{settings.database.host}:{settings.database.port}/{settings.database.name}")
            config_table.add_row("Model embeddings", settings.embedding.sentence_transformers_model)
            config_table.add_row("Waga semantyczna", str(settings.search.default_semantic_weight))
            
            console.print(config_table)
            
        except Exception as e:
            console.print(f"[red]❌ Błąd inicjalizacji: {e}[/red]")
            raise click.Abort()

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """📊 Wyświetla status systemu i statystyki"""
    db = ctx.obj['db']
    
    try:
        stats = db.get_stats()
        
        # Utworzenie tabeli statusu
        status_table = Table(title="📈 Status Systemu")
        status_table.add_column("Metryka", style="cyan")
        status_table.add_column("Wartość", style="green")
        
        status_table.add_row("Dokumenty", str(stats.get('active_documents', 0)))
        status_table.add_row("Embeddings", str(stats.get('total_embeddings', 0)))
        status_table.add_row("Modele", str(stats.get('unique_models', 0)))
        
        console.print(status_table)
        
        if stats.get('active_documents', 0) == 0:
            console.print("\n[yellow]💡 Brak dokumentów w systemie. Użyj 'semantic-docs docs add' aby dodać pierwszy dokument.[/yellow]")
                
    except Exception as e:
        console.print(f"[red]❌ Błąd pobierania statusu: {e}[/red]")

@cli.command()
@click.pass_context  
def health(ctx: click.Context) -> None:
    """🩺 Sprawdza kondycję systemu"""
    settings = ctx.obj['settings']
    
    health_checks: list[tuple[str, str, str]] = []
    
    # Sprawdzenie bazy danych
    try:
        db = ctx.obj['db']
        db.test_connection()
        health_checks.append(("Baza danych PostgreSQL", "✅ OK", "Połączenie nawiązane"))
        
        # Sprawdzenie pgvector
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'")
                vector_version = cur.fetchone()
                
        if vector_version:
            health_checks.append(("Rozszerzenie pgvector", "✅ OK", vector_version[0] or "Zainstalowane"))
        else:
            health_checks.append(("Rozszerzenie pgvector", "❌ BŁĄD", "Nie zainstalowane"))
            
    except Exception as e:
        health_checks.append(("Baza danych PostgreSQL", "❌ BŁĄD", str(e)))
    
    # Sprawdzenie modeli embeddings
    try:
        from semantic_doc_search.core.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        test_embedding = provider.encode("test", provider_name="sentence-transformers")
        health_checks.append(("Model Sentence Transformers", "✅ OK", f"Wymiar: {len(test_embedding)}"))
    except Exception as e:
        health_checks.append(("Model Sentence Transformers", "❌ BŁĄD", str(e)))
    
    # Sprawdzenie OpenAI (jeśli skonfigurowane)
    if settings.embedding.openai_api_key:
        try:
            from semantic_doc_search.core.embeddings import EmbeddingProvider
            provider = EmbeddingProvider()
            test_embedding = provider.encode("test", provider_name="openai")
            health_checks.append(("Model OpenAI", "✅ OK", f"Wymiar: {len(test_embedding)}"))
        except Exception as e:
            health_checks.append(("Model OpenAI", "❌ BŁĄD", str(e)))
    else:
        health_checks.append(("Model OpenAI", "⚠️ POMIŃ", "Brak klucza API"))
    
    # Wyświetlenie wyników
    health_table = Table(title="🩺 Sprawdzenie Kondycji Systemu")
    health_table.add_column("Komponent", style="cyan")
    health_table.add_column("Status", style="bold")
    health_table.add_column("Szczegóły", style="dim")
    
    for component, status, details in health_checks:
        health_table.add_row(component, status, details)
    
    console.print(health_table)

# Rejestracja grup komend
cli.add_command(document_group, name="docs")
cli.add_command(search_group, name="search")

if __name__ == "__main__":
    cli()