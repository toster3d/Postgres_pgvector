"""
Główny punkt wejścia dla CLI aplikacji semantycznego wyszukiwania dokumentów.
"""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

# Dodaj ścieżkę projektu do PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_doc_search.config.settings import config, validate_config
from semantic_doc_search.core.database import db_manager
from semantic_doc_search.cli.commands.docs import docs_group
from semantic_doc_search.cli.commands.search import search_group

console = Console()
logger = logging.getLogger(__name__)


def setup_cli_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Konfiguruje logowanie dla CLI."""
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@click.group(name="semantic-docs")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Włącz szczegółowe logowanie"
)
@click.option(
    "--quiet", "-q", 
    is_flag=True,
    help="Wyłącz większość wiadomości (tylko błędy)"
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Ścieżka do pliku konfiguracyjnego .env"
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config_file: str) -> None:
    """
    🔍 Semantyczne Wyszukiwanie Dokumentów
    
    System do semantycznego wyszukiwania i rekomendacji dokumentów 
    z wykorzystaniem PostgreSQL i pgvector.
    
    Przykłady użycia:
    
        # Dodaj dokument
        semantic-docs docs add --title "Mój dokument" --content "Treść..." --embed
        
        # Wyszukaj semantycznie  
        semantic-docs search semantic "Jaka jest natura świadomości?"
        
        # Wyszukanie hybrydowe
        semantic-docs search hybrid "AI i świadomość" --semantic-weight 0.7
    """
    # Konfiguruj logowanie
    setup_cli_logging(verbose, quiet)
    
    # Ustaw globalne ustawienia CLI
    config.cli_verbose = verbose
    config.cli_quiet = quiet
    
    # Załaduj dodatkowy plik konfiguracyjny jeśli podany
    if config_file:
        from dotenv import load_dotenv
        load_dotenv(config_file, override=True)
        if verbose:
            console.print(f"🔧 Załadowano konfigurację z: {config_file}")
    
    # Waliduj konfigurację
    try:
        validate_config()
        if verbose:
            console.print("✅ Konfiguracja jest poprawna")
    except ValueError as e:
        console.print(f"❌ Błąd konfiguracji: {e}", style="red")
        ctx.exit(1)
    
    # Przygotuj kontekst dla podkomend
    ctx.ensure_object(dict)
    ctx.obj['console'] = console
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Inicjalizuje bazę danych i tworzy wymagane tabele."""
    console = ctx.obj['console']
    verbose = ctx.obj['verbose']
    
    console.print("🚀 Inicjalizacja bazy danych...", style="bold blue")
    
    try:
        # Testuj połączenie
        if not db_manager.test_connection():
            console.print("❌ Nie można połączyć z bazą danych", style="red")
            console.print(f"📋 Sprawdź ustawienia połączenia: {config.database.url}")
            ctx.exit(1)
        
        console.print("✅ Połączenie z bazą danych OK")
        
        # Utwórz tabele
        db_manager.create_tables()
        console.print("✅ Tabele zostały utworzone")
        
        # Pokaż informacje o bazie
        if verbose:
            info = db_manager.get_database_info()
            console.print("📊 Informacje o bazie danych:")
            for key, value in info.items():
                console.print(f"   {key}: {value}")
        
        console.print("🎉 Inicjalizacja zakończona pomyślnie!", style="green")
        
    except Exception as e:
        console.print(f"❌ Błąd podczas inicjalizacji: {e}", style="red")
        if verbose:
            console.print_exception()
        ctx.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Sprawdza status systemu i wyświetla informacje diagnostyczne."""
    console = ctx.obj['console']
    verbose = ctx.obj['verbose']
    
    console.print("🔍 Status systemu semantycznego wyszukiwania", style="bold blue")
    console.print()
    
    # Status połączenia z bazą danych
    console.print("🗄️  Baza danych:", style="bold")
    try:
        if db_manager.test_connection():
            console.print("   ✅ Połączenie: OK", style="green")
            
            info = db_manager.get_database_info()
            console.print(f"   📦 PostgreSQL: {info.get('postgresql_version', 'nieznana')}")
            console.print(f"   🔌 pgvector: {info.get('pgvector_version', 'nieznana')}")
            console.print(f"   📊 Dokumenty: {info.get('documents_count', 0)}")
            console.print(f"   🧠 Embeddings: {info.get('embeddings_count', 0)}")
            
        else:
            console.print("   ❌ Połączenie: BŁĄD", style="red")
            
    except Exception as e:
        console.print(f"   ❌ Błąd: {e}", style="red")
    
    console.print()
    
    # Status modeli embeddings
    console.print("🧠 Modele embeddings:", style="bold")
    try:
        from semantic_doc_search.core.embeddings import embedding_manager
        
        available_models = embedding_manager.get_available_models()
        
        if available_models:
            for model_type, models in available_models.items():
                console.print(f"   📚 {model_type}:")
                for model in models:
                    console.print(f"      • {model}")
        else:
            console.print("   ⚠️  Brak dostępnych modeli", style="yellow")
            
    except Exception as e:
        console.print(f"   ❌ Błąd: {e}", style="red")
    
    console.print()
    
    # Konfiguracja
    if verbose:
        console.print("⚙️  Konfiguracja:", style="bold")
        console.print(f"   🎯 Domyślny model: {config.embedding.default_model}")
        console.print(f"   📏 Domyślny limit: {config.search.default_limit}")
        console.print(f"   ⚖️  Waga semantyczna: {config.search.default_semantic_weight}")
        console.print(f"   🌐 Język FTS: {config.search.fts_language}")
        console.print(f"   🔧 Debug: {config.debug}")


@cli.command()
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Wymuś ponowne utworzenie indeksów"
)
@click.pass_context
def create_indexes(ctx: click.Context, force: bool) -> None:
    """Tworzy indeksy wektorowe dla lepszej wydajności wyszukiwania."""
    console = ctx.obj['console']
    
    console.print("🔧 Tworzenie indeksów wektorowych...", style="bold blue")
    
    try:
        db_manager.create_vector_indexes(force=force)
        console.print("✅ Indeksy zostały utworzone pomyślnie!", style="green")
        
    except Exception as e:
        console.print(f"❌ Błąd podczas tworzenia indeksów: {e}", style="red")
        if ctx.obj['verbose']:
            console.print_exception()
        ctx.exit(1)


@cli.command()
@click.option(
    "--limit", "-l",
    default=10,
    help="Liczba ostatnich wyszukiwań do wyświetlenia"
)
@click.pass_context
def history(ctx: click.Context, limit: int) -> None:
    """Wyświetla historię ostatnich wyszukiwań."""
    console = ctx.obj['console']
    
    try:
        from semantic_doc_search.core.database import get_sync_session
        from semantic_doc_search.core.models import SearchHistory
        from sqlalchemy import desc
        
        with get_sync_session() as session:
            searches = session.query(SearchHistory)\
                .order_by(desc(SearchHistory.created_at))\
                .limit(limit)\
                .all()
            
            if not searches:
                console.print("📭 Brak historii wyszukiwań", style="yellow")
                return
            
            console.print(f"📚 Ostatnie {len(searches)} wyszukiwań:", style="bold blue")
            console.print()
            
            for search in searches:
                time_str = search.created_at.strftime("%Y-%m-%d %H:%M:%S")
                console.print(f"🕐 {time_str}")
                console.print(f"   🔍 Zapytanie: {search.query}")
                console.print(f"   🎯 Typ: {search.search_type}")
                if search.model_used:
                    console.print(f"   🧠 Model: {search.model_used}")
                console.print(f"   📊 Wyniki: {search.results_count or 0}")
                if search.search_time_ms:
                    console.print(f"   ⏱️  Czas: {search.search_time_ms}ms")
                console.print()
                
    except Exception as e:
        console.print(f"❌ Błąd podczas pobierania historii: {e}", style="red")
        if ctx.obj['verbose']:
            console.print_exception()


# Dodaj grupy komend
cli.add_command(docs_group)
cli.add_command(search_group)


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Wyświetla informacje o wersji."""
    console = ctx.obj['console']
    
    console.print("📦 Semantyczne Wyszukiwanie Dokumentów", style="bold blue")
    console.print("   Wersja: 1.0.0")
    console.print("   Python: 3.10+")
    console.print("   PostgreSQL: 17.5+")
    console.print("   pgvector: 0.8.0+")
    console.print()
    console.print("🔗 Więcej informacji:")
    console.print("   GitHub: https://github.com/example/semantic-doc-search")
    console.print("   Dokumentacja: https://semantic-doc-search.readthedocs.io/")


if __name__ == "__main__":
    cli()