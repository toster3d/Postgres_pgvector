# semantic_doc_search/cli/document_manager.py
"""
Moduł CLI do zarządzania dokumentami.
Obsługuje operacje CRUD na dokumentach z embeddingami.
"""

import click
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from semantic_doc_search.core.database import DatabaseManager
from semantic_doc_search.core.embeddings import EmbeddingProvider
from semantic_doc_search.core.models import Document

console = Console()

@click.group()
def document_group():
    """📄 Zarządzanie dokumentami w systemie"""
    pass

@document_group.command("add")
@click.option("--title", "-t", required=True, help="Tytuł dokumentu")
@click.option("--content", "-c", help="Treść dokumentu (alternatywa dla --file)")
@click.option("--file", "-f", type=click.Path(exists=True), help="Ścieżka do pliku z treścią")
@click.option("--source", "-s", help="Źródło dokumentu (opcjonalne)")
@click.option("--metadata", "-m", help="Metadane w formacie JSON")
@click.option("--embed/--no-embed", default=True, help="Czy generować embeddings")
@click.option("--model", default="sentence-transformers", 
              type=click.Choice(["sentence-transformers", "openai", "sklearn"]),
              help="Model do generowania embeddingów")
@click.pass_context
def add_document(ctx, title: str, content: Optional[str], file: Optional[str], 
                source: Optional[str], metadata: Optional[str], embed: bool, model: str):
    """➕ Dodaje nowy dokument do systemu"""
    
    # Walidacja wejścia
    if not content and not file:
        console.print("[red]❌ Musisz podać treść dokumentu (--content) lub plik (--file)[/red]")
        raise click.Abort()
    
    if content and file:
        console.print("[red]❌ Podaj tylko jeden z parametrów: --content lub --file[/red]")
        raise click.Abort()
    
    # Odczytanie treści z pliku
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            console.print(f"[green]📖 Odczytano treść z pliku: {file}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Błąd odczytu pliku: {e}[/red]")
            raise click.Abort()
    
    # Parsowanie metadanych
    parsed_metadata = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ Błędny format JSON w metadanych: {e}[/red]")
            raise click.Abort()
    
    # Dodanie źródła do metadanych
    if source:
        parsed_metadata["source"] = source
    
    db = ctx.obj['db']
    
    with console.status("[bold green]Dodawanie dokumentu..."):
        try:
            # Utworzenie dokumentu
            document = Document(
                title=title,
                content=content,
                metadata=parsed_metadata
            )
            
            # Zapisanie w bazie danych
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO documents (title, content, search_vector, metadata, created_at)
                        VALUES (%s, %s, to_tsvector('polish', %s), %s, CURRENT_TIMESTAMP)
                        RETURNING id
                    """, (document.title, document.content, document.content, json.dumps(document.metadata)))
                    
                    document_id = cur.fetchone()[0]
                    conn.commit()
            
            console.print(f"[green]✅ Dokument dodany z ID: {document_id}[/green]")
            
            # Generowanie embeddingów jeśli wymagane
            if embed:
                with console.status(f"[bold blue]Generowanie embeddingów ({model})..."):
                    try:
                        provider = EmbeddingProvider()
                        embeddings = provider.encode(content, provider_name=model)
                        
                        # Zapisanie embeddingów
                        with db.get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO document_embeddings (document_id, embedding, model_name)
                                    VALUES (%s, %s, %s)
                                """, (document_id, embeddings, model))
                                conn.commit()
                        
                        console.print(f"[green]🧠 Embeddings wygenerowane ({len(embeddings)} wymiarów)[/green]")
                        
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Błąd generowania embeddingów: {e}[/yellow]")
            
            # Wyświetlenie podsumowania
            summary_table = Table(title=f"📄 Dodany Dokument (ID: {document_id})")
            summary_table.add_column("Atrybut", style="cyan")
            summary_table.add_column("Wartość", style="green")
            
            summary_table.add_row("Tytuł", title)
            summary_table.add_row("Długość treści", f"{len(content)} znaków")
            summary_table.add_row("Embeddings", "✅ Tak" if embed else "❌ Nie")
            if source:
                summary_table.add_row("Źródło", source)
            
            console.print(summary_table)
            
        except Exception as e:
            console.print(f"[red]❌ Błąd dodawania dokumentu: {e}[/red]")
            raise click.Abort()

@document_group.command("show")
@click.argument("document_id", type=int)
@click.option("--show-content/--no-content", default=False, help="Wyświetl pełną treść")
@click.option("--show-embeddings/--no-embeddings", default=False, help="Wyświetl informacje o embeddingach")
@click.pass_context
def show_document(ctx, document_id: int, show_content: bool, show_embeddings: bool):
    """👁️ Wyświetla szczegóły dokumentu"""
    
    db = ctx.obj['db']
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Pobranie dokumentu
                cur.execute("""
                    SELECT id, title, content, metadata, created_at, updated_at
                    FROM documents WHERE id = %s
                """, (document_id,))
                
                row = cur.fetchone()
                if not row:
                    console.print(f"[red]❌ Nie znaleziono dokumentu o ID: {document_id}[/red]")
                    raise click.Abort()
                
                doc_id, title, content, metadata, created_at, updated_at = row
                
                # Podstawowe informacje
                info_table = Table(title=f"📄 Dokument ID: {doc_id}")
                info_table.add_column("Atrybut", style="cyan")
                info_table.add_column("Wartość", style="green")
                
                info_table.add_row("Tytuł", title)
                info_table.add_row("Długość treści", f"{len(content)} znaków")
                info_table.add_row("Utworzony", str(created_at))
                if updated_at != created_at:
                    info_table.add_row("Zaktualizowany", str(updated_at))
                
                if metadata:
                    info_table.add_row("Metadane", json.dumps(metadata, indent=2, ensure_ascii=False))
                
                console.print(info_table)
                
                # Pełna treść jeśli wymagana
                if show_content:
                    content_panel = Panel(
                        content[:1000] + ("..." if len(content) > 1000 else ""),
                        title="📖 Treść dokumentu",
                        border_style="blue"
                    )
                    console.print(content_panel)
                
                # Informacje o embeddingach
                if show_embeddings:
                    cur.execute("""
                        SELECT model_name, array_length(embedding, 1) as dimension
                        FROM document_embeddings WHERE document_id = %s
                    """, (document_id,))
                    
                    embeddings_data = cur.fetchall()
                    
                    if embeddings_data:
                        emb_table = Table(title="🧠 Embeddings")
                        emb_table.add_column("Model", style="cyan")
                        emb_table.add_column("Wymiary", style="green")
                        
                        for model_name, dimension in embeddings_data:
                            emb_table.add_row(model_name, str(dimension))
                        
                        console.print(emb_table)
                    else:
                        console.print("[yellow]⚠️ Brak embeddingów dla tego dokumentu[/yellow]")
                
    except Exception as e:
        console.print(f"[red]❌ Błąd pobierania dokumentu: {e}[/red]")

@document_group.command("list")
@click.option("--limit", "-l", default=10, help="Liczba dokumentów do wyświetlenia")
@click.option("--offset", "-o", default=0, help="Przesunięcie (do paginacji)")
@click.option("--source", "-s", help="Filtr według źródła")
@click.pass_context
def list_documents(ctx, limit: int, offset: int, source: Optional[str]):
    """📋 Listuje dokumenty w systemie"""
    
    db = ctx.obj['db']
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Przygotowanie zapytania z filtrem
                where_clause = ""
                params = []
                
                if source:
                    where_clause = "WHERE metadata->>'source' = %s"
                    params.append(source)
                
                # Pobranie dokumentów
                query = f"""
                    SELECT id, title, 
                           SUBSTRING(content FROM 1 FOR 100) as preview,
                           metadata->>'source' as source,
                           created_at
                    FROM documents 
                    {where_clause}
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                """
                params.extend([limit, offset])
                
                cur.execute(query, params)
                documents = cur.fetchall()
                
                # Liczba wszystkich dokumentów
                count_query = f"SELECT COUNT(*) FROM documents {where_clause}"
                cur.execute(count_query, params[:-2] if source else [])
                total_count = cur.fetchone()[0]
                
                if not documents:
                    console.print("[yellow]📭 Brak dokumentów do wyświetlenia[/yellow]")
                    return
                
                # Utworzenie tabeli
                table = Table(title=f"📋 Dokumenty ({offset + 1}-{offset + len(documents)} z {total_count})")
                table.add_column("ID", style="cyan", width=6)
                table.add_column("Tytuł", style="green", width=30)
                table.add_column("Podgląd", style="dim", width=40)
                table.add_column("Źródło", style="yellow", width=15)
                table.add_column("Data", style="blue", width=12)
                
                for doc_id, title, preview, doc_source, created_at in documents:
                    table.add_row(
                        str(doc_id),
                        title[:30] + "..." if len(title) > 30 else title,
                        preview + "..." if preview else "",
                        doc_source or "-",
                        created_at.strftime("%Y-%m-%d")
                    )
                
                console.print(table)
                
                # Informacja o paginacji
                if total_count > offset + limit:
                    console.print(f"\n[dim]💡 Użyj --offset {offset + limit} aby zobaczyć kolejne dokumenty[/dim]")
                
    except Exception as e:
        console.print(f"[red]❌ Błąd listowania dokumentów: {e}[/red]")

@document_group.command("update")
@click.argument("document_id", type=int)
@click.option("--title", "-t", help="Nowy tytuł dokumentu")
@click.option("--content", "-c", help="Nowa treść dokumentu")
@click.option("--file", "-f", type=click.Path(exists=True), help="Plik z nową treścią")
@click.option("--metadata", "-m", help="Nowe metadane w formacie JSON")
@click.option("--regenerate-embeddings/--keep-embeddings", default=False, 
              help="Czy regenerować embeddings po aktualizacji")
@click.pass_context
def update_document(ctx, document_id: int, title: Optional[str], content: Optional[str], 
                   file: Optional[str], metadata: Optional[str], regenerate_embeddings: bool):
    """✏️ Aktualizuje istniejący dokument"""
    
    if content and file:
        console.print("[red]❌ Podaj tylko jeden z parametrów: --content lub --file[/red]")
        raise click.Abort()
    
    # Odczytanie treści z pliku
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            console.print(f"[red]❌ Błąd odczytu pliku: {e}[/red]")
            raise click.Abort()
    
    # Parsowanie metadanych
    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ Błędny format JSON w metadanych: {e}[/red]")
            raise click.Abort()
    
    db = ctx.obj['db']
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Sprawdzenie czy dokument istnieje
                cur.execute("SELECT id, title, content FROM documents WHERE id = %s", (document_id,))
                row = cur.fetchone()
                if not row:
                    console.print(f"[red]❌ Nie znaleziono dokumentu o ID: {document_id}[/red]")
                    raise click.Abort()
                
                old_id, old_title, old_content = row
                
                # Przygotowanie zapytania aktualizacji
                updates = []
                params = []
                
                if title:
                    updates.append("title = %s")
                    params.append(title)
                
                if content:
                    updates.append("content = %s")
                    params.append(content)
                    updates.append("search_vector = to_tsvector('polish', %s)")
                    params.append(content)
                
                if parsed_metadata is not None:
                    updates.append("metadata = %s")
                    params.append(json.dumps(parsed_metadata))
                
                if updates:
                    updates.append("updated_at = CURRENT_TIMESTAMP")
                    params.append(document_id)
                    
                    query = f"UPDATE documents SET {', '.join(updates)} WHERE id = %s"
                    cur.execute(query, params)
                    conn.commit()
                    
                    console.print(f"[green]✅ Dokument {document_id} został zaktualizowany[/green]")
                
                # Regeneracja embeddingów jeśli wymagana
                if regenerate_embeddings and content:
                    with console.status("[bold blue]Regenerowanie embeddingów..."):
                        # Usunięcie starych embeddingów
                        cur.execute("DELETE FROM document_embeddings WHERE document_id = %s", (document_id,))
                        
                        # Wygenerowanie nowych
                        provider = EmbeddingProvider()
                        embeddings = provider.encode(content, provider_name="sentence-transformers")
                        
                        cur.execute("""
                            INSERT INTO document_embeddings (document_id, embedding, model_name)
                            VALUES (%s, %s, %s)
                        """, (document_id, embeddings, "sentence-transformers"))
                        
                        conn.commit()
                        console.print("[green]🧠 Embeddings zostały zregenerowane[/green]")
                
    except Exception as e:
        console.print(f"[red]❌ Błąd aktualizacji dokumentu: {e}[/red]")

@document_group.command("delete")
@click.argument("document_id", type=int)
@click.option("--force", "-f", is_flag=True, help="Usuń bez potwierdzenia")
@click.pass_context
def delete_document(ctx, document_id: int, force: bool):
    """🗑️ Usuwa dokument z systemu"""
    
    db = ctx.obj['db']
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Sprawdzenie czy dokument istnieje
                cur.execute("SELECT title FROM documents WHERE id = %s", (document_id,))
                row = cur.fetchone()
                if not row:
                    console.print(f"[red]❌ Nie znaleziono dokumentu o ID: {document_id}[/red]")
                    raise click.Abort()
                
                title = row[0]
                
                # Potwierdzenie usunięcia
                if not force:
                    if not Confirm.ask(f"Czy na pewno chcesz usunąć dokument '{title}' (ID: {document_id})?"):
                        console.print("[yellow]❌ Operacja anulowana[/yellow]")
                        return
                
                # Usunięcie dokumentu (embeddings zostaną usunięte przez CASCADE)
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                deleted_count = cur.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    console.print(f"[green]✅ Dokument '{title}' został usunięty[/green]")
                else:
                    console.print("[yellow]⚠️ Dokument nie został usunięty[/yellow]")
                
    except Exception as e:
        console.print(f"[red]❌ Błąd usuwania dokumentu: {e}[/red]")