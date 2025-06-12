"""
Komendy CLI do zarządzania dokumentami.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from semantic_doc_search.core.database import get_sync_session, db_manager
from semantic_doc_search.core.models import Document, DocumentEmbedding, EmbeddingModel
from semantic_doc_search.core.embeddings import embedding_manager
from semantic_doc_search.config.settings import config

console = Console()
logger = logging.getLogger(__name__)


@click.group(name="docs")
def docs_group():
    """Zarządzanie dokumentami w systemie."""
    pass


@docs_group.command()
@click.option(
    "--title", "-t",
    required=True,
    help="Tytuł dokumentu"
)
@click.option(
    "--content", "-c",
    help="Treść dokumentu (alternatywnie użyj --file)"
)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Ścieżka do pliku z treścią dokumentu"
)
@click.option(
    "--source", "-s",
    help="Źródło dokumentu (np. nazwa pliku, URL)"
)
@click.option(
    "--metadata",
    help="Metadane dokumentu w formacie JSON"
)
@click.option(
    "--embed/--no-embed",
    default=True,
    help="Czy generować embeddings dla dokumentu"
)
@click.option(
    "--model", "-m",
    help="Model embeddings do użycia (domyślnie z konfiguracji)"
)
@click.option(
    "--chunk-size",
    type=int,
    help="Rozmiar chunków tekstu (domyślnie z konfiguracji)"
)
def add(
    title: str,
    content: Optional[str],
    file: Optional[str],
    source: Optional[str],
    metadata: Optional[str],
    embed: bool,
    model: Optional[str],
    chunk_size: Optional[int]
):
    """Dodaje nowy dokument do systemu."""
    
    # Pobierz treść z pliku jeśli podano
    if file:
        file_path = Path(file)
        try:
            content = file_path.read_text(encoding='utf-8')
            if not source:
                source = file_path.name
        except Exception as e:
            console.print(f"❌ Błąd podczas czytania pliku: {e}", style="red")
            return
    
    if not content:
        console.print("❌ Musisz podać treść dokumentu (--content) lub plik (--file)", style="red")
        return
    
    # Parsuj metadane
    doc_metadata = {}
    if metadata:
        try:
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"❌ Błędny format JSON w metadanych: {e}", style="red")
            return
    
    try:
        with get_sync_session() as session:
            # Utwórz dokument
            document = Document(
                title=title,
                content=content,
                source=source,
                doc_metadata=doc_metadata
            )
            
            session.add(document)
            session.flush()  # Pobierz ID bez commita
            
            console.print(f"✅ Dokument dodany z ID: {document.id}")
            
            # Generuj embeddings jeśli requested
            if embed:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("🧠 Generowanie embeddings...", total=None)
                    
                    try:
                        _generate_embeddings_for_document(
                            session, document, model, chunk_size
                        )
                        progress.update(task, description="✅ Embeddings wygenerowane")
                        
                    except Exception as e:
                        progress.update(task, description="❌ Błąd embeddings")
                        console.print(f"⚠️  Błąd podczas generowania embeddings: {e}", style="yellow")
            
            session.commit()
            
            # Pokaż podsumowanie
            _show_document_summary(document, embed)
            
    except Exception as e:
        console.print(f"❌ Błąd podczas dodawania dokumentu: {e}", style="red")
        logger.exception("Error adding document")


@docs_group.command()
@click.argument("document_id", type=int)
@click.option(
    "--show-content/--no-content",
    default=False,
    help="Czy pokazać pełną treść dokumentu"
)
@click.option(
    "--show-embeddings/--no-embeddings",
    default=False,
    help="Czy pokazać informacje o embeddings"
)
def show(document_id: int, show_content: bool, show_embeddings: bool):
    """Wyświetla szczegóły dokumentu."""
    
    try:
        with get_sync_session() as session:
            document = session.query(Document).filter(Document.id == document_id).first()
            
            if not document:
                console.print(f"❌ Dokument o ID {document_id} nie istnieje", style="red")
                return
            
            # Panel z podstawowymi informacjami
            info_text = f"""
📄 **Tytuł:** {document.title}
🏷️  **ID:** {document.id}
📅 **Utworzony:** {document.created_at.strftime('%Y-%m-%d %H:%M:%S')}
📝 **Zaktualizowany:** {document.updated_at.strftime('%Y-%m-%d %H:%M:%S')}
📊 **Długość:** {len(document.content)} znaków
"""
            
            if document.source:
                info_text += f"🔗 **Źródło:** {document.source}\n"
            
            if document.doc_metadata:
                info_text += f"📋 **Metadane:** {json.dumps(document.doc_metadata, indent=2, ensure_ascii=False)}\n"
            
            console.print(Panel(info_text, title="📄 Informacje o dokumencie", border_style="blue"))
            
            # Treść dokumentu
            if show_content:
                content_preview = document.content[:1000] + "..." if len(document.content) > 1000 else document.content
                console.print(Panel(content_preview, title="📝 Treść dokumentu", border_style="green"))
            
            # Embeddings
            if show_embeddings:
                embeddings = session.query(DocumentEmbedding)\
                    .filter(DocumentEmbedding.document_id == document_id)\
                    .all()
                
                if embeddings:
                    table = Table(title="🧠 Embeddings")
                    table.add_column("Model", style="cyan")
                    table.add_column("Wymiar", style="magenta")
                    table.add_column("Chunki", style="green")
                    table.add_column("Utworzone", style="yellow")
                    
                    models_stats = {}
                    for emb in embeddings:
                        if emb.embedding_model not in models_stats:
                            models_stats[emb.embedding_model] = {
                                'dimension': emb.embedding_dimension,
                                'chunks': 0,
                                'created': emb.created_at
                            }
                        models_stats[emb.embedding_model]['chunks'] += 1
                    
                    for model, stats in models_stats.items():
                        table.add_row(
                            model,
                            str(stats['dimension']),
                            str(stats['chunks']),
                            stats['created'].strftime('%Y-%m-%d %H:%M')
                        )
                    
                    console.print(table)
                else:
                    console.print("📭 Brak embeddings dla tego dokumentu", style="yellow")
                    
    except Exception as e:
        console.print(f"❌ Błąd podczas pobierania dokumentu: {e}", style="red")
        logger.exception("Error showing document")


@docs_group.command()
@click.argument("document_id", type=int)
@click.option(
    "--title", "-t",
    help="Nowy tytuł dokumentu"
)
@click.option(
    "--content", "-c",
    help="Nowa treść dokumentu"
)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Plik z nową treścią dokumentu"
)
@click.option(
    "--source", "-s",
    help="Nowe źródło dokumentu"
)
@click.option(
    "--metadata",
    help="Nowe metadane w formacie JSON"
)
@click.option(
    "--regenerate-embeddings/--keep-embeddings",
    default=False,
    help="Czy przegenererować embeddings po aktualizacji"
)
@click.option(
    "--model", "-m",
    help="Model embeddings do użycia przy regeneracji"
)
def update(
    document_id: int,
    title: Optional[str],
    content: Optional[str],
    file: Optional[str],
    source: Optional[str],
    metadata: Optional[str],
    regenerate_embeddings: bool,
    model: Optional[str]
):
    """Aktualizuje istniejący dokument."""
    
    # Pobierz treść z pliku jeśli podano
    if file:
        file_path = Path(file)
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"❌ Błąd podczas czytania pliku: {e}", style="red")
            return
    
    # Parsuj metadane
    doc_metadata = None
    if metadata:
        try:
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"❌ Błędny format JSON w metadanych: {e}", style="red")
            return
    
    try:
        with get_sync_session() as session:
            document = session.query(Document).filter(Document.id == document_id).first()
            
            if not document:
                console.print(f"❌ Dokument o ID {document_id} nie istnieje", style="red")
                return
            
            # Aktualizuj pola
            updated_fields = []
            
            if title is not None:
                document.title = title
                updated_fields.append("tytuł")
            
            if content is not None:
                document.content = content
                updated_fields.append("treść")
            
            if source is not None:
                document.source = source
                updated_fields.append("źródło")
            
            if doc_metadata is not None:
                document.doc_metadata = doc_metadata
                updated_fields.append("metadane")
            
            if not updated_fields:
                console.print("⚠️  Nie podano żadnych zmian", style="yellow")
                return
            
            session.flush()
            
            console.print(f"✅ Zaktualizowano: {', '.join(updated_fields)}")
            
            # Regeneruj embeddings jeśli requested
            if regenerate_embeddings:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("🔄 Regenerowanie embeddings...", total=None)
                    
                    try:
                        # Usuń stare embeddings
                        session.query(DocumentEmbedding)\
                            .filter(DocumentEmbedding.document_id == document_id)\
                            .delete()
                        
                        # Generuj nowe
                        _generate_embeddings_for_document(session, document, model)
                        progress.update(task, description="✅ Embeddings zregenerowane")
                        
                    except Exception as e:
                        progress.update(task, description="❌ Błąd regeneracji")
                        console.print(f"⚠️  Błąd podczas regeneracji embeddings: {e}", style="yellow")
            
            session.commit()
            console.print("🎉 Dokument zaktualizowany pomyślnie!", style="green")
            
    except Exception as e:
        console.print(f"❌ Błąd podczas aktualizacji dokumentu: {e}", style="red")
        logger.exception("Error updating document")


@docs_group.command()
@click.argument("document_id", type=int)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Usuń bez potwierdzenia"
)
def delete(document_id: int, force: bool):
    """Usuwa dokument z systemu."""
    
    try:
        with get_sync_session() as session:
            document = session.query(Document).filter(Document.id == document_id).first()
            
            if not document:
                console.print(f"❌ Dokument o ID {document_id} nie istnieje", style="red")
                return
            
            # Pokaż informacje o dokumencie
            console.print(f"📄 Dokument do usunięcia:")
            console.print(f"   ID: {document.id}")
            console.print(f"   Tytuł: {document.title}")
            console.print(f"   Długość: {len(document.content)} znaków")
            
            # Sprawdź liczbę embeddings
            embeddings_count = session.query(DocumentEmbedding)\
                .filter(DocumentEmbedding.document_id == document_id)\
                .count()
            
            if embeddings_count > 0:
                console.print(f"   🧠 Embeddings: {embeddings_count}")
            
            # Potwierdzenie
            if not force:
                if not click.confirm("❓ Czy na pewno chcesz usunąć ten dokument?"):
                    console.print("🚫 Anulowano", style="yellow")
                    return
            
            # Usuń dokument (embeddings zostaną usunięte przez CASCADE)
            session.delete(document)
            session.commit()
            
            console.print("✅ Dokument został usunięty", style="green")
            
    except Exception as e:
        console.print(f"❌ Błąd podczas usuwania dokumentu: {e}", style="red")
        logger.exception("Error deleting document")


@docs_group.command()
@click.option(
    "--limit", "-l",
    default=20,
    help="Liczba dokumentów do wyświetlenia"
)
@click.option(
    "--offset", "-o",
    default=0,
    help="Przesunięcie (paginacja)"
)
@click.option(
    "--source",
    help="Filtruj po źródle"
)
@click.option(
    "--format", "output_format",
    type=click.Choice(['table', 'json']),
    default='table',
    help="Format wyjściowy"
)
def list(limit: int, offset: int, source: Optional[str], output_format: str):
    """Wyświetla listę dokumentów w systemie."""
    
    try:
        with get_sync_session() as session:
            query = session.query(Document)
            
            # Filtr po źródle
            if source:
                query = query.filter(Document.source == source)
            
            # Zlicz total
            total = query.count()
            
            # Pobierz dokumenty z paginacją
            documents = query.order_by(Document.created_at.desc())\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            if not documents:
                console.print("📭 Brak dokumentów", style="yellow")
                return
            
            if output_format == 'json':
                # Format JSON
                docs_data = []
                for doc in documents:
                    docs_data.append(doc.to_dict())
                
                console.print(json.dumps(docs_data, indent=2, ensure_ascii=False, default=str))
            
            else:
                # Format tabeli
                table = Table(title=f"📚 Dokumenty ({offset + 1}-{offset + len(documents)} z {total})")
                table.add_column("ID", style="cyan", width=6)
                table.add_column("Tytuł", style="bold")
                table.add_column("Źródło", style="green")
                table.add_column("Długość", style="magenta", width=10)
                table.add_column("Utworzony", style="yellow", width=12)
                
                for doc in documents:
                    title = doc.title[:50] + "..." if len(doc.title) > 50 else doc.title
                    source_display = doc.source[:20] + "..." if doc.source and len(doc.source) > 20 else (doc.source or "-")
                    
                    table.add_row(
                        str(doc.id),
                        title,
                        source_display,
                        f"{len(doc.content):,}",
                        doc.created_at.strftime('%m-%d %H:%M')
                    )
                
                console.print(table)
                
                # Informacje o paginacji
                if total > offset + limit:
                    console.print(f"\n💡 Użyj --offset {offset + limit} aby zobaczyć więcej")
                    
    except Exception as e:
        console.print(f"❌ Błąd podczas pobierania listy dokumentów: {e}", style="red")
        logger.exception("Error listing documents")


def _generate_embeddings_for_document(
    session,
    document: Document,
    model_name: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> None:
    """Generuje embeddings dla dokumentu."""
    
    if model_name is None:
        model_name = config.embedding.default_model
    
    if chunk_size is None:
        chunk_size = config.embedding.chunk_size
    
    # Podziel treść na chunki
    content = document.content
    chunks = []
    
    if len(content) <= chunk_size:
        chunks = [content]
    else:
        overlap = config.embedding.chunk_overlap
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
    
    # Generuj embeddings
    embedding_result = embedding_manager.generate_embeddings(chunks, model_name)
    
    # Zapisz embeddings do bazy
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embedding_result.embeddings)):
        doc_embedding = DocumentEmbedding(
            document_id=document.id,
            chunk_index=i,
            chunk_text=chunk_text,
            embedding_model=model_name,
            embedding_dimension=embedding_result.dimension,
            embedding=embedding
        )
        session.add(doc_embedding)


def _show_document_summary(document: Document, has_embeddings: bool) -> None:
    """Wyświetla podsumowanie dodanego dokumentu."""
    
    summary_text = f"""
✅ **Dokument został pomyślnie dodany!**

📄 **ID:** {document.id}
🏷️  **Tytuł:** {document.title}
📊 **Długość:** {len(document.content):,} znaków
"""
    
    if document.source:
        summary_text += f"🔗 **Źródło:** {document.source}\n"
    
    if has_embeddings:
        summary_text += "🧠 **Embeddings:** Wygenerowane\n"
    
    console.print(Panel(summary_text, border_style="green"))