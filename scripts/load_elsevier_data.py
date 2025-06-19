import os
import argparse
import logging
import uuid
from typing import Any, cast, TypedDict
from datetime import datetime

# Biblioteka do pracy z datasetami
from datasets import Dataset, load_dataset, concatenate_datasets # type: ignore

# Biblioteka do komunikacji z PostgreSQL
import psycopg
from psycopg.rows import dict_row
from psycopg.connection import Connection

# Biblioteka do generowania embeddingów
from sentence_transformers import SentenceTransformer

# Biblioteki do tworzenia ładnego interfejsu w konsoli
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn # type: ignore

# --- Konfiguracja ---

# Konfiguracja logowania do pliku dla celów deweloperskich
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_loader.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Inicjalizacja konsoli Rich do ładnego wyświetlania w terminalu
console = Console()

# --- Definicje typów dla lepszej czytelności i bezpieczeństwa kodu ---

class ProcessedDocument(TypedDict):
    """Struktura danych dla przetworzonego dokumentu, gotowego do zapisu."""
    doc_id: str
    title: str
    abstract: str
    full_text: str
    pub_year: int | None
    doi: str | None
    pmid: str | None
    asjc_codes: list[int]
    subject_areas: list[str]
    keywords: list[str]
    author_highlights: list[str]
    word_count: int
    status: str

class TextChunk(TypedDict):
    """Struktura danych dla pojedynczego chunku tekstu z embeddingiem."""
    text: str
    type: str
    index: int
    start_pos: int
    end_pos: int
    embedding: list[float]

# --- Główna klasa procesora danych ---

class ElsevierDataLoader:
    """
    Główna klasa odpowiedzialna za cały proces ETL (Extract, Transform, Load)
    dla datasetu Elsevier.
    """
    
    def __init__(self, db_config: dict[str, str]):
        """
        Inicjalizacja loadera.
        
        Args:
            db_config: Słownik z konfiguracją połączenia do bazy danych.
        """
        self.db_config = db_config
        self.connection: Connection[Any] | None = None
        self.embedding_model: SentenceTransformer | None = None
        
    def connect_to_database(self) -> bool:
        """Nawiązuje połączenie z bazą danych PostgreSQL i weryfikuje jej stan."""
        try:
            conn_string = (
                f"host={self.db_config['host']} "
                f"port={self.db_config['port']} "
                f"dbname={self.db_config['dbname']} "
                f"user={self.db_config['user']} "
                f"password={self.db_config['password']}"
            )
            
            self.connection = psycopg.connect(conn_string, row_factory=dict_row) # type: ignore
            
            with self.connection.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()
                console.print(f"Połączono z PostgreSQL: {version['version'][:50] if version else 'N/A'}...")
                
                cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
                pgvector_info = cur.fetchone()
                if pgvector_info:
                    console.print(f"pgvector dostępne: wersja {pgvector_info['extversion']}")
                else:
                    console.print("[bold red]Błąd: Rozszerzenie pgvector nie jest dostępne w bazie danych![/bold red]")
                    return False
            return True
        except psycopg.Error as e:
            console.print(f"[bold red]Błąd połączenia z bazą danych: {e}[/bold red]")
            return False
    
    def initialize_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Inicjalizuje i ładuje model SentenceTransformer do pamięci."""
        try:
            console.print(f"Ładowanie modelu embeddingu: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            console.print("Model embeddingu załadowany pomyślnie.")
        except Exception as e:
            console.print(f"[bold red]Błąd ładowania modelu embeddingu: {e}[/bold red]")
            raise

    def load_elsevier_dataset(self, split: str = "train", limit: int | None = None) -> Dataset:
        """
        Pobiera dataset z Hugging Face. Obsługuje różne 'splity' i limitowanie rekordów.
        """
        try:
            if split == "all":
                console.print("Ładowanie całego datasetu (wszystkie splity)...")
                datasets = [
                    cast(Dataset, load_dataset("orieg/elsevier-oa-cc-by", split=s, trust_remote_code=True))
                    for s in ["train", "test", "validation"]
                ]
                dataset = concatenate_datasets(datasets)
                console.print(f"Załadowano łącznie {len(dataset)} dokumentów.")
            else:
                console.print(f"Ładowanie datasetu (split: {split})...")
                dataset = cast(Dataset, load_dataset("orieg/elsevier-oa-cc-by", split=split, trust_remote_code=True))
                console.print(f"Załadowano {len(dataset)} dokumentów.")
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset)))) # type: ignore
                console.print(f"Ograniczono do {len(dataset)} dokumentów (tryb testowy).")
                
            return dataset
        except Exception as e:
            console.print(f"[bold red]Błąd ładowania datasetu z Hugging Face: {e}[/bold red]")
            raise

    def process_document(self, doc_data: dict[str, Any]) -> ProcessedDocument | None:
        """
        Przetwarza surowy dokument z datasetu do ustrukturyzowanej, czystej formy.
        To serce logiki transformacji (T w ETL).
        """
        try:
            core_data: dict[str, Any] = doc_data.get('coredata', {}) or {}
            
            # Używamy DOI jako głównego identyfikatora. Jest to kluczowe dla mechanizmu ON CONFLICT.
            doc_id = core_data.get('doi') or core_data.get('pii') or str(uuid.uuid4())
            
            title = str(doc_data.get('title', '')).strip()
            abstract = str(doc_data.get('abstract', '')).strip()
            
            body_text_list = doc_data.get('body_text', [])
            full_text = ' '.join(filter(None, [str(part).strip() for part in body_text_list]))

            pub_date = core_data.get('prism:publicationDate', {})
            pub_year_raw = pub_date.get('year') if isinstance(pub_date, dict) else None # type: ignore
            
            # Bezpieczne parsowanie danych, które mogą mieć różne typy
            pub_year = int(pub_year_raw) if pub_year_raw and str(pub_year_raw).isdigit() else None # type: ignore
            doi = str(core_data.get('doi')) if core_data.get('doi') else None
            pmid = str(core_data.get('pmid')) if core_data.get('pmid') else None
            
            keywords = [str(k).strip() for k in doc_data.get('keywords', []) or []] # type: ignore
            subject_areas = [str(s).strip() for s in doc_data.get('subjareas', []) or []] # type: ignore
            asjc_codes = [int(c) for c in doc_data.get('asjc', []) or [] if str(c).isdigit()] # type: ignore
            author_highlights = [str(h).strip() for h in doc_data.get('author_highlights', []) or []] # type: ignore

            processed_doc: ProcessedDocument = {
                'doc_id': doc_id,
                'title': title,
                'abstract': abstract,
                'full_text': full_text[:50000], # Ograniczenie dla wydajności bazy danych
                'pub_year': pub_year,
                'doi': doi,
                'pmid': pmid,
                'asjc_codes': asjc_codes,
                'subject_areas': subject_areas,
                'keywords': keywords,
                'author_highlights': author_highlights,
                'word_count': len((title + ' ' + abstract + ' ' + full_text).split()),
                'status': 'completed' if title or abstract or full_text else 'error'
            }
            return processed_doc
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania dokumentu (dane wejściowe: {doc_data.get('coredata', {}).get('doi')}): {e}", exc_info=True)
            return None

    def create_text_chunks(self, document: ProcessedDocument, chunk_size: int = 1000, overlap: int = 200) -> list[dict[str, Any]]:
        """
        Dzieli tekst dokumentu na mniejsze, semantycznie spójne "chunki".
        Stosuje strategię dzielenia po zdaniach, aby zachować kontekst.
        """
        chunks: list[dict[str, Any]] = []
        
        if document['title']:
            chunks.append({'text': document['title'], 'type': 'title', 'index': 0, 'start_pos': 0, 'end_pos': len(document['title'])})
        
        if document['abstract']:
            chunks.append({'text': document['abstract'], 'type': 'abstract', 'index': 1, 'start_pos': 0, 'end_pos': len(document['abstract'])})
        
        if document['full_text']:
            text = document['full_text']
            text_length = len(text)
            chunk_index = 2
            start = 0
            while start < text_length:
                end = min(start + chunk_size, text_length)
                # Próba znalezienia naturalnej granicy (koniec zdania) dla lepszej jakości embeddingów
                if end < text_length:
                    for i in range(end, max(end - 100, start), -1):
                        if text[i] in '.!?':
                            end = i + 1
                            break
                
                chunk_text = text[start:end].strip()
                if len(chunk_text) > 50:  # Ignorowanie bardzo krótkich, bezwartościowych chunków
                    chunks.append({'text': chunk_text, 'type': 'content', 'index': chunk_index, 'start_pos': start, 'end_pos': end})
                    chunk_index += 1
                
                start = max(start + chunk_size - overlap, end)
        
        return chunks

    def generate_embeddings(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generuje embeddingi dla listy chunków, wykorzystując przetwarzanie batchowe."""
        if not self.embedding_model:
            raise ValueError("Model embeddingu nie został zainicjalizowany.")
        
        try:
            texts = [str(chunk['text']) for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True) # type: ignore
            
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()
            return chunks
        except Exception as e:
            logger.error(f"Błąd generowania embeddingów: {e}", exc_info=True)
            raise

    def save_document_to_database(self, document: ProcessedDocument, chunks_with_embeddings: list[dict[str, Any]]) -> bool:
        """Zapisuje jeden przetworzony dokument i jego embeddingi do bazy danych w jednej transakcji."""
        if not self.connection:
            logger.error("Brak połączenia z bazą danych w `save_document_to_database`")
            return False
        try:
            with self.connection.cursor() as cur:
                cur.execute("BEGIN")
                
                insert_doc_query = """
                INSERT INTO semantic.documents (
                    doc_id, title, abstract, full_text, pub_year, doi, pmid, 
                    asjc_codes, subject_areas, keywords, author_highlights, word_count, status, processed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, abstract = EXCLUDED.abstract, full_text = EXCLUDED.full_text,
                    pub_year = EXCLUDED.pub_year, doi = EXCLUDED.doi, pmid = EXCLUDED.pmid,
                    asjc_codes = EXCLUDED.asjc_codes, subject_areas = EXCLUDED.subject_areas,
                    keywords = EXCLUDED.keywords, author_highlights = EXCLUDED.author_highlights,
                    word_count = EXCLUDED.word_count, status = EXCLUDED.status, 
                    updated_at = CURRENT_TIMESTAMP, processed_at = EXCLUDED.processed_at
                RETURNING document_id
                """
                
                cur.execute(insert_doc_query, (
                    document['doc_id'], document['title'], document['abstract'], document['full_text'],
                    document['pub_year'], document['doi'], document['pmid'], document['asjc_codes'],
                    document['subject_areas'], document['keywords'], document['author_highlights'],
                    document['word_count'], document['status'], datetime.now()
                ))
                
                result = cur.fetchone()
                if not result:
                    raise Exception("Nie udało się pobrać document_id po operacji INSERT/UPDATE")
                
                document_id = result['document_id']
                
                # Usunięcie starych embeddingów jest kluczowe przy aktualizacji
                cur.execute("DELETE FROM semantic.document_embeddings WHERE document_id = %s", (document_id,))
                
                insert_embedding_query = """
                INSERT INTO semantic.document_embeddings (
                    document_id, chunk_text, chunk_index, chunk_type,
                    embedding, start_position, end_position, word_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                for chunk in chunks_with_embeddings:
                    cur.execute(insert_embedding_query, (
                        document_id, chunk['text'], chunk['index'], chunk['type'],
                        chunk['embedding'], chunk['start_pos'], chunk['end_pos'],
                        len(str(chunk['text']).split())
                    ))
                
                cur.execute("COMMIT")
                return True
        except psycopg.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Błąd zapisu do bazy danych dla doc_id {document['doc_id']}: {e}", exc_info=True)
            return False

    def load_documents(self, limit: int | None = None, split: str = "train"):
        """Orkiestruje cały proces: łączy się z bazą, ładuje dane, przetwarza i zapisuje."""
        try:
            dataset = self.load_elsevier_dataset(split=split, limit=limit)
            self.initialize_embedding_model()
            
            total_docs = len(dataset)
            processed_docs, successful_docs, failed_docs = 0, 0, 0
            
            console.print(f"\nRozpoczynanie przetwarzania {total_docs} dokumentów...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress: # type: ignore
                task = progress.add_task("Przetwarzanie...", total=total_docs) # type: ignore
                
                for doc_data in dataset: # type: ignore
                    try:
                        processed_doc = self.process_document(cast(dict[str, Any], doc_data))
                        if not processed_doc:
                            failed_docs += 1
                            continue
                        
                        chunks = self.create_text_chunks(processed_doc)
                        if not chunks:
                            # Jeśli dokument nie ma tekstu do z-chunkowania
                            logger.warning(f"Dokument {processed_doc['doc_id']} nie wygenerował żadnych chunków.")
                            failed_docs += 1
                            continue
                        
                        chunks_with_embeddings = self.generate_embeddings(chunks)
                        
                        if self.save_document_to_database(processed_doc, chunks_with_embeddings):
                            successful_docs += 1
                        else:
                            failed_docs += 1
                    except Exception as e:
                        logger.error(f"Nieoczekiwany błąd w głównej pętli przetwarzania: {e}", exc_info=True)
                        failed_docs += 1
                    finally:
                        processed_docs += 1
                        progress.update(task, advance=1, description=f"Przetworzono {processed_docs}/{total_docs} (Sukces: {successful_docs}, Błędy: {failed_docs})") # type: ignore
            
            self.print_summary(total_docs, successful_docs, failed_docs)
            
        except Exception as e:
            console.print(f"\n[bold red]Krytyczny błąd przerwał działanie programu: {e}[/bold red]")
        finally:
            self.close_connection()

    def print_summary(self, total: int, successful: int, failed: int):
        """Wyświetla końcowe podsumowanie w formie tabeli i statystyki z bazy."""
        table = Table(title="Podsumowanie Ładowania Danych")
        table.add_column("Metryka", style="cyan")
        table.add_column("Wartość", style="magenta")
        table.add_column("Procent", style="green")
        
        table.add_row("Dokumenty do przetworzenia", str(total), "100%")
        table.add_row("Pomyślnie załadowane", str(successful), f"{(successful/total*100):.1f}%" if total > 0 else "N/A")
        table.add_row("Błędy", str(failed), f"{(failed/total*100):.1f}%" if total > 0 else "N/A")
        console.print(table)
        
        try:
            if not self.connection or self.connection.closed:
                console.print("\nPołączenie z bazą zostało zamknięte - nie można pobrać statystyk.")
                return
            with self.connection.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM semantic.documents")
                result = cur.fetchone()
                if result:
                    console.print(f"\nAktualna liczba dokumentów w bazie danych: {result['count']}")
        except Exception as e:
            console.print(f"\n[yellow]Nie udało się pobrać statystyk z bazy danych: {e}[/yellow]")

    def close_connection(self):
        """Zamyka połączenie z bazą danych, jeśli jest otwarte."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            console.print("Połączenie z bazą danych zostało zamknięte.")

def main():
    """Główna funkcja uruchamiająca skrypt z obsługą argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Ładowanie danych Elsevier OA CC-BY do PostgreSQL z pgvector.")
    parser.add_argument("--limit", type=int, help="Ogranicz liczbę dokumentów do przetworzenia (dla testów).")
    parser.add_argument("--split", default="train", choices=["train", "test", "validation", "all"], help="Wybierz podział datasetu do załadowania.")
    
    # Argumenty do konfiguracji bazy danych, z fallbackiem na zmienne środowiskowe
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-port", default=os.getenv("DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "semantic_docs"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", "semantic_password_2024"))
    
    args = parser.parse_args()
    
    db_config = {
        "host": args.db_host, "port": args.db_port, "dbname": args.db_name,
        "user": args.db_user, "password": args.db_password
    }
    
    loader = ElsevierDataLoader(db_config)
    
    try:
        if loader.connect_to_database():
            loader.load_documents(limit=args.limit, split=args.split)
    except KeyboardInterrupt:
        console.print("\n[yellow]Przerwano przez użytkownika.[/yellow]")
    except Exception as e:
        logger.critical(f"Krytyczny błąd w main: {e}", exc_info=True)
    finally:
        console.print("Zakończono działanie skryptu.")

if __name__ == "__main__":
    main()