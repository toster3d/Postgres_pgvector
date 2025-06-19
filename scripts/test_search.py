import os
import sys
import argparse
import psycopg
import psycopg.rows
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from typing import Any, TypedDict, cast
from datetime import datetime

console = Console()


class DatabaseStats(TypedDict):
    total_documents: int
    total_embeddings: int
    completed_documents: int
    latest_document: datetime | None


class DocumentResult(TypedDict):
    document_id: int
    doc_id: str
    title: str
    abstract: str
    full_text_preview: str | None
    keywords: list[str]
    subject_areas: list[str]
    asjc_codes: list[int]


class SearchResult(DocumentResult):
    similarity_score: float | None
    rank_score: float | None
    combined_score: float | None
    chunk_text: str | None
    chunk_type: str | None
    semantic_score: float | None
    fulltext_score: float | None


class SemanticSearchEngine:
    """Silnik wyszukiwania semantycznego dla dokumentów naukowych"""
    
    def __init__(self, db_config: dict[str, str]):
        """
        Inicjalizacja silnika wyszukiwania
        
        Args:
            db_config: Konfiguracja bazy danych
        """
        self.db_config = db_config
        self.connection: psycopg.Connection[Any] | None = None
        self.embedding_model: SentenceTransformer | None = None
        
    def connect_to_database(self) -> bool:
        """Nawiązanie połączenia z bazą danych"""
        try:
            conn_string = (
                f"host={self.db_config['host']} "
                f"port={self.db_config['port']} "
                f"dbname={self.db_config['dbname']} "
                f"user={self.db_config['user']} "
                f"password={self.db_config['password']}"
            )
            
            self.connection = psycopg.connect(conn_string)
            self.connection.row_factory = psycopg.rows.dict_row  # type: ignore
            
            console.print("Połączono z bazą danych PostgreSQL")
            return True
            
        except Exception as e:
            console.print(f"Błąd połączenia z bazą danych: {e}")
            return False
    
    def initialize_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Inicjalizacja modelu embeddingu"""
        try:
            console.print(f"Ładowanie modelu embeddingu: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            console.print("Model embeddingu załadowany pomyślnie")
        except Exception as e:
            console.print(f"Błąd ładowania modelu: {e}")
            raise
    
    def get_database_stats(self) -> DatabaseStats:
        """Pobranie statystyk bazy danych"""
        try:
            if not self.connection:
                raise ValueError("Brak połączenia z bazą danych")
            with self.connection.cursor() as cur:
                # Liczba dokumentów
                cur.execute("SELECT COUNT(*) as count FROM semantic.documents")
                docs_count_row = cur.fetchone()
                docs_count = cast(dict[str, Any], docs_count_row)['count'] if docs_count_row else 0
                
                # Liczba embeddingów
                cur.execute("SELECT COUNT(*) as count FROM semantic.document_embeddings")
                embeddings_count_row = cur.fetchone()
                embeddings_count = cast(dict[str, Any], embeddings_count_row)['count'] if embeddings_count_row else 0
                
                # Liczba dokumentów ze statusem 'completed'
                cur.execute("SELECT COUNT(*) as count FROM semantic.documents WHERE status = 'completed'")
                completed_count_row = cur.fetchone()
                completed_count = cast(dict[str, Any], completed_count_row)['count'] if completed_count_row else 0
                
                # Najnowszy dokument
                cur.execute("SELECT MAX(created_at) as latest FROM semantic.documents")
                latest_doc_row = cur.fetchone()
                latest_doc = cast(dict[str, Any], latest_doc_row)['latest'] if latest_doc_row else None
                
                return {
                    'total_documents': docs_count,
                    'total_embeddings': embeddings_count,
                    'completed_documents': completed_count,
                    'latest_document': latest_doc
                }
                
        except Exception as e:
            console.print(f"Błąd pobierania statystyk: {e}")
            return {
                'total_documents': 0,
                'total_embeddings': 0,
                'completed_documents': 0,
                'latest_document': None
            }
    
    def semantic_search(self, query: str, limit: int = 10, similarity_threshold: float = 0.3) -> list[SearchResult]:
        """
        Wyszukiwanie semantyczne na podstawie zapytania
        
        Args:
            query: Zapytanie wyszukiwania
            limit: Maksymalna liczba wyników
            similarity_threshold: Próg podobieństwa (0-1)
            
        Returns:
            Lista wyników wyszukiwania
        """
        try:
            if not self.embedding_model:
                raise ValueError("Model embeddingu nie został zainicjalizowany")
            
            # Generowanie embeddingu dla zapytania
            query_embedding = self.embedding_model.encode([query])[0].tolist()  # type: ignore
            
            # Zapytanie SQL z wyszukiwaniem wektorowym
            search_query = """
            SELECT 
                d.document_id,
                d.doc_id,
                d.title,
                d.abstract,
                SUBSTRING(d.full_text, 1, 200) as full_text_preview,
                d.keywords,
                d.subject_areas,
                d.asjc_codes,
                e.chunk_text,
                e.chunk_type,
                1 - (e.embedding <=> %s::vector) as similarity_score
            FROM semantic.document_embeddings e
            JOIN semantic.documents d ON e.document_id = d.document_id
            WHERE 1 - (e.embedding <=> %s::vector) >= %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """
            
            if not self.connection:
                raise ValueError("Brak połączenia z bazą danych")
            
            with self.connection.cursor() as cur:
                cur.execute(search_query, (
                    query_embedding, 
                    query_embedding, 
                    similarity_threshold, 
                    query_embedding, 
                    limit
                ))
                results = cur.fetchall()
                
            return [cast(SearchResult, dict(row)) for row in results]
            
        except Exception as e:
            console.print(f"Błąd wyszukiwania semantycznego: {e}")
            return []
    
    def fulltext_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Wyszukiwanie pełnotekstowe
        
        Args:
            query: Zapytanie wyszukiwania
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista wyników wyszukiwania
        """
        try:
            # Zapytanie SQL z full-text search
            search_query = """
            SELECT 
                d.document_id,
                d.doc_id,
                d.title,
                d.abstract,
                SUBSTRING(d.full_text, 1, 200) as full_text_preview,
                d.keywords,
                d.subject_areas,
                d.asjc_codes,
                ts_rank_cd(
                    d.title_tsvector || d.abstract_tsvector || d.full_text_tsvector, 
                    plainto_tsquery('english', %s)
                ) as rank_score
            FROM semantic.documents d
            WHERE (
                d.title_tsvector @@ plainto_tsquery('english', %s) OR
                d.abstract_tsvector @@ plainto_tsquery('english', %s) OR
                d.full_text_tsvector @@ plainto_tsquery('english', %s)
            )
            ORDER BY ts_rank_cd(
                d.title_tsvector || d.abstract_tsvector || d.full_text_tsvector, 
                plainto_tsquery('english', %s)
            ) DESC
            LIMIT %s
            """
            
            if not self.connection:
                raise ValueError("Brak połączenia z bazą danych")
            
            with self.connection.cursor() as cur:
                cur.execute(search_query, (query, query, query, query, query, limit))
                results = cur.fetchall()
                
            return [dict(row) for row in results]
            
        except Exception as e:
            console.print(f"Błąd wyszukiwania pełnotekstowego: {e}")
            return []
    
    def hybrid_search(self, query: str, semantic_weight: float = 0.7, limit: int = 10) -> list[dict[str, Any]]:
        """
        Wyszukiwanie hybrydowe (semantyczne + pełnotekstowe)
        
        Args:
            query: Zapytanie wyszukiwania
            semantic_weight: Waga wyszukiwania semantycznego (0-1)
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista wyników wyszukiwania
        """
        try:
            # Pobranie wyników z obu metod
            semantic_results = self.semantic_search(query, limit * 2)
            fulltext_results = self.fulltext_search(query, limit * 2)
            
            # Normalizacja wyników i łączenie
            combined_results: dict[int, dict[str, Any]] = {}
            
            # Dodanie wyników semantycznych
            for result in semantic_results:
                doc_id = result['document_id']
                similarity_score = result.get('similarity_score', 0.0)
                score = (similarity_score or 0.0) * semantic_weight
                combined_results[doc_id] = {
                    **result,
                    'combined_score': score,
                    'semantic_score': result['similarity_score'],
                    'fulltext_score': 0.0
                }
            
            # Dodanie wyników pełnotekstowych
            fulltext_weight = 1.0 - semantic_weight
            for result in fulltext_results:
                doc_id = result['document_id']
                score = result['rank_score'] * fulltext_weight
                
                if doc_id in combined_results:
                    # Aktualizacja istniejącego wyniku
                    combined_results[doc_id]['combined_score'] += score
                    combined_results[doc_id]['fulltext_score'] = result['rank_score']
                else:
                    # Nowy wynik
                    combined_results[doc_id] = {
                        **result,
                        'combined_score': score,
                        'semantic_score': 0.0,
                        'fulltext_score': result['rank_score']
                    }
            
            # Sortowanie według połączonego wyniku
            sorted_results: list[dict[str, Any]] = sorted(
                list(combined_results.values()),
                key=lambda x: x.get('combined_score', 0.0),
                reverse=True
            )
            
            return sorted_results[:limit]
            
        except Exception as e:
            console.print(f"Błąd wyszukiwania hybrydowego: {e}")
            return []
    
    def display_results(self, results: list[dict[str, Any]] | list[SearchResult], search_type: str = "semantyczne"):
        """Wyświetlenie wyników wyszukiwania"""
        if not results:
            console.print(f"Brak wyników dla wyszukiwania {search_type}")
            return
        
        console.print(f"\nWyniki wyszukiwania {search_type}go ({len(results)} dokumentów):")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Lp.", width=3)
        table.add_column("Tytuł", width=40)
        table.add_column("Rok", width=6)
        table.add_column("DOI", width=20)
        
        if results and 'similarity_score' in results[0]:
            table.add_column("Podobieństwo", width=12)
        elif results and 'rank_score' in results[0]:
            table.add_column("Ranking", width=12)
        elif results and 'combined_score' in results[0]:
            table.add_column("Wynik łączny", width=12)
        
        for i, result in enumerate(results[:10], 1):
            title = cast(str, result.get('title', 'Brak tytułu'))
            if len(title) > 35:
                title = title[:32] + "..."
            
            year = str(result.get('pub_year', 'N/A'))
            doi = cast(str | None, result.get('doi', 'N/A'))
            if doi and len(doi) > 18:
                doi = doi[:15] + "..."
            
            # Wybór odpowiedniego wyniku
            score_val: float | None = None
            if 'similarity_score' in result and result['similarity_score'] is not None:
                score_val = cast(float, result['similarity_score'])
            elif 'rank_score' in result and result['rank_score'] is not None:
                score_val = cast(float, result['rank_score'])
            elif 'combined_score' in result and result['combined_score'] is not None:
                score_val = cast(float, result['combined_score'])

            score = f"{score_val:.3f}" if score_val is not None else "N/A"
            
            table.add_row(str(i), title, year, doi or "N/A", score)
        
        console.print(table)
        
        # Wyświetlenie szczegółów pierwszego wyniku
        if results:
            self.display_document_details(results[0])
    
    def display_document_details(self, document: dict[str, Any] | SearchResult):
        """Wyświetlenie szczegółów dokumentu"""
        title = cast(str, document.get('title', 'Brak tytułu'))
        abstract = cast(str, document.get('abstract', 'Brak abstraktu'))
        keywords = cast(list[str], document.get('keywords', []))
        subject_areas = cast(list[str], document.get('subject_areas', []))
        asjc_codes = cast(list[int], document.get('asjc_codes', []))
        
        # Ograniczenie długości abstraktu
        if len(abstract) > 500:
            abstract = abstract[:497] + "..."
        
        details_text = f"""
[bold]Tytuł:[/bold] {title}

[bold]Abstrakt:[/bold] {abstract}

[bold]Słowa kluczowe:[/bold] {', '.join(keywords) if keywords else 'Brak'}
[bold]Obszary tematyczne:[/bold] {', '.join(subject_areas) if subject_areas else 'Brak'}
[bold]Kody ASJC:[/bold] {', '.join(map(str, asjc_codes)) if asjc_codes else 'Brak'}
        """
        
        panel = Panel(
            details_text.strip(),
            title="Szczegóły pierwszego wyniku",
            border_style="blue"
        )
        console.print(panel)
    
    def interactive_search(self):
        """Interaktywny interfejs wyszukiwania"""
        console.print(Panel.fit(
            "[bold blue]Semantic Document Search[/bold blue]\n"
            "Interaktywny interfejs wyszukiwania dokumentów naukowych",
            border_style="blue"
        ))
        
        while True:
            console.print("\nOpcje wyszukiwania:")
            console.print("1. Wyszukiwanie semantyczne")
            console.print("2. Wyszukiwanie pełnotekstowe")
            console.print("3. Wyszukiwanie hybrydowe")
            console.print("4. Statystyki bazy danych")
            console.print("5. Wyjście")
            
            choice = Prompt.ask("Wybierz opcję", choices=["1", "2", "3", "4", "5"])
            
            if choice == "5":
                console.print("Do widzenia!")
                break
            elif choice == "4":
                self.show_database_stats()
            elif choice in ["1", "2", "3"]:
                query = Prompt.ask("Wprowadź zapytanie wyszukiwania")
                limit = int(Prompt.ask("Liczba wyników", default="10"))
                
                if choice == "1":
                    threshold = float(Prompt.ask("Próg podobieństwa (0-1)", default="0.3"))
                    results = self.semantic_search(query, limit, threshold)
                    self.display_results(results, "semantyczne")
                elif choice == "2":
                    results = self.fulltext_search(query, limit)
                    self.display_results(results, "pełnotekstowe")
                elif choice == "3":
                    weight = float(Prompt.ask("Waga semantyczna (0-1)", default="0.3"))
                    results = self.hybrid_search(query, weight, limit)
                    self.display_results(results, "hybrydowe")
    
    def show_database_stats(self):
        """Wyświetlenie statystyk bazy danych"""
        stats = self.get_database_stats()
        
        if stats:
            table = Table(title="Statystyki bazy danych")
            table.add_column("Metryka", style="cyan")
            table.add_column("Wartość", style="magenta")
            
            table.add_row("Dokumenty przetworzone", str(stats['total_documents']))
            table.add_row("Łączna liczba embeddingów", str(stats['total_embeddings']))
            table.add_row("Ostatni dokument", str(stats['latest_document']))
            
            console.print(table)
    
    def close_connection(self):
        """Zamknięcie połączenia z bazą danych"""
        if self.connection:
            self.connection.close()
            console.print("Połączenie z bazą danych zostało zamknięte")


def main():
    """Główna funkcja aplikacji"""
    parser = argparse.ArgumentParser(description="Testowanie wyszukiwania semantycznego")
    parser.add_argument("--query", help="Zapytanie do przetestowania (np. 'machine learning', 'safety management', 'neural networks')")
    parser.add_argument("--type", choices=["semantic", "fulltext", "hybrid"], 
                       default="semantic", help="Typ wyszukiwania")
    parser.add_argument("--limit", type=int, default=10, help="Liczba wyników")
    parser.add_argument("--interactive", action="store_true", help="Tryb interaktywny")
    parser.add_argument("--stats", action="store_true", help="Wyświetl statystyki bazy danych")
    
    # Parametry bazy danych
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-port", default=os.getenv("DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "semantic_docs"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", "semantic_password_2024"))
    
    args = parser.parse_args()
    
    # Konfiguracja bazy danych
    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password
    }
    
    # Utworzenie silnika wyszukiwania
    search_engine = SemanticSearchEngine(db_config)
    
    try:
        # Połączenie z bazą danych
        if not search_engine.connect_to_database():
            sys.exit(1)
        
        # Inicjalizacja modelu embeddingu
        search_engine.initialize_embedding_model()
        
        if args.stats:
            # Wyświetlenie statystyk
            search_engine.show_database_stats()
        elif args.interactive:
            # Tryb interaktywny
            search_engine.interactive_search()
        elif args.query:
            # Pojedyncze zapytanie
            console.print(f"Wyszukiwanie: '{args.query}' (typ: {args.type})")
            
            results: list[dict[str, Any]] = []
            if args.type == "semantic":
                results = cast(list[dict[str, Any]], search_engine.semantic_search(args.query, args.limit))
            elif args.type == "fulltext":
                results = search_engine.fulltext_search(args.query, args.limit)
            elif args.type == "hybrid":
                results = search_engine.hybrid_search(args.query, limit=args.limit)
            
            search_engine.display_results(results, args.type)
        else:
            console.print("Podaj zapytanie (--query) lub użyj trybu interaktywnego (--interactive)")
            
    except KeyboardInterrupt:
        console.print("\nPrzerwano przez użytkownika")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Błąd: {e}[/bold red]")
        sys.exit(1)
    finally:
        search_engine.close_connection()


if __name__ == "__main__":
    main()