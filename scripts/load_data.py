# scripts/load_data.py
# Skrypt do ładowania różnorodnych danych do systemu semantycznego wyszukiwania

"""
Skrypt do ładowania danych z różnych źródeł do bazy PostgreSQL z pgvector.
Obsługuje pliki lokalne, artykuły z Wikipedii oraz generowanie przykładowych danych.
"""

import sys
import csv
import json
import logging
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

# Dodanie ścieżki do głównego modułu
sys.path.append('/app')

from semantic_doc_search.core.database import db_manager
from semantic_doc_search.core.embeddings import EmbeddingProvider
from semantic_doc_search.models.document import Document, DocumentCreate

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/load_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Klasa do ładowania różnych typów danych."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicjalizacja loadera danych.
        
        Args:
            embedding_model: Model do generowania embeddings
        """
        self.embedding_provider = EmbeddingProvider(model_name=embedding_model)
        self.stats = {
            'loaded': 0,
            'failed': 0,
            'skipped': 0
        }
        
    def load_from_directory(self, directory: str, pattern: str = "*.txt", 
                           generate_embeddings: bool = True) -> int:
        """
        Ładuje dokumenty z katalogu.
        
        Args:
            directory: Ścieżka do katalogu
            pattern: Wzorzec nazw plików
            generate_embeddings: Czy generować embeddings
            
        Returns:
            Liczba załadowanych dokumentów
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return 0
        
        files = list(directory_path.glob(pattern))
        if not files:
            logger.warning(f"No files matching {pattern} found in {directory}")
            return 0
        
        logger.info(f"Found {len(files)} files to process")
        
        for file_path in tqdm(files, desc="Loading files"):
            try:
                # Odczytanie zawartości pliku
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty file: {file_path}")
                    self.stats['skipped'] += 1
                    continue
                
                # Utworzenie dokumentu
                doc_create = DocumentCreate(
                    title=file_path.stem,
                    content=content,
                    source=f"file://{file_path}",
                    metadata={
                        'file_size': file_path.stat().st_size,
                        'file_extension': file_path.suffix,
                        'created_at': datetime.now().isoformat()
                    }
                )
                
                # Dodanie do bazy danych
                doc_id = self._add_document(doc_create, generate_embeddings)
                if doc_id:
                    self.stats['loaded'] += 1
                    logger.debug(f"Loaded document {doc_id} from {file_path}")
                else:
                    self.stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                self.stats['failed'] += 1
        
        return self.stats['loaded']
    
    def load_from_csv(self, csv_path: str, title_col: str = "title", 
                     content_col: str = "content", generate_embeddings: bool = True) -> int:
        """
        Ładuje dokumenty z pliku CSV.
        
        Args:
            csv_path: Ścieżka do pliku CSV
            title_col: Nazwa kolumny z tytułami
            content_col: Nazwa kolumny z treścią
            generate_embeddings: Czy generować embeddings
            
        Returns:
            Liczba załadowanych dokumentów
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"CSV file {csv_path} does not exist")
            return 0
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            logger.info(f"Found {len(rows)} rows in CSV")
            
            for row in tqdm(rows, desc="Loading CSV data"):
                try:
                    title = row.get(title_col, '').strip()
                    content = row.get(content_col, '').strip()
                    
                    if not title or not content:
                        logger.warning(f"Empty title or content in row: {row}")
                        self.stats['skipped'] += 1
                        continue
                    
                    # Utworzenie dokumentu
                    doc_create = DocumentCreate(
                        title=title,
                        content=content,
                        source=f"csv://{csv_file.name}",
                        metadata={
                            'row_data': {k: v for k, v in row.items() 
                                       if k not in [title_col, content_col]},
                            'loaded_at': datetime.now().isoformat()
                        }
                    )
                    
                    # Dodanie do bazy danych
                    doc_id = self._add_document(doc_create, generate_embeddings)
                    if doc_id:
                        self.stats['loaded'] += 1
                    else:
                        self.stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing CSV row: {e}")
                    self.stats['failed'] += 1
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return 0
        
        return self.stats['loaded']
    
    def load_wikipedia_articles(self, topics: List[str], limit_per_topic: int = 5,
                               generate_embeddings: bool = True) -> int:
        """
        Ładuje artykuły z Wikipedii.
        
        Args:
            topics: Lista tematów do wyszukania
            limit_per_topic: Limit artykułów na temat
            generate_embeddings: Czy generować embeddings
            
        Returns:
            Liczba załadowanych artykułów
        """
        logger.info(f"Loading Wikipedia articles for topics: {topics}")
        
        for topic in topics:
            try:
                articles = self._fetch_wikipedia_articles(topic, limit_per_topic)
                
                for article in tqdm(articles, desc=f"Loading {topic}"):
                    try:
                        doc_create = DocumentCreate(
                            title=article['title'],
                            content=article['content'],
                            source=article['url'],
                            metadata={
                                'topic': topic,
                                'wikipedia_id': article.get('pageid'),
                                'loaded_at': datetime.now().isoformat()
                            }
                        )
                        
                        # Dodanie do bazy danych
                        doc_id = self._add_document(doc_create, generate_embeddings)
                        if doc_id:
                            self.stats['loaded'] += 1
                        else:
                            self.stats['failed'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing Wikipedia article: {e}")
                        self.stats['failed'] += 1
                        
            except Exception as e:
                logger.error(f"Error fetching Wikipedia articles for {topic}: {e}")
        
        return self.stats['loaded']
    
    def generate_demo_data(self, count: int = 20, generate_embeddings: bool = True) -> int:
        """
        Generuje przykładowe dane demonstracyjne.
        
        Args:
            count: Liczba dokumentów do wygenerowania
            generate_embeddings: Czy generować embeddings
            
        Returns:
            Liczba wygenerowanych dokumentów
        """
        demo_topics = [
            ("Sztuczna Inteligencja", [
                "Uczenie maszynowe to gałąź sztucznej inteligencji, która umożliwia komputerom uczenie się bez jawnego programowania.",
                "Sieci neuronowe to modele matematyczne inspirowane biologicznymi sieciami neuronowymi.",
                "Deep learning wykorzystuje wielowarstwowe sieci neuronowe do rozpoznawania wzorców w danych.",
                "Natural Language Processing (NLP) pozwala komputerom rozumieć i przetwarzać język naturalny.",
                "Computer vision umożliwia maszynom interpretację i analizę obrazów cyfrowych."
            ]),
            ("Bazy Danych", [
                "PostgreSQL to zaawansowany system zarządzania relacyjną bazą danych.",
                "NoSQL to kategoria systemów bazodanowych różniących się od tradycyjnych baz relacyjnych.",
                "Indeksy bazodanowe przyspieszają wyszukiwanie danych w tabelach.",
                "ACID to zestaw właściwości gwarantujących niezawodność transakcji bazodanowych.",
                "Sharding to technika podziału dużej bazy danych na mniejsze, łatwiejsze w zarządzaniu części."
            ]),
            ("Programowanie", [
                "Python to język programowania wysokiego poziomu o czytelnej składni.",
                "Git to rozproszony system kontroli wersji używany do śledzenia zmian w kodzie.",
                "Docker to platforma do konteneryzacji aplikacji.",
                "REST API to architektoniczny styl projektowania usług webowych.",
                "Test-driven development (TDD) to metodologia, gdzie testy są pisane przed kodem."
            ]),
            ("Nauka", [
                "Teoria względności Einsteina zrewolucjonizowała nasze rozumienie przestrzeni i czasu.",
                "Mechanika kwantowa opisuje zachowanie materii i energii na poziomie atomowym.",
                "DNA zawiera genetyczne instrukcje dla rozwoju i funkcjonowania organizmów żywych.",
                "Fotosynteza to proces konwersji światła słonecznego w energię chemiczną przez rośliny.",
                "Ewolucja to proces zmian genetycznych w populacjach organizmów w czasie."
            ])
        ]
        
        logger.info(f"Generating {count} demo documents")
        
        import random
        
        for i in tqdm(range(count), desc="Generating demo data"):
            try:
                # Wybierz losowy temat i treść
                topic, contents = random.choice(demo_topics)
                content = random.choice(contents)
                
                # Dodaj wariacje do treści
                variations = [
                    f"Wprowadzenie: {content}",
                    f"{content} Jest to fundamentalny koncept w tej dziedzinie.",
                    f"Badania pokazują, że {content.lower()}",
                    f"Zgodnie z najnowszymi odkryciami, {content.lower()}",
                    f"{content} Ma to szerokie zastosowania praktyczne."
                ]
                
                full_content = random.choice(variations)
                
                doc_create = DocumentCreate(
                    title=f"{topic} - Dokument {i+1}",
                    content=full_content,
                    source="generated://demo",
                    metadata={
                        'topic': topic,
                        'generated': True,
                        'demo_id': i + 1,
                        'created_at': datetime.now().isoformat()
                    }
                )
                
                # Dodanie do bazy danych
                doc_id = self._add_document(doc_create, generate_embeddings)
                if doc_id:
                    self.stats['loaded'] += 1
                else:
                    self.stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error generating demo document {i}: {e}")
                self.stats['failed'] += 1
        
        return self.stats['loaded']
    
    def _fetch_wikipedia_articles(self, topic: str, limit: int) -> List[Dict]:
        """
        Pobiera artykuły z Wikipedii dla danego tematu.
        
        Args:
            topic: Temat do wyszukania
            limit: Maksymalna liczba artykułów
            
        Returns:
            Lista artykułów z metadanymi
        """
        # Wikipedia API endpoints
        search_url = "https://pl.wikipedia.org/w/api.php"
        
        # Wyszukaj artykuły
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': topic,
            'srlimit': limit
        }
        
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        search_data = response.json()
        
        articles = []
        
        for item in search_data['query']['search']:
            try:
                # Pobierz pełną treść artykułu
                content_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': item['title'],
                    'prop': 'extracts',
                    'exintro': True,
                    'explaintext': True,
                    'exsectionformat': 'plain'
                }
                
                content_response = requests.get(search_url, params=content_params)
                content_response.raise_for_status()
                content_data = content_response.json()
                
                pages = content_data['query']['pages']
                for page_id, page in pages.items():
                    if 'extract' in page and page['extract'].strip():
                        articles.append({
                            'title': page['title'],
                            'content': page['extract'],
                            'url': f"https://pl.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
                            'pageid': page_id
                        })
                        
            except Exception as e:
                logger.warning(f"Error fetching content for {item['title']}: {e}")
                continue
        
        return articles
    
    def _add_document(self, doc_create: DocumentCreate, generate_embeddings: bool) -> Optional[int]:
        """
        Dodaje dokument do bazy danych.
        
        Args:
            doc_create: Dane dokumentu do utworzenia
            generate_embeddings: Czy generować embeddings
            
        Returns:
            ID utworzonego dokumentu lub None w przypadku błędu
        """
        try:
            # Sprawdź czy dokument już istnieje
            existing = db_manager.execute_query(
                "SELECT id FROM documents WHERE title = %s AND source = %s",
                (doc_create.title, doc_create.source)
            )
            
            if existing:
                logger.debug(f"Document already exists: {doc_create.title}")
                self.stats['skipped'] += 1
                return existing[0]['id']
            
            # Dodaj dokument
            doc_id = db_manager.execute_command(
                """
                INSERT INTO documents (title, content, source, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    doc_create.title,
                    doc_create.content,
                    doc_create.source,
                    json.dumps(doc_create.metadata) if doc_create.metadata else None,
                    datetime.now()
                )
            )
            
            if generate_embeddings and doc_id:
                # Generuj embeddings
                try:
                    embedding = self.embedding_provider.generate_embedding(doc_create.content)
                    
                    # Dodaj embedding do bazy
                    db_manager.execute_command(
                        """
                        INSERT INTO document_embeddings (document_id, model_name, embedding, created_at)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            doc_id,
                            self.embedding_provider.model_name,
                            embedding,
                            datetime.now()
                        )
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for document {doc_id}: {e}")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to database: {e}")
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Zwraca statystyki ładowania."""
        return self.stats.copy()


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Ładowanie danych do systemu semantycznego wyszukiwania"
    )
    
    # Źródło danych
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--directory', '-d', 
                             help="Katalog z plikami tekstowymi")
    source_group.add_argument('--csv', '-c',
                             help="Plik CSV z danymi")
    source_group.add_argument('--wikipedia', '-w', nargs='+',
                             help="Tematy do pobrania z Wikipedii")
    source_group.add_argument('--demo', action='store_true',
                             help="Generuj przykładowe dane")
    
    # Opcje
    parser.add_argument('--count', type=int, default=20,
                       help="Liczba dokumentów (dla demo/wikipedia)")
    parser.add_argument('--pattern', default="*.txt",
                       help="Wzorzec plików (dla directory)")
    parser.add_argument('--title-col', default="title",
                       help="Kolumna z tytułami (dla CSV)")
    parser.add_argument('--content-col', default="content",
                       help="Kolumna z treścią (dla CSV)")
    parser.add_argument('--embed', action='store_true',
                       help="Generuj embeddings")
    parser.add_argument('--model', default="all-MiniLM-L6-v2",
                       help="Model embeddings")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Tryb verbose")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Inicjalizacja
    logger.info("Initializing data loader...")
    
    try:
        # Test połączenia z bazą
        if not db_manager.test_connection():
            logger.error("Cannot connect to database")
            return 1
        
        # Inicjalizacja loadera
        loader = DataLoader(embedding_model=args.model)
        
        # Wykonaj ładowanie w zależności od źródła
        if args.directory:
            count = loader.load_from_directory(
                args.directory, args.pattern, args.embed
            )
        elif args.csv:
            count = loader.load_from_csv(
                args.csv, args.title_col, args.content_col, args.embed
            )
        elif args.wikipedia:
            count = loader.load_wikipedia_articles(
                args.wikipedia, args.count, args.embed
            )
        elif args.demo:
            count = loader.generate_demo_data(args.count, args.embed)
        else:
            logger.error("No data source specified")
            return 1
        
        # Wyświetl statystyki
        stats = loader.get_stats()
        logger.info("Loading completed!")
        logger.info(f"Documents loaded: {stats['loaded']}")
        logger.info(f"Documents failed: {stats['failed']}")
        logger.info(f"Documents skipped: {stats['skipped']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())