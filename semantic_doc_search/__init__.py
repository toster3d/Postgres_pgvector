"""
Semantic Document Search
========================

System semantycznego wyszukiwania i rekomendacji dokumentów wykorzystujący
PostgreSQL z rozszerzeniem pgvector. Umożliwia przechowywanie dokumentów tekstowych,
generowanie wektorowych reprezentacji ich treści (embeddings) oraz wyszukiwanie
podobnych dokumentów na podstawie znaczenia semantycznego.

Ten pakiet zawiera kompletny system z interfejsem wiersza poleceń (CLI)
oraz modułami core do embeddingów i wyszukiwania.

Autorzy:
    Semantic Doc Search Team <team@semanticdocs.com>

Licencja:
    MIT

Wersja:
    1.0.0
"""

__version__ = "1.0.0"
__author__ = "Semantic Doc Search Team"
__email__ = "team@semanticdocs.com"
__license__ = "MIT"

# Eksport głównych komponentów
from semantic_doc_search.cli.main import cli
from semantic_doc_search.core.database import DatabaseManager, db_manager
from semantic_doc_search.core.models import Document
from semantic_doc_search.core.embeddings import BaseEmbeddingProvider, embedding_manager
from semantic_doc_search.core.search import SemanticSearchEngine
from semantic_doc_search.config.settings import config

# Wersje zależności
__dependencies__ = {
    "pgvector": "0.8.0",
    "psycopg": "3.2.9",
    "sentence-transformers": "4.1.0"
}

# Eksport głównych publicznych API
__all__ = [
    "cli",
    "DatabaseManager",
    "db_manager",
    "Document",
    "BaseEmbeddingProvider",
    "embedding_manager",
    "SemanticSearchEngine",
    "config",
    "__version__",
    "__author__",
    "__dependencies__"
]