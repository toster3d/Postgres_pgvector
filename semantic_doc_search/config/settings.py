import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Ścieżka główna projektu
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class DatabaseConfig:
    """Konfiguracja bazy danych PostgreSQL."""
    
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "semantic_docs")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "postgres")
    
    # Psycopg3 specific settings
    min_pool_size: int = int(os.getenv("DB_MIN_POOL_SIZE", "5"))
    max_pool_size: int = int(os.getenv("DB_MAX_POOL_SIZE", "20"))
    connection_timeout: float = float(os.getenv("DB_CONNECTION_TIMEOUT", "30.0"))
    
    @property
    def url(self) -> str:
        """Zwraca URL połączenia do bazy danych."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def async_url(self) -> str:
        """Zwraca URL połączenia async do bazy danych."""
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@dataclass
class EmbeddingConfig:
    """Konfiguracja modeli embeddings."""
    
    # Domyślny model
    default_model: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Sentence Transformers
    sentence_transformers_cache_dir: str = os.getenv(
        "SENTENCE_TRANSFORMERS_CACHE", 
        str(PROJECT_ROOT / ".cache" / "sentence_transformers")
    )
    sentence_transformers_device: str = os.getenv("DEVICE", "cpu")
    
    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30.0"))
    
    # Chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Batch processing
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


@dataclass
class SearchConfig:
    """Konfiguracja wyszukiwania."""
    
    # Domyślne limity
    default_limit: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
    max_limit: int = int(os.getenv("MAX_SEARCH_LIMIT", "100"))
    
    # Hybrid search
    default_semantic_weight: float = float(os.getenv("DEFAULT_SEMANTIC_WEIGHT", "0.7"))
    
    # Vector index settings
    ivfflat_lists: int = int(os.getenv("IVFFLAT_LISTS", "100"))
    
    # Full-text search language
    fts_language: str = os.getenv("FTS_LANGUAGE", "english")


@dataclass
class AppConfig:
    """Główna konfiguracja aplikacji."""
    
    # Podstawowe ustawienia
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Komponenty konfiguracji
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    # CLI settings
    cli_verbose: bool = False
    cli_quiet: bool = False
    
    # Export settings
    export_formats: Dict[str, str] = field(default_factory=lambda: {
        "json": "application/json",
        "csv": "text/csv",
        "txt": "text/plain"
    })


# Globalna instancja konfiguracji
config = AppConfig()


# Modele embeddings dostępne w systemie
AVAILABLE_EMBEDDING_MODELS = {
    # Sentence Transformers
    "sentence-transformers": {
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "description": "Szybki i wydajny model dla języka angielskiego",
            "multilingual": False
        },
        "all-mpnet-base-v2": {
            "dimension": 768,
            "description": "Wysokiej jakości model dla języka angielskiego",
            "multilingual": False
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "dimension": 384,
            "description": "Wielojęzyczny model (w tym polski)",
            "multilingual": True
        },
        "paraphrase-multilingual-mpnet-base-v2": {
            "dimension": 768,
            "description": "Wysokiej jakości wielojęzyczny model",
            "multilingual": True
        }
    },
    
    # OpenAI
    "openai": {
        "text-embedding-ada-002": {
            "dimension": 1536,
            "description": "Starszy model OpenAI (przestarzały)",
            "multilingual": True
        },
        "text-embedding-3-small": {
            "dimension": 1536,
            "description": "Nowy wydajny model OpenAI",
            "multilingual": True
        },
        "text-embedding-3-large": {
            "dimension": 3072,
            "description": "Najlepszy model OpenAI",
            "multilingual": True
        }
    },
    
    # Scikit-learn (dla demonstracji)
    "sklearn": {
        "tfidf-vectorizer": {
            "dimension": 1000,
            "description": "Prosty model TF-IDF",
            "multilingual": False
        }
    }
}


# Funkcje pomocnicze
def get_model_info(model_type: str, model_name: str) -> Dict[str, Any]:
    """Zwraca informacje o modelu embeddings."""
    return AVAILABLE_EMBEDDING_MODELS.get(model_type, {}).get(model_name, {})


def validate_config() -> None:
    """Waliduje konfigurację aplikacji."""
    errors: list[str] = []
    
    # Sprawdź klucz OpenAI jeśli wymagany
    if (config.embedding.default_model.startswith("text-embedding") and 
        not config.embedding.openai_api_key):
        errors.append("OPENAI_API_KEY is required for OpenAI models")
    
    # Sprawdź parametry bazy danych
    if not config.database.db_host:
        errors.append("Database host is required")
    
    if not config.database.db_name:
        errors.append("Database name is required")
    
    # Sprawdź limity wyszukiwania
    if config.search.default_limit > config.search.max_limit:
        errors.append("Default search limit cannot exceed max limit")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")


def setup_logging() -> None:
    """Konfiguruje logowanie aplikacji."""
    import logging
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Poziom logowania
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Podstawowa konfiguracja
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Ustawienia dla zewnętrznych bibliotek
    if not config.debug:
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


# Automatyczna walidacja przy imporcie (opcjonalna)
if __name__ == "__main__":
    try:
        validate_config()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        exit(1)