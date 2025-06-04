"""Settings and configuration for semantic document search."""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Konfiguracja bazy danych PostgreSQL."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = Field(default="localhost", description="Host bazy danych")
    port: int = Field(default=5432, description="Port bazy danych")
    name: str = Field(default="semantic_docs", description="Nazwa bazy danych")
    user: str = Field(default="postgres", description="Użytkownik bazy danych")
    password: str = Field(default="postgres", description="Hasło do bazy danych")
    
    # Connection pool settings
    min_pool_size: int = Field(default=5, description="Minimalna liczba połączeń w puli")
    max_pool_size: int = Field(default=20, description="Maksymalna liczba połączeń w puli")
    pool_timeout: int = Field(default=30, description="Timeout połączenia w sekundach")
    
    # pgvector settings
    ivfflat_lists: int = Field(default=100, description="Liczba list dla indeksu IVFFlat")
    hnsw_m: int = Field(default=16, description="Parametr M dla indeksu HNSW")
    hnsw_ef_construction: int = Field(default=200, description="ef_construction dla HNSW")
    
    @property
    def url(self) -> str:
        """Zwraca URL połączenia z bazą danych."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def async_url(self) -> str:
        """Zwraca asynchroniczny URL połączenia z bazą danych."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class EmbeddingSettings(BaseSettings):
    """Konfiguracja modeli embeddings."""
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    
    # Domyślny model embeddings
    default_model: Literal["sentence-transformers", "openai", "sklearn"] = Field(
        default="sentence-transformers",
        description="Domyślny provider embeddings"
    )
    
    # Sentence Transformers settings
    sentence_transformers_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model Sentence Transformers"
    )
    device: str = Field(default="cpu", description="Urządzenie dla Sentence Transformers")
    cache_folder: Optional[Path] = Field(default=None, description="Folder cache modeli")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="Klucz API OpenAI")
    openai_model: str = Field(
        default="text-embedding-3-small", 
        description="Model OpenAI embeddings"
    )
    openai_base_url: Optional[str] = Field(default=None, description="Bazowy URL OpenAI API")
    
    # Chunk settings
    chunk_size: int = Field(default=1000, description="Rozmiar chunka tekstu")
    chunk_overlap: int = Field(default=200, description="Nakładanie się chunków")
    
    # Processing settings
    batch_size: int = Field(default=32, description="Rozmiar batch do przetwarzania")
    enable_cache: bool = Field(default=True, description="Czy włączyć cache embeddings")
    
    @field_validator("cache_folder", mode="before")
    @classmethod
    def validate_cache_folder(cls, v: Any) -> Optional[Path]:
        """Waliduje folder cache."""
        if v is None:
            return None
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path


class SearchSettings(BaseSettings):
    """Konfiguracja wyszukiwania."""
    
    model_config = SettingsConfigDict(env_prefix="SEARCH_")
    
    # Semantic search settings
    default_similarity_metric: Literal["cosine", "l2", "inner_product"] = Field(
        default="cosine",
        description="Domyślna metryka podobieństwa"
    )
    default_limit: int = Field(default=10, description="Domyślny limit wyników")
    max_limit: int = Field(default=100, description="Maksymalny limit wyników")
    
    # Hybrid search settings
    default_semantic_weight: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Domyślna waga wyszukiwania semantycznego w hybrydowym"
    )
    
    # Full-text search settings
    fulltext_language: str = Field(default="polish", description="Język dla full-text search")
    min_similarity_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0, 
        description="Minimalny wynik podobieństwa"
    )
    
    # Performance settings
    use_parallel_search: bool = Field(default=True, description="Czy używać równoległego wyszukiwania")
    search_timeout: int = Field(default=30, description="Timeout wyszukiwania w sekundach")


class LoggingSettings(BaseSettings):
    """Konfiguracja logowania."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Poziom logowania"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format logów"
    )
    file_path: Optional[Path] = Field(default=None, description="Ścieżka do pliku logów")
    max_file_size: int = Field(default=10485760, description="Maksymalny rozmiar pliku logów (bytes)")
    backup_count: int = Field(default=5, description="Liczba plików backup")
    
    @field_validator("file_path", mode="before")
    @classmethod
    def validate_log_file(cls, v: Any) -> Optional[Path]:
        """Waliduje ścieżkę pliku logów."""
        if v is None:
            return None
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class AppSettings(BaseSettings):
    """Główna konfiguracja aplikacji."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # App metadata
    app_name: str = Field(default="Semantic Document Search", description="Nazwa aplikacji")
    app_version: str = Field(default="1.0.0", description="Wersja aplikacji")
    debug: bool = Field(default=False, description="Tryb debug")
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Środowisko aplikacji"
    )
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # CLI settings
    cli_progress: bool = Field(default=True, description="Czy pokazywać progress bars w CLI")
    cli_colors: bool = Field(default=True, description="Czy używać kolorów w CLI")
    export_formats: list[str] = Field(
        default=["json", "csv"],
        description="Dostępne formaty eksportu"
    )
    
    # Security settings
    max_document_size: int = Field(
        default=10485760, 
        description="Maksymalny rozmiar dokumentu w bajtach (10MB)"
    )
    allowed_file_extensions: list[str] = Field(
        default=[".txt", ".md", ".pdf", ".docx"],
        description="Dozwolone rozszerzenia plików"
    )
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Waliduje środowisko."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be one of: development, staging, production")
        return v
    
    def get_database_url(self, async_driver: bool = False) -> str:
        """Zwraca URL bazy danych."""
        return self.database.async_url if async_driver else self.database.url
    
    def is_development(self) -> bool:
        """Sprawdza czy jest to środowisko deweloperskie."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Sprawdza czy jest to środowisko produkcyjne."""
        return self.environment == "production"


# Global settings instance
settings = AppSettings()


def get_settings() -> AppSettings:
    """Zwraca instancję ustawień aplikacji."""
    return settings


def reload_settings() -> AppSettings:
    """Przeładowuje ustawienia aplikacji."""
    global settings
    settings = AppSettings()
    return settings