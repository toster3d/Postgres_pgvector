"""
Konfiguracja bazy danych PostgreSQL.
"""
import os
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Konfiguracja bazy danych
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "semantic_search")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# Wymiar wektorów embeddings
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))

# Konfiguracja połączenia
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" 