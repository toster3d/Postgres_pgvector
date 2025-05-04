"""
Konfiguracja modułu embeddings.
"""
import os
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Model embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))

# Klucz API OpenAI (tylko jeśli używamy modeli OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 