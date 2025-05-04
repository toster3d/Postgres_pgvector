"""
Skrypt do inicjalizacji bazy danych.
"""
import os
import sys
import logging

# Dodaj katalog nadrzędny do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db import init_db

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Funkcja główna inicjalizująca bazę danych."""
    try:
        logger.info("Inicjalizacja bazy danych...")
        init_db()
        logger.info("Baza danych zainicjalizowana pomyślnie!")
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji bazy danych: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 