"""
Moduł do połączenia z bazą danych PostgreSQL.
"""
import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

# Konfiguracja połączenia z bazą danych
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "semantic_search"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Kontekstowy manager do połączenia z bazą danych PostgreSQL.
    
    Yields:
        psycopg.Connection: Obiekt połączenia z bazą danych.
    """
    conn = psycopg.connect(**DB_CONFIG, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_db_cursor() -> Generator[psycopg.Cursor, None, None]:
    """
    Kontekstowy manager do uzyskania kursora bazy danych.
    
    Yields:
        psycopg.Cursor: Obiekt kursora bazy danych.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            yield cursor


@asynccontextmanager
async def get_async_db_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """
    Asynchroniczny kontekstowy manager do połączenia z bazą danych PostgreSQL.
    
    Yields:
        psycopg.AsyncConnection: Obiekt asynchronicznego połączenia z bazą danych.
    """
    async with await psycopg.AsyncConnection.connect(
        **DB_CONFIG, row_factory=dict_row
    ) as conn:
        yield conn


@asynccontextmanager
async def get_async_db_cursor() -> AsyncGenerator[psycopg.AsyncCursor, None]:
    """
    Asynchroniczny kontekstowy manager do uzyskania kursora bazy danych.
    
    Yields:
        psycopg.AsyncCursor: Obiekt asynchronicznego kursora bazy danych.
    """
    async with await get_async_db_connection() as conn:
        async with conn.cursor() as cursor:
            yield cursor


def test_connection() -> bool:
    """
    Testuje połączenie z bazą danych.
    
    Returns:
        bool: True, jeśli połączenie jest udane, False w przeciwnym razie.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
    except Exception as e:
        print(f"Błąd połączenia z bazą danych: {e}")
        return False


if __name__ == "__main__":
    # Test połączenia z bazą danych
    if test_connection():
        print("Połączenie z bazą danych udane!")
    else:
        print("Nie udało się połączyć z bazą danych.")