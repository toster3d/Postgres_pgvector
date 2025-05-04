"""
Główny plik API.
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.database.db import init_db
from src.api.routes import documents, categories, search, visualization

# Inicjalizacja bazy danych przy starcie
try:
    init_db()
except Exception as e:
    print(f"Błąd inicjalizacji bazy danych: {e}")

# Utwórz aplikację FastAPI
app = FastAPI(
    title="Semantyczne Wyszukiwanie Dokumentów",
    description="API do semantycznego wyszukiwania i rekomendacji dokumentów z wykorzystaniem PostgreSQL i pgvector",
    version="1.0.0"
)

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji należy ograniczyć do konkretnych domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obsługa wyjątków
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Globalny handler wyjątków."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Wystąpił nieoczekiwany błąd: {str(exc)}"}
    )

# Podstawowy endpoint zdrowia
@app.get("/", tags=["health"])
async def root():
    """Zwraca podstawowe informacje o API."""
    return {
        "status": "healthy",
        "message": "Semantyczne Wyszukiwanie Dokumentów API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

# Rejestracja routerów
app.include_router(categories.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(visualization.router)


if __name__ == "__main__":
    """Uruchomienie aplikacji w trybie deweloperskim."""
    import uvicorn
    
    # Konfiguracja z zmiennych środowiskowych
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    # Uruchom serwer
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True) 