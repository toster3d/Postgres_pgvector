# Semantyczne Wyszukiwanie Dokumentów

System do semantycznego wyszukiwania i rekomendacji dokumentów z wykorzystaniem PostgreSQL i pgvector. Umożliwia przechowywanie dokumentów tekstowych, generowanie wektorowych reprezentacji ich treści (embeddings) oraz wyszukiwanie podobnych dokumentów na podstawie znaczenia semantycznego.

## Funkcjonalności

- Przechowywanie dokumentów tekstowych w bazie PostgreSQL
- Generowanie wektorowych reprezentacji (embeddings) dla dokumentów
- Semantyczne wyszukiwanie podobnych dokumentów
- Hybrydowe wyszukiwanie (łączące full-text search z wyszukiwaniem wektorowym)
- System rekomendacji podobnych dokumentów
- Wizualizacja podobieństwa dokumentów

## Wymagania

- Python 3.13+
- PostgreSQL 17.4+
- Rozszerzenie pgvector dla PostgreSQL

## Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/username/semantic-search.git
cd semantic-search
```

2. Utwórz wirtualne środowisko i zainstaluj zależności:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

3. Skonfiguruj zmienne środowiskowe:
```bash
cp .env.example .env
```
Następnie edytuj plik `.env` i dodaj swoje dane dostępowe do bazy danych oraz klucz API (jeśli używasz OpenAI).

4. Inicjalizacja bazy danych:
```bash
cd database/sql
psql -U postgres -d postgres -f init_db.sql
psql -U postgres -d semantic_search -f create_tables.sql
```

## Uruchomienie

1. Uruchom API:
```bash
cd api
uvicorn main:app --reload
```

2. Otwórz przeglądarkę:
```
http://localhost:8000/docs
```

## Struktura projektu

- `database/` - moduł zarządzania bazą danych PostgreSQL
- `embeddings/` - moduł generowania wektorowych reprezentacji dokumentów
- `search/` - implementacja wyszukiwania semantycznego i hybrydowego
- `visualization/` - wizualizacje podobieństwa dokumentów
- `api/` - interfejs API REST z użyciem FastAPI
- `data/` - skrypty do generowania przykładowych danych
- `tests/` - testy jednostkowe

## Licencja

MIT