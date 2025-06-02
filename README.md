# Semantyczne Wyszukiwanie Dokumentów

System do semantycznego wyszukiwania i rekomendacji dokumentów z wykorzystaniem PostgreSQL i pgvector. Umożliwia przechowywanie dokumentów tekstowych, generowanie wektorowych reprezentacji ich treści (embeddings) oraz wyszukiwanie podobnych dokumentów na podstawie znaczenia semantycznego.

## Funkcjonalności

- Przechowywanie dokumentów tekstowych w bazie PostgreSQL
- Generowanie wektorowych reprezentacji (embeddings) dla dokumentów
- Semantyczne wyszukiwanie podobnych dokumentów
- Hybrydowe wyszukiwanie (łączące full-text search z wyszukiwaniem wektorowym)
- System rekomendacji podobnych dokumentów

## Wymagania

- Python 3.8+
- PostgreSQL 12+
- Rozszerzenie pgvector dla PostgreSQL

## Instalacja

### 1. Instalacja PostgreSQL z rozszerzeniem pgvector

Najpierw zainstaluj PostgreSQL zgodnie z instrukcjami dla Twojego systemu operacyjnego:
[https://www.postgresql.org/download/](https://www.postgresql.org/download/)

Następnie zainstaluj rozszerzenie pgvector:
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

### 2. Inicjalizacja bazy danych

Uruchom skrypty SQL w następującej kolejności:
```bash
psql -U postgres -f semantic_doc_search/sql/init_db.sql
psql -U postgres -d semantic_docs -f semantic_doc_search/sql/create_tables.sql
```

### 3. Instalacja pakietu Python

```bash
# Utworzenie wirtualnego środowiska
python -m venv venv
source venv/bin/activate  # W Windows: venv\Scripts\activate

# Instalacja pakietu
pip install -e .
```

### 4. Konfiguracja (opcjonalna)

Jeśli chcesz korzystać z OpenAI do generowania embeddings, utwórz plik .env w katalogu głównym projektu:
```
DB_NAME=semantic_docs
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
OPENAI_API_KEY=your_api_key_here
```

## Użycie

System działa jako narzędzie wiersza poleceń (CLI).

### Zarządzanie dokumentami

1. Dodawanie dokumentu:
```bash
# Dodanie dokumentu z zawartością podaną jako argument
semantic-docs docs add --title "Mój dokument" --content "To jest treść mojego dokumentu." --embed

# Dodanie dokumentu z zawartością wczytaną z pliku
semantic-docs docs add --title "Mój dokument z pliku" --file path/to/document.txt --embed
```

2. Wyświetlanie dokumentu:
```bash
semantic-docs docs show 1  # Gdzie 1 to ID dokumentu
```

3. Aktualizacja dokumentu:
```bash
semantic-docs docs update 1 --title "Nowy tytuł" --regenerate-embeddings
```

4. Usunięcie dokumentu:
```bash
semantic-docs docs delete 1
```

5. Listowanie dokumentów:
```bash
semantic-docs docs list --limit 20
```

### Wyszukiwanie dokumentów

1. Wyszukiwanie pełnotekstowe:
```bash
semantic-docs search text "słowa kluczowe"
```

2. Wyszukiwanie semantyczne:
```bash
semantic-docs search semantic "Jaka jest natura świadomości?"
```

3. Wyszukiwanie hybrydowe (semantyczne + pełnotekstowe):
```bash
semantic-docs search hybrid "Sztuczna inteligencja i świadomość" --semantic-weight 0.7
```

4. Rekomendacje podobnych dokumentów:
```bash
semantic-docs search recommend 1  # Znajdź dokumenty podobne do dokumentu o ID 1
```

### Eksport wyników

Możesz eksportować wyniki wyszukiwania do formatu JSON:
```bash
semantic-docs search semantic "Zapytanie" --export json --output results.json
```

## Modele embeddings

System obsługuje różne modele do generowania embeddings:

1. **Sentence Transformers** (domyślnie all-MiniLM-L6-v2)
```bash
semantic-docs docs add --title "Dokument" --content "Treść" --embed --model sentence-transformers
```

2. **OpenAI embeddings** (wymaga klucza API)
```bash
semantic-docs docs add --title "Dokument" --content "Treść" --embed --model openai
```

3. **Scikit-learn** (prosty model TF-IDF do celów demonstracyjnych)
```bash
semantic-docs docs add --title "Dokument" --content "Treść" --embed --model sklearn
```

## Przykłady użycia

### Scenariusz 1: Baza artykułów naukowych

```bash
# Dodaj dokumenty
semantic-docs docs add --title "Teoria względności" --file articles/relativity.txt --embed
semantic-docs docs add --title "Mechanika kwantowa" --file articles/quantum.txt --embed

# Znajdź podobne artykuły
semantic-docs search semantic "Grawitacja a mechanika kwantowa" --show-content
```

### Scenariusz 2: Analiza dokumentów firmowych

```bash
# Dodaj dokumenty
semantic-docs docs add --title "Raport 2023" --file reports/2023.txt --source "Dział finansowy" --embed
semantic-docs docs add --title "Strategia 2024" --file reports/strategy.txt --source "Zarząd" --embed

# Wyszukiwanie hybrydowe
semantic-docs search hybrid "Prognozy finansowe na 2024" --semantic-weight 0.6

# Eksport wyników
semantic-docs search semantic "Cele strategiczne" --export json --output strategy_matches.json
```

## Ograniczenia

- System jest zoptymalizowany dla języka angielskiego (indeksy pełnotekstowe)
- Domyślne modele mają ograniczoną długość dokumentów, które mogą przetworzyć
- W przypadku dużych kolekcji dokumentów, zalecane jest wykorzystanie bardziej zaawansowanych modeli embeddings