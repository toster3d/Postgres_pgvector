# Semantyczny System Wyszukiwania Dokumentów Naukowych

## 1. Wprowadzenie i Cel Projektu

Celem projektu było stworzenie zaawansowanego systemu do wyszukiwania w korpusie dokumentów naukowych. W odróżnieniu od tradycyjnych wyszukiwarek opartych wyłącznie na słowach kluczowych, głównym założeniem było zaimplementowanie **wyszukiwania semantycznego**. Pozwala ono na odnajdywanie dokumentów na podstawie ich znaczenia i kontekstu, a nie tylko dokładnego dopasowania fraz.

System został zaprojektowany tak, aby odpowiadać na zapytania w języku naturalnym (np. "wpływ zmian klimatu na rolnictwo") i zwracać artykuły, które merytorycznie odpowiadają na to zapytanie, nawet jeśli nie zawierają dokładnie tych samych słów. **Należy podkreślić, że ze względu na użyty model językowy, system działa dla zapytań i dokumentów w języku angielskim.**

## 2. Architektura i Uzasadnienie Wyboru Technologii

System został zbudowany w oparciu o skonteneryzowaną architekturę z wykorzystaniem następujących technologii:

- **Baza Danych: PostgreSQL (wersja 17)**
  - **Uzasadnienie:** PostgreSQL został wybrany jako fundament systemu ze względu na swoją niezawodność, zgodność ze standardem SQL oraz ogromne możliwości rozbudowy za pomocą rozszerzeń.

- **Wyszukiwanie Wektorowe: Rozszerzenie `pgvector`**
  - **Uzasadnienie:** `pgvector` to rozszerzenie do PostgreSQL, które dodaje nowy typ danych `VECTOR` oraz możliwość wykonywania ultra-szybkiego wyszukiwania podobieństwa wektorowego (ANN). Zastosowanie `pgvector` było kluczową decyzją architektoniczną, pozwalającą na integrację wyszukiwania semantycznego bezpośrednio z silnikiem bazy danych.

- **Konteneryzacja: Docker i Docker Compose**
  - **Uzasadnienie:** Cały projekt został zamknięty w kontenerach, aby zapewnić **reprodukowalność** i **izolację środowiska**.

- **Język i Biblioteki: Python 3.13**
  - **Uzasadnienie:** Python jest standardem w dziedzinie Data Science i NLP.
    - `sentence-transformers`: Biblioteka do generowania wysokiej jakości **embeddingów** – wektorowych reprezentacji tekstu.
    - `psycopg`: Nowoczesny, wydajny sterownik do komunikacji z bazą PostgreSQL.
    - `rich`: Umożliwia tworzenie przyjaznych i czytelnych interfejsów wiersza poleceń (CLI).

## 3. Opis Zbioru Danych (Dataset)

W projekcie wykorzystano publicznie dostępny zbiór danych **Elsevier OA CC-BY Corpus**, dostępny na platformie Hugging Face.

- **Oficjalna strona:** [Elsevier Digital Commons Data](https://elsevier.digitalcommonsdata.com/datasets/zm33cdndxs/3)
- **Licencja:** CC-BY-4.0
- **Zawartość:** Zbiór zawiera ponad 40,000 artykułów naukowych z różnych dziedzin. Każdy rekord zawiera m.in. tytuł, abstrakt, pełną treść, słowa kluczowe, obszary tematyczne oraz listę najważniejszych punktów wskazanych przez autorów.

## 4. Instrukcja Uruchomienia i Użycia

#### Wymagania
- Zainstalowany Docker i Docker Compose.

#### Kroki Uruchomienia
1. **Klonowanie repozytorium**
   Sklonuj repozytorium na swój lokalny komputer.

2. **Budowa i uruchomienie kontenerów**
   W głównym katalogu projektu uruchom polecenie, które zbuduje obrazy i uruchomi kontenery w tle:
   ```bash
   docker-compose up -d --build
   ```

3. **Ładowanie danych do bazy**
   Po uruchomieniu kontenerów, załaduj dane do bazy. Można ograniczyć liczbę dokumentów za pomocą flagi `--limit` dla celów testowych.
   ```bash
   # Załaduj 100 - 1000 dokumentów (rekomendowane do testów - ładowanie pełnego zbioru danych zajmuje wiele godzin)
   docker-compose exec semantic-cli python scripts/load_elsevier_data.py --limit 100

   # Załaduj 1000 dokumentów
   docker-compose exec semantic-cli python scripts/load_elsevier_data.py --limit 1000
   ```
   *Uwaga: Ładowanie pełnego datasetu może potrwać bardzo długo.*

4. **Testowanie wyszukiwarki**
   Gdy dane są załadowane, można przetestować działanie systemu.

#### Przykładowe Komendy Użycia

- **Wyświetlenie statystyk bazy danych:**
  ```bash
  docker-compose exec semantic-cli python scripts/test_search.py --stats
  ```

- **Wyszukiwanie semantyczne (domyślne):**
  ```bash
  docker-compose exec semantic-cli python scripts/test_search.py --query "impact of artificial intelligence on healthcare"
  ```

- **Wyszukiwanie pełnotekstowe:**
    ```bash
    docker-compose exec semantic-cli python scripts/test_search.py --query "nanoparticles" --type fulltext
    ```

- **Wyszukiwanie hybrydowe (najlepsze rezultaty):**
  ```bash
  docker-compose exec semantic-cli python scripts/test_search.py --query "carbon capture technology" --type hybrid
  ```

- **Uruchomienie trybu interaktywnego:**
  ```bash
  # Flaga -it jest kluczowa dla działania trybu interaktywnego
  docker-compose exec -it semantic-cli python scripts/test_search.py --interactive
  ```

## 5. Architektura Bazy Danych

Baza danych została zaprojektowana w sposób znormalizowany, aby efektywnie przechowywać zarówno metadane dokumentów, jak i ich reprezentacje wektorowe.

#### Struktura Projektu

```
.
├── docker/                     # Konfiguracja Docker dla PostgreSQL
│   ├── docker_pg_hba.conf      # Ustawienia uwierzytelniania
│   └── docker_postgres.conf    # Optymalizacja wydajności PostgreSQL
├── scripts/                    # Skrypty Python
│   ├── load_elsevier_data.py   # Skrypt ETL do ładowania danych
│   └── test_search.py          # Skrypt do testowania wyszukiwania
├── sql/                        # Skrypty inicjalizacyjne SQL
│   ├── 00_init_extensions.sql  # Włączenie rozszerzenia 'vector'
│   ├── 01_create_schema.sql    # Tworzenie schematu 'semantic'
│   └── 02_create_tables.sql    # Definicje tabel i indeksów
├── docker-compose.yml          # Plik orkiestracji usług Docker
├── Dockerfile                  # Definicja kontenera dla aplikacji Python
└── pyproject.toml              # Zarządzanie zależnościami i konfiguracja projektu
```

#### Schemat i Tabele (`sql/*.sql`)

- **Schema `semantic`:** Utworzono dedykowaną przestrzeń nazw, aby oddzielić tabele projektu od domyślnego schematu `public`.

- **Tabela `documents`:** Przechowuje metadane każdego artykułu. Zawiera m.in. kolumny `title`, `abstract`, `full_text`, `pub_year`, `keywords` oraz `author_highlights`. Kolumna `doc_id` (DOI) posiada ograniczenie `UNIQUE`, co w połączeniu z klauzulą `ON CONFLICT` w skrypcie ładującym zapobiega tworzeniu duplikatów.

- **Tabela `document_embeddings`:** Przechowuje fragmenty ("chunki") tekstu i odpowiadające im embeddingi. Jest połączona z tabelą `documents` relacją jeden-do-wielu. Kluczowa kolumna to `embedding VECTOR(384)`, która przechowuje wektorową reprezentację znaczenia tekstu.

#### Indeksy i Optymalizacja

Zastosowano trzy kluczowe typy indeksów w pliku `02_create_tables.sql`, aby zapewnić maksymalną wydajność zapytań:

1.  **Indeks B-Tree:** Na kolumnie `doc_id` w tabeli `documents` dla szybkiej obsługi `ON CONFLICT`.
2.  **Indeks GIN (`fts_idx`):** Na kolumnach `tsvector` do ultra-szybkiego wyszukiwania pełnotekstowego (FTS).
3.  **Indeks HNSW (`embeddings_embedding_idx`):** Na kolumnie `embedding` do błyskawicznego wyszukiwania wektorowego (ANN). Zastosowano metrykę odległości cosinusowej (`vector_cosine_ops`), która jest standardem w zadaniach NLP.

## 6. Proces Przetwarzania Danych i Wyszukiwania

#### Potok Przetwarzania Danych (`load_elsevier_data.py`)

Skrypt realizuje klasyczny proces ETL (Extract, Transform, Load):
1.  **Extract:** Pobiera dane z Hugging Face.
2.  **Transform:** Dla każdego dokumentu dzieli jego treść na mniejsze fragmenty (**chunking**), a następnie generuje dla nich **embeddingi** (reprezentacje wektorowe) za pomocą modelu `all-MiniLM-L6-v2`.
3.  **Load:** Zapisuje przetworzone dane (metadane i embeddingi) do bazy PostgreSQL.

#### Mechanizmy Wyszukiwania (`test_search.py`)

Skrypt testujący implementuje trzy strategie wyszukiwania:
1.  **Semantyczne:** Konwertuje zapytanie użytkownika na wektor i szuka w bazie tekstów o najbliższym znaczeniu, używając operatora odległości kosinusowej `<=>` z `pgvector`.
2.  **Pełnotekstowe:** Wykorzystuje wbudowane w PostgreSQL funkcje FTS do znalezienia dokładnych słów kluczowych.
3.  **Hybrydowe:** Łączy wyniki z obu powyższych metod, tworząc jeden, trafniejszy ranking.

## 7. Podsumowanie

Projekt z powodzeniem demonstruje, jak można zbudować system do wyszukiwania semantycznego, opierając się na duecie PostgreSQL + pgvector.

---
*Projekt utworzony na potrzeby zaliczenia przedmiotu "Bazy i Hurtownie Danych" na studiach podyplomowych Data Science.*
*Dataset Elsevier dostępny na licencji CC BY 4.0.*