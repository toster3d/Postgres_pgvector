🔍 Semantyczne Wyszukiwanie Dokumentów
System do semantycznego wyszukiwania i rekomendacji dokumentów z wykorzystaniem PostgreSQL i pgvector. Umożliwia przechowywanie dokumentów tekstowych, generowanie wektorowych reprezentacji ich treści (embeddings) oraz wyszukiwanie podobnych dokumentów na podstawie znaczenia semantycznego.

✨ Funkcjonalności
📄 Przechowywanie dokumentów tekstowych w bazie PostgreSQL

🧠 Generowanie embeddings dla dokumentów (Sentence Transformers, OpenAI, sklearn)

🔍 Semantyczne wyszukiwanie podobnych dokumentów

🔄 Hybrydowe wyszukiwanie (łączące full-text search z wyszukiwaniem wektorowym)

💡 System rekomendacji podobnych dokumentów

🖥️ Nowoczesny CLI z kolorowym interfejsem

📊 Eksport wyników do JSON/CSV

⚡ Wydajne indeksy wektorowe (IVFFlat, HNSW)

📋 Wymagania
Python 3.10+

PostgreSQL 17.5+

Rozszerzenie pgvector 0.8.0+

🚀 Instalacja
1. Instalacja PostgreSQL z rozszerzeniem pgvector
Ubuntu/Debian
bash
# Zainstaluj PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-server-dev-all

# Zainstaluj pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
macOS (Homebrew)
bash
# Zainstaluj PostgreSQL
brew install postgresql

# Zainstaluj pgvector
brew install pgvector
Windows
Pobierz PostgreSQL z oficjalnej strony i zainstaluj pgvector zgodnie z instrukcjami.

2. Inicjalizacja bazy danych
bash
# Uruchom PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Inicjalizuj bazę danych
psql -U postgres -f sql/init_db.sql
psql -U postgres -d semantic_docs -f sql/create_tables.sql
3. Instalacja pakietu Python
bash
# Utwórz wirtualne środowisko
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Zainstaluj pakiet
pip install -e .

# Lub zainstaluj wymagania ręcznie
pip install -r requirements.txt
4. Konfiguracja
bash
# Skopiuj przykładową konfigurację
cp .env.example .env

# Edytuj konfigurację
nano .env
Przykładowa konfiguracja:

text
DB_NAME=semantic_docs
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Opcjonalnie dla OpenAI
OPENAI_API_KEY=your_api_key_here
5. Inicjalizacja systemu
bash
# Zainicjalizuj bazę danych i sprawdź status
semantic-docs init
semantic-docs status
🎯 Użycie
Zarządzanie dokumentami
Dodawanie dokumentu z treścią
bash
semantic-docs docs add \
  --title "Wprowadzenie do AI" \
  --content "Sztuczna inteligencja to dziedzina informatyki..." \
  --embed
Dodawanie dokumentu z pliku
bash
semantic-docs docs add \
  --title "Raport roczny 2024" \
  --file "./documents/raport_2024.txt" \
  --source "Dział finansowy" \
  --embed \
  --model all-mpnet-base-v2
Wyświetlanie dokumentu
bash
semantic-docs docs show 1 --show-content --show-embeddings
Aktualizacja dokumentu
bash
semantic-docs docs update 1 \
  --title "Nowy tytuł" \
  --regenerate-embeddings
Usunięcie dokumentu
bash
semantic-docs docs delete 1 --force
Lista dokumentów
bash
semantic-docs docs list --limit 20 --format table
Wyszukiwanie dokumentów
Wyszukiwanie pełnotekstowe
bash
semantic-docs search text "sztuczna inteligencja" --show-content
Wyszukiwanie semantyczne
bash
semantic-docs search semantic \
  "Jaka jest natura świadomości?" \
  --model all-mpnet-base-v2 \
  --show-content
Wyszukiwanie hybrydowe
bash
semantic-docs search hybrid \
  "machine learning i deep learning" \
  --semantic-weight 0.7 \
  --show-content
Rekomendacje podobnych dokumentów
bash
semantic-docs search recommend 1 --limit 5 --show-content
Eksport wyników
bash
# Eksport do JSON
semantic-docs search semantic "AI ethics" \
  --export json \
  --output results.json

# Eksport do CSV
semantic-docs search hybrid "blockchain technology" \
  --export csv \
  --output blockchain_results.csv
🧠 Modele embeddings
System obsługuje różne modele do generowania embeddings:

Sentence Transformers (domyślnie)
bash
# Szybki model angielski (384 wymiary)
semantic-docs docs add --title "Document" --content "Content" \
  --embed --model all-MiniLM-L6-v2

# Wysokiej jakości model angielski (768 wymiarów)
semantic-docs docs add --title "Document" --content "Content" \
  --embed --model all-mpnet-base-v2

# Model wielojęzyczny (polski obsługiwany)
semantic-docs docs add --title "Dokument" --content "Treść po polsku" \
  --embed --model paraphrase-multilingual-mpnet-base-v2
OpenAI (wymaga klucza API)
bash
semantic-docs docs add --title "Document" --content "Content" \
  --embed --model text-embedding-3-small
Scikit-learn (demonstracyjny)
bash
semantic-docs docs add --title "Document" --content "Content" \
  --embed --model tfidf-vectorizer
📊 Zarządzanie indeksami
bash
# Utwórz indeksy wektorowe dla lepszej wydajności
semantic-docs create-indexes

# Wymuś ponowne utworzenie indeksów
semantic-docs create-indexes --force
🔧 Przykłady użycia
Scenariusz 1: Baza artykułów naukowych
bash
# Dodaj artykuły
semantic-docs docs add \
  --title "Teoria względności Einsteina" \
  --file articles/relativity.txt \
  --source "Physics Journal" \
  --embed

semantic-docs docs add \
  --title "Mechanika kwantowa - wprowadzenie" \
  --file articles/quantum.txt \
  --source "Science Magazine" \
  --embed

# Znajdź podobne artykuły
semantic-docs search semantic \
  "Związek między grawitacją a mechaniką kwantową" \
  --show-content \
  --export json \
  --output physics_search.json
Scenariusz 2: Dokumenty firmowe
bash
# Dodaj dokumenty
semantic-docs docs add \
  --title "Raport finansowy Q3 2024" \
  --file reports/q3_2024.txt \
  --source "Dział finansowy" \
  --metadata '{"quarter": "Q3", "year": 2024, "department": "finance"}' \
  --embed

semantic-docs docs add \
  --title "Strategia rozwoju 2025" \
  --file docs/strategy_2025.txt \
  --source "Zarząd" \
  --metadata '{"type": "strategy", "year": 2025}' \
  --embed

# Wyszukiwanie hybrydowe
semantic-docs search hybrid \
  "prognozy finansowe na przyszły rok" \
  --semantic-weight 0.6 \
  --show-content

# Eksportuj wyniki strategiczne
semantic-docs search semantic "cele strategiczne rozwoju" \
  --export csv \
  --output strategy_matches.csv
Scenariusz 3: Dokumentacja techniczna
bash
# Dodaj dokumentację
for file in docs/technical/*.md; do
  semantic-docs docs add \
    --title "$(basename "$file" .md)" \
    --file "$file" \
    --source "Technical Documentation" \
    --embed \
    --model all-mpnet-base-v2
done

# Znajdź dokumenty podobne do konkretnego
semantic-docs search recommend 5 --limit 10 --show-content
📈 Monitorowanie i diagnostyka
bash
# Sprawdź status systemu
semantic-docs status

# Wyświetl historię wyszukiwań
semantic-docs history --limit 20

# Sprawdź wersję
semantic-docs version
⚙️ Konfiguracja zaawansowana
Dostrajanie wydajności
text
# W pliku .env
EMBEDDING_BATCH_SIZE=64        # Większe batche dla GPU
DB_MAX_POOL_SIZE=50           # Więcej połączeń dla dużego ruchu
IVFFLAT_LISTS=200             # Więcej list dla większych zbiorów
CHUNK_SIZE=1500               # Większe chunki dla długich dokumentów
Optymalizacja dla dużych zbiorów
bash
# Utwórz indeksy z większą liczbą list
semantic-docs create-indexes --force

# Użyj modeli o wyższej wymiarowości dla lepszej jakości
semantic-docs docs add --title "..." --content "..." \
  --embed --model all-mpnet-base-v2  # 768 wymiarów zamiast 384
🚨 Rozwiązywanie problemów
Problem: Brak połączenia z bazą danych
bash
# Sprawdź status PostgreSQL
sudo systemctl status postgresql

# Sprawdź konfigurację połączenia
semantic-docs status
Problem: Błąd instalacji pgvector
bash
# Upewnij się, że masz zainstalowane dev headers
sudo apt install postgresql-server-dev-all  # Ubuntu/Debian
Problem: Brak modeli Sentence Transformers
bash
# Ręczne pobieranie modeli
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
Problem: Powolne wyszukiwanie
bash
# Utwórz indeksy wektorowe
semantic-docs create-indexes

# Sprawdź statystyki bazy danych
semantic-docs status --verbose
🎛️ Parametry CLI
Globalne opcje
--verbose/-v: Szczegółowe logowanie

--quiet/-q: Tylko błędy

--config-file: Ścieżka do pliku .env

Zarządzanie dokumentami
docs add: Dodaj dokument

docs show: Wyświetl dokument

docs update: Aktualizuj dokument

docs delete: Usuń dokument

docs list: Lista dokumentów

Wyszukiwanie
search text: Wyszukiwanie pełnotekstowe

search semantic: Wyszukiwanie semantyczne

search hybrid: Wyszukiwanie hybrydowe

search recommend: Rekomendacje

Narzędzia
init: Inicjalizacja systemu

status: Status systemu

create-indexes: Tworzenie indeksów

history: Historia wyszukiwań

version: Informacje o wersji

🛡️ Ograniczenia
System jest zoptymalizowany dla języka angielskiego (indeksy pełnotekstowe)

Domyślne modele mają ograniczoną długość dokumentów (512 tokenów dla BERT-based)

W przypadku dużych kolekcji dokumentów zalecane jest wykorzystanie modeli o wyższej wymiarowości

OpenAI API ma limity rate-limiting i kosztów

🔗 Linki
Dokumentacja pgvector

Sentence Transformers

PostgreSQL

OpenAI Embeddings

📄 Licencja
MIT License - zobacz plik LICENSE dla szczegółów.

🤝 Wkład w projekt
Chętnie przyjmujemy Pull Requesty! Zobacz CONTRIBUTING.md dla wytycznych.

💬 Wsparcie
🐛 Zgłoś bug: GitHub Issues

📧 Email: support@example.com

💬 Discord: Link do serwera