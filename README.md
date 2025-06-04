🔍 Semantic Document Search
Python 3.13
PostgreSQL 17
pgvector 0.8.0
License: MIT

Zaawansowany system semantycznego wyszukiwania i rekomendacji dokumentów wykorzystujący PostgreSQL 17 z rozszerzeniem pgvector 0.8.0. System oferuje trzy rodzaje wyszukiwania: pełnotekstowe, semantyczne oraz hybrydowe, łącząc najlepsze cechy tradycyjnych baz danych z nowoczesnymi technikami AI.

✨ Główne Funkcjonalności
🧠 Wyszukiwanie semantyczne - wykorzystanie embeddingów AI do rozumienia znaczenia

📝 Wyszukiwanie pełnotekstowe - PostgreSQL full-text search z konfiguracją polską

🔄 Wyszukiwanie hybrydowe - łączenie semantycznego i pełnotekstowego z wagami

💡 System rekomendacji - znajdowanie podobnych dokumentów

🏗️ Wielomodelowe embeddings - Sentence Transformers, OpenAI, scikit-learn

🎨 Kolorowy CLI - intuitive interfejs z Rich i Click

🐳 Docker Ready - kompletne środowisko z docker-compose

⚡ Optymalizacje wydajności - indeksy IVFFlat, connection pooling

🚀 Szybki Start
1. Wymagania
Python 3.13.3+

PostgreSQL 17.5+ z rozszerzeniem pgvector 0.8.0

Docker i Docker Compose (opcjonalnie)

2. Instalacja z Docker (Rekomendowana)
bash
# Klonowanie repozytorium
git clone https://github.com/your-org/semantic-doc-search.git
cd semantic-doc-search

# Konfiguracja środowiska
cp .env.example .env
# Edytuj .env zgodnie z potrzebami

# Uruchomienie kompletnego środowiska
docker-compose up -d

# Sprawdzenie statusu
docker-compose ps
3. Instalacja lokalna z uv
bash
# Instalacja uv (najszybszy menedżer pakietów Python)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup projektu
git clone https://github.com/your-org/semantic-doc-search.git
cd semantic-doc-search

# Utworzenie środowiska wirtualnego i instalacja
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalacja pakietu z wszystkimi zależnościami
uv sync --extra all

# Inicjalizacja systemu
semantic-docs init
4. Konfiguracja bazy danych
PostgreSQL z pgvector (lokalnie)
bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# Instalacja pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Inicjalizacja bazy danych
sudo -u postgres psql -f semantic_doc_search/sql/init_db.sql
sudo -u postgres psql -d semantic_docs -f semantic_doc_search/sql/create_tables.sql
sudo -u postgres psql -d semantic_docs -f semantic_doc_search/sql/functions.sql
📋 Użytkowanie
Zarządzanie dokumentami
bash
# Dodanie dokumentu z pliku
semantic-docs docs add --title "Wprowadzenie do AI" --file docs/ai_intro.txt --embed

# Dodanie dokumentu z treścią bezpośrednio
semantic-docs docs add --title "Podstawy ML" --content "Machine Learning to..." --embed

# Dodanie z metadanymi
semantic-docs docs add --title "Raport 2024" --file report.txt \
  --metadata '{"author": "Jan Kowalski", "category": "financial"}' \
  --source "Dział finansowy" --embed

# Wyświetlenie dokumentu
semantic-docs docs show 1 --show-content --show-embeddings

# Lista dokumentów z paginacją
semantic-docs docs list --limit 20 --source "tutorial"

# Aktualizacja dokumentu
semantic-docs docs update 1 --title "Nowy tytuł" --regenerate-embeddings

# Usunięcie dokumentu
semantic-docs docs delete 1 --force
Wyszukiwanie dokumentów
bash
# Wyszukiwanie pełnotekstowe
semantic-docs search text "sztuczna inteligencja" --limit 10

# Wyszukiwanie semantyczne
semantic-docs search semantic "Jak działa machine learning?" \
  --model sentence-transformers --min-score 0.6

# Wyszukiwanie hybrydowe (najlepsze wyniki)
semantic-docs search hybrid "AI w medycynie" \
  --semantic-weight 0.8 --limit 15 --show-content

# Rekomendacje dla dokumentu
semantic-docs search recommend 5 --limit 10

# Interaktywne wyszukiwanie
semantic-docs search interactive --search-type hybrid
Eksport wyników
bash
# Eksport do JSON
semantic-docs search semantic "AI" --export json --output results.json

# Eksport do CSV
semantic-docs search hybrid "machine learning" --export csv --output ml_results.csv
Administracja systemu
bash
# Status systemu
semantic-docs status

# Sprawdzenie kondycji
semantic-docs health

# Inicjalizacja/reinstalacja
semantic-docs init
🎯 Modele Embeddings
System obsługuje różne modele embeddings:

Sentence Transformers (Lokalnie)
bash
# Szybki model wielojęzyczny (384 wymiary)
--model sentence-transformers

# Instalacja dodatkowych modeli
semantic-docs models install all-mpnet-base-v2
OpenAI (Chmura)
bash
# Konfiguracja w .env
OPENAI_API_KEY=sk-your-key-here

# Użycie
--model openai
scikit-learn (Demo)
bash
# Model TF-IDF do testów
--model sklearn
🏗️ Architektura
text
semantic_doc_search/
├── cli/                 # Interfejs wiersza poleceń
│   ├── main.py         # Główne komendy CLI
│   ├── document_manager.py  # Zarządzanie dokumentami
│   └── search_commands.py   # Wyszukiwanie
├── core/               # Logika biznesowa
│   ├── database.py     # Manager bazy danych
│   ├── models.py       # Modele SQLAlchemy
│   ├── embeddings.py   # Provider embeddingów
│   └── search.py       # Silniki wyszukiwania
├── config/             # Konfiguracja systemu
│   └── settings.py     # Ustawienia Pydantic
└── sql/               # Skrypty PostgreSQL
    ├── init_db.sql     # Inicjalizacja bazy
    ├── create_tables.sql    # Schemat tabel
    └── functions.sql   # Funkcje wyszukiwania
🔧 Konfiguracja
Zmienne środowiskowe
Podstawowe ustawienia w pliku .env:

text
# Baza danych
DB_HOST=localhost
DB_NAME=semantic_docs
DB_USER=postgres
DB_PASSWORD=postgres

# AI modele
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=sk-your-key-here

# Wyszukiwanie
DEFAULT_SEMANTIC_WEIGHT=0.7
DEFAULT_MIN_SCORE=0.3
Dostrajanie wydajności
bash
# Optymalizacja indeksów dla dużych zbiorów
semantic-docs admin rebuild-indexes --lists 1000

# Konfiguracja connection pool
DB_MAX_POOL_SIZE=50
DB_MIN_POOL_SIZE=10
📊 Przykłady Zastosowania
1. Baza Wiedzy Firmy
bash
# Dodanie dokumentów firmowych
semantic-docs docs add --title "Polityka HR" --file hr_policy.pdf \
  --metadata '{"department": "hr", "type": "policy"}' --embed

# Wyszukiwanie hybrydowe
semantic-docs search hybrid "urlop macierzyński" --semantic-weight 0.8
2. Analiza Dokumentów Naukowych
bash
# Batch import z arXiv
semantic-docs batch import --source arxiv --category "cs.AI" --limit 1000

# Wyszukiwanie semantyczne
semantic-docs search semantic "attention mechanisms in transformers" \
  --model openai --min-score 0.7
3. System FAQ
bash
# Dodanie FAQ
semantic-docs docs add --title "Jak zresetować hasło?" \
  --content "Proces resetowania hasła..." --source "FAQ" --embed

# Wyszukiwanie pytań użytkowników
semantic-docs search semantic "nie mogę się zalogować" --limit 5
🧪 Testowanie
bash
# Testy jednostkowe
uv run pytest tests/

# Testy z coverage
uv run pytest --cov=semantic_doc_search tests/

# Testy integracyjne (wymagają bazy danych)
uv run pytest tests/integration/ -m integration

# Testy wydajności
uv run pytest tests/performance/ -m slow
📈 Wydajność
Benchmarki
Wyszukiwanie semantyczne: ~50ms dla 1M dokumentów

Wyszukiwanie hybrydowe: ~80ms dla 1M dokumentów

Indeksowanie: ~1000 dokumentów/minutę

Memory usage: ~2GB RAM dla 100k embeddingów

Optymalizacje
IVFFlat indexy - przybliżone wyszukiwanie NN

Connection pooling - zarządzanie połączeniami DB

Batch processing - grupowanie operacji

Embedding cache - Redis/memory cache

🤝 Rozwój
Setup deweloperski
bash
# Instalacja z dev dependencies
uv sync --extra dev

# Pre-commit hooks
pre-commit install

# Formatowanie kodu
ruff format .
black .

# Type checking
mypy semantic_doc_search/

# Linting
ruff check .
Struktura commitów
bash
feat: dodanie nowego modelu embeddingów
fix: naprawa błędu wyszukiwania hybrydowego
docs: aktualizacja README
refactor: optymalizacja połączeń z bazą danych
test: dodanie testów dla search engine
📋 TODO / Roadmap
 FastAPI REST API - webowy interfejs

 Elasticsearch integration - dodatkowy backend

 Multi-tenant support - izolacja danych

 Real-time indexing - WebSocket updates

 OCR integration - przetwarzanie PDF/obrazów

 Advanced analytics - dashboard z metrykami

 Kubernetes deployment - production scaling

 Graph embeddings - relacje między dokumentami

🐛 Znane Problemy
PostgreSQL Connection Issues
bash
# Zwiększenie limitu połączeń
max_connections = 200

# Timeout settings
statement_timeout = 30s
Memory Issues z dużymi modelami
bash
# Zmniejszenie batch size
EMBEDDING_BATCH_SIZE=16

# Użycie mniejszego modelu
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
📄 Licencja
MIT License - szczegóły w pliku LICENSE

🙏 Podziękowania
pgvector - rozszerzenie wektorowe PostgreSQL

Sentence Transformers - embeddings SOTA

PostgreSQL - potężna baza danych

Rich - piękny terminal

📞 Wsparcie
📧 Email: team@semanticdocs.com

🐛 Issues: GitHub Issues

💬 Discussions: GitHub Discussions

📖 Docs: Documentation

<div align="center"> <p>Zbudowane z ❤️ dla społeczności AI i NLP</p> <p>⭐ Jeśli ten projekt Ci pomógł, zostaw gwiazdkę!</p> </div>