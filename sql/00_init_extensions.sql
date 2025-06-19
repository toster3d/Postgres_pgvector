\echo 'Inicjalizacja rozszerzeń PostgreSQL...'

-- Tworzenie rozszerzenia pgvector dla operacji wektorowych
CREATE EXTENSION IF NOT EXISTS vector;

-- Sprawdzenie wersji pgvector
\echo 'Wersja pgvector:'
SELECT extversion FROM pg_extension WHERE extname = 'vector';

\echo 'Rozszerzenia zostały pomyślnie zainstalowane!'
\echo 'Dostępne typy danych wektorowych: vector, halfvec, sparsevec'