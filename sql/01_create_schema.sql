\echo 'Tworzenie schematu bazy danych...'

-- Tworzenie dedykowanego schematu dla wyszukiwania semantycznego
CREATE SCHEMA IF NOT EXISTS semantic;

-- Ustawienie domyślnego schematu search_path
ALTER DATABASE semantic_docs SET search_path TO semantic, public;

-- Tworzenie typu enum dla statusu dokumentu
CREATE TYPE semantic.document_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'error'
);

-- Tworzenie typu enum dla typu dokumentu
-- Chociaż obecny dataset skupia się na artykułach, typ ENUM
-- umożliwia przyszłe rozszerzenie bazy o inne rodzaje publikacji (np. raporty, książki).
CREATE TYPE semantic.document_type AS ENUM (
    'article',
    'paper',
    'report',
    'thesis',
    'book_chapter',
    'review',
    'other'
);

-- Tworzenie typu enum dla modeli embeddingu
-- Mimo że aktualnie używany jest tylko jeden model ('all-MiniLM-L6-v2'),
-- typ ENUM zapewnia integralność danych i przygotowuje bazę na przyszłe
-- eksperymenty z innymi modelami bez konieczności zmiany schematu.
CREATE TYPE semantic.embedding_model AS ENUM (
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'text-embedding-3-small',
    'text-embedding-3-large',
    'text-embedding-ada-002'
);

\echo 'Schema semantic został utworzony!'
\echo 'Typy enum zostały zdefiniowane!'
\echo 'Search path został ustawiony na: semantic, public'