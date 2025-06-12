-- Inicjalizacja bazy danych PostgreSQL z rozszerzeniem pgvector
-- Uruchom jako superuser: psql -U postgres -f init_db.sql

-- Tworzenie bazy danych
DROP DATABASE IF EXISTS semantic_docs;
CREATE DATABASE semantic_docs 
    WITH 
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

-- Połączenie z nową bazą danych
\c semantic_docs;

-- Instalacja rozszerzenia pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tworzenie użytkownika aplikacji (opcjonalne)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'semantic_user') THEN
        CREATE ROLE semantic_user WITH LOGIN PASSWORD 'semantic_password';
    END IF;
END
$$;

-- Nadanie uprawnień
GRANT CONNECT ON DATABASE semantic_docs TO semantic_user;
GRANT USAGE ON SCHEMA public TO semantic_user;
GRANT CREATE ON SCHEMA public TO semantic_user;

-- Informacja o wersji pgvector
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Sprawdzenie dostępnych operatorów dla typu vector
SELECT oprname, oprleft::regtype, oprright::regtype, oprresult::regtype
FROM pg_operator 
WHERE oprname IN ('<->', '<=>', '<#>')
ORDER BY oprname;