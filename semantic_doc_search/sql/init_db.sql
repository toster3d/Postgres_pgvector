-- semantic_doc_search/sql/init_db.sql
-- Skrypt inicjalizacji bazy danych dla systemu semantycznego wyszukiwania dokumentów
-- Kompatybilny z PostgreSQL 17+ i pgvector 0.8.0

-- Tworzenie bazy danych
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'semantic_docs') THEN
        PERFORM dblink_exec('dbname=' || current_database(), 'CREATE DATABASE semantic_docs');
    END IF;
END
$$;

-- Połączenie z nową bazą danych
\c semantic_docs;

-- Włączenie niezbędnych rozszerzeń
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Konfiguracja dla języka polskiego (opcjonalna)
CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS polish_advanced (COPY = pg_catalog.polish);

-- Poprawa konfiguracji wyszukiwania dla języka polskiego
-- ALTER TEXT SEARCH CONFIGURATION polish_advanced
--     ALTER MAPPING FOR asciiword, asciihword, hword_asciipart, word, hword, hword_part
--     WITH unaccent, polish_stem;

-- Tworzenie roli dla aplikacji (opcjonalne)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'semantic_app') THEN
        CREATE ROLE semantic_app LOGIN PASSWORD 'semantic_pass';
    END IF;
END
$$;

-- Przyznanie uprawnień
GRANT CONNECT ON DATABASE semantic_docs TO semantic_app;
GRANT USAGE ON SCHEMA public TO semantic_app;
GRANT CREATE ON SCHEMA public TO semantic_app;

-- Pokaż informacje o zainstalowanych rozszerzeniach
SELECT 
    name,
    installed_version,
    comment
FROM pg_available_extensions 
WHERE name IN ('vector', 'pg_trgm', 'btree_gin', 'unaccent')
    AND installed_version IS NOT NULL
ORDER BY name;