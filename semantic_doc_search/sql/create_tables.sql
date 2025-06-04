-- semantic_doc_search/sql/create_tables.sql
-- Definicje tabel dla systemu semantycznego wyszukiwania dokumentów
-- Optymalizowane dla PostgreSQL 17+ z pgvector 0.8.0

-- Usunięcie istniejących tabel (ostrożnie!)
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS document_embeddings CASCADE;
DROP TABLE IF EXISTS embedding_models CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Tabela główna dla dokumentów
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    search_vector TSVECTOR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ograniczenia
    CONSTRAINT documents_title_not_empty CHECK (LENGTH(TRIM(title)) > 0),
    CONSTRAINT documents_content_not_empty CHECK (LENGTH(TRIM(content)) > 0)
);

-- Tabela modeli embeddings (konfiguracja)
CREATE TABLE embedding_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    provider VARCHAR(50) NOT NULL, -- 'sentence-transformers', 'openai', 'sklearn'
    dimension INTEGER NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ograniczenia
    CONSTRAINT embedding_models_dimension_positive CHECK (dimension > 0),
    CONSTRAINT embedding_models_provider_valid CHECK (provider IN ('sentence-transformers', 'openai', 'sklearn'))
);

-- Tabela embeddingów wektorowych
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
    embedding vector(384), -- Domyślnie 384 wymiary dla all-MiniLM-L6-v2
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unikalność - jeden embedding na dokument per model
    CONSTRAINT document_embeddings_unique UNIQUE (document_id, model_id)
);

-- Tabela historii wyszukiwań (opcjonalna, dla analiz)
CREATE TABLE search_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    search_type VARCHAR(20) NOT NULL, -- 'text', 'semantic', 'hybrid'
    model_name VARCHAR(100),
    semantic_weight DECIMAL(3,2),
    results_count INTEGER NOT NULL DEFAULT 0,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ograniczenia
    CONSTRAINT search_history_type_valid CHECK (search_type IN ('text', 'semantic', 'hybrid')),
    CONSTRAINT search_history_weight_valid CHECK (semantic_weight IS NULL OR (semantic_weight >= 0 AND semantic_weight <= 1))
);

-- === INDEKSY PODSTAWOWE ===

-- Indeksy dla tabeli documents
CREATE INDEX idx_documents_title ON documents USING GIN (to_tsvector('polish', title));
CREATE INDEX idx_documents_content_gin ON documents USING GIN (search_vector);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);
CREATE INDEX idx_documents_title_trigram ON documents USING GIN (title gin_trgm_ops);

-- Indeksy dla tabeli document_embeddings
CREATE INDEX idx_embeddings_document_id ON document_embeddings (document_id);
CREATE INDEX idx_embeddings_model_id ON document_embeddings (model_id);

-- Indeksy wektorowe - będą tworzone dynamicznie przez funkcje
-- CREATE INDEX idx_embeddings_vector_cosine ON document_embeddings 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Indeksy dla tabeli search_history
CREATE INDEX idx_search_history_created_at ON search_history (created_at DESC);
CREATE INDEX idx_search_history_query_text ON search_history USING GIN (to_tsvector('polish', query_text));
CREATE INDEX idx_search_history_type ON search_history (search_type);

-- === TRIGGERY ===

-- Automatyczne generowanie search_vector dla dokumentów
CREATE OR REPLACE FUNCTION update_document_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('polish', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, ''));
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_document_search_vector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_document_search_vector();

-- === WSTAWIENIE POCZĄTKOWYCH DANYCH ===

-- Domyślne modele embeddings
INSERT INTO embedding_models (name, provider, dimension, description) VALUES
    ('all-MiniLM-L6-v2', 'sentence-transformers', 384, 'Szybki model wielojęzyczny, dobry stosunek jakość/szybkość'),
    ('all-mpnet-base-v2', 'sentence-transformers', 768, 'Wyższa jakość, wolniejszy od MiniLM'),
    ('text-embedding-3-small', 'openai', 1536, 'OpenAI model - mały, szybki'),
    ('text-embedding-3-large', 'openai', 3072, 'OpenAI model - duży, najwyższa jakość'),
    ('tfidf-sklearn', 'sklearn', 1000, 'Prosty model TF-IDF do testów i demonstracji')
ON CONFLICT (name) DO NOTHING;

-- === UPRAWNIENIA ===

-- Przyznanie uprawnień do tabel
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO semantic_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO semantic_app;

-- === STATYSTYKI ===

-- Analiza tabel dla optymalizacji
ANALYZE documents;
ANALYZE document_embeddings;
ANALYZE embedding_models;
ANALYZE search_history;

-- Wyświetlenie podsumowania utworzonych tabel
SELECT 
    schemaname,
    tablename,
    tableowner,
    tablespace,
    hasindexes,
    hasrules,
    hastriggers
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('documents', 'document_embeddings', 'embedding_models', 'search_history')
ORDER BY tablename;