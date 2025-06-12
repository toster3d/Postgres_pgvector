-- Tworzenie tabel dla systemu semantycznego wyszukiwania dokumentów
-- Uruchom po init_db.sql: psql -U postgres -d semantic_docs -f create_tables.sql

-- Tabela dokumentów
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indeksy full-text search
    content_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED
);

-- Tabela embeddings dla semantycznego wyszukiwania
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER DEFAULT 0,
    chunk_text TEXT NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'sentence-transformers',
    embedding_dimension INTEGER NOT NULL DEFAULT 384,
    embedding vector, -- Wymiar będzie ustawiony dynamicznie
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(document_id, chunk_index, embedding_model)
);

-- Tabela konfiguracji modeli embeddings
CREATE TABLE IF NOT EXISTS embedding_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'sentence-transformers', 'openai', 'sklearn'
    embedding_dimension INTEGER NOT NULL,
    model_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela historii wyszukiwań (opcjonalna)
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    search_type VARCHAR(20) NOT NULL, -- 'text', 'semantic', 'hybrid'
    model_used VARCHAR(100),
    results_count INTEGER,
    search_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indeksy dla performance
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents USING GIN (to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_documents_content_vector ON documents USING GIN (content_vector);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);

-- Indeksy dla embeddings - będą utworzone dynamicznie po dodaniu danych
-- CREATE INDEX idx_embeddings_vector_l2 ON document_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
-- CREATE INDEX idx_embeddings_vector_cosine ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX idx_embeddings_vector_ip ON document_embeddings USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON document_embeddings (document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON document_embeddings (embedding_model);

-- Trigger do aktualizacji updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Wstawianie domyślnych modeli embeddings
INSERT INTO embedding_models (model_name, model_type, embedding_dimension, model_config) VALUES
    ('all-MiniLM-L6-v2', 'sentence-transformers', 384, '{"normalize_embeddings": true}'),
    ('all-mpnet-base-v2', 'sentence-transformers', 768, '{"normalize_embeddings": true}'),
    ('text-embedding-ada-002', 'openai', 1536, '{"model": "text-embedding-ada-002"}'),
    ('text-embedding-3-small', 'openai', 1536, '{"model": "text-embedding-3-small"}'),
    ('text-embedding-3-large', 'openai', 3072, '{"model": "text-embedding-3-large"}'),
    ('tfidf-vectorizer', 'sklearn', 1000, '{"max_features": 1000, "ngram_range": [1, 2]}')
ON CONFLICT (model_name) DO NOTHING;

-- Funkcja do aktualizacji wymiaru embeddings
CREATE OR REPLACE FUNCTION update_embedding_dimension(table_name TEXT, new_dimension INTEGER)
RETURNS VOID AS $$
BEGIN
    EXECUTE format('ALTER TABLE %I ALTER COLUMN embedding TYPE vector(%s)', table_name, new_dimension);
END;
$$ LANGUAGE plpgsql;

-- Funkcja do tworzenia indeksów wektorowych
CREATE OR REPLACE FUNCTION create_vector_indexes(
    table_name TEXT DEFAULT 'document_embeddings',
    lists_param INTEGER DEFAULT 100
)
RETURNS VOID AS $$
BEGIN
    -- IVFFlat indeksy dla różnych metryk odległości
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (embedding vector_l2_ops) WITH (lists = %s)', 
        table_name || '_vector_l2_idx', table_name, lists_param);
    
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (embedding vector_cosine_ops) WITH (lists = %s)', 
        table_name || '_vector_cosine_idx', table_name, lists_param);
    
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (embedding vector_ip_ops) WITH (lists = %s)', 
        table_name || '_vector_ip_idx', table_name, lists_param);
    
    RAISE NOTICE 'Vector indexes created for table %', table_name;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do hybrydowego wyszukiwania
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector,
    semantic_weight FLOAT DEFAULT 0.5,
    limit_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    doc_id INTEGER,
    title VARCHAR(500),
    content TEXT,
    source VARCHAR(255),
    semantic_score FLOAT,
    text_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT 
            d.id,
            d.title,
            d.content,
            d.source,
            1 - (de.embedding <=> query_embedding) AS semantic_score
        FROM documents d
        JOIN document_embeddings de ON d.id = de.document_id
        WHERE de.embedding IS NOT NULL
        ORDER BY de.embedding <=> query_embedding
        LIMIT limit_results * 2
    ),
    text_results AS (
        SELECT 
            d.id,
            d.title,
            d.content,
            d.source,
            ts_rank_cd(d.content_vector, plainto_tsquery('english', query_text)) AS text_score
        FROM documents d
        WHERE d.content_vector @@ plainto_tsquery('english', query_text)
        ORDER BY ts_rank_cd(d.content_vector, plainto_tsquery('english', query_text)) DESC
        LIMIT limit_results * 2
    )
    SELECT 
        COALESCE(sr.id, tr.id) AS doc_id,
        COALESCE(sr.title, tr.title) AS title,
        COALESCE(sr.content, tr.content) AS content,
        COALESCE(sr.source, tr.source) AS source,
        COALESCE(sr.semantic_score, 0.0) AS semantic_score,
        COALESCE(tr.text_score, 0.0) AS text_score,
        (COALESCE(sr.semantic_score, 0.0) * semantic_weight + 
         COALESCE(tr.text_score, 0.0) * (1 - semantic_weight)) AS combined_score
    FROM semantic_results sr
    FULL OUTER JOIN text_results tr ON sr.id = tr.id
    ORDER BY combined_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Informacja o utworzonych tabelach
SELECT 
    schemaname,
    tablename,
    hasindexes,
    hastriggers
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('documents', 'document_embeddings', 'embedding_models', 'search_history')
ORDER BY tablename;