-- semantic_doc_search/sql/functions.sql
-- Zaawansowane funkcje PostgreSQL dla systemu semantycznego wyszukiwania dokumentów
-- Kompatybilne z PostgreSQL 17+ i pgvector 0.8.0

-- === FUNKCJE POMOCNICZE ===

-- Funkcja do dynamicznego tworzenia indeksów HNSW dla wektorów
CREATE OR REPLACE FUNCTION create_hnsw_index()
RETURNS VOID AS $$
BEGIN
    -- Usunięcie istniejącego indeksu jeśli istnieje
    DROP INDEX IF EXISTS idx_embeddings_vector_hnsw;
    
    -- Utworzenie nowego indeksu HNSW z optymalnymi parametrami
    EXECUTE 'CREATE INDEX idx_embeddings_vector_hnsw ON document_embeddings 
             USING hnsw (embedding vector_cosine_ops) 
             WITH (m = 16, ef_construction = 64)';
    
    RAISE NOTICE 'Utworzono indeks HNSW dla embeddingów';
END;
$$ LANGUAGE plpgsql;

-- Funkcja do dynamicznego tworzenia indeksów IVFFlat dla wektorów
CREATE OR REPLACE FUNCTION create_ivfflat_index(lists_count INTEGER DEFAULT 100)
RETURNS VOID AS $$
BEGIN
    -- Usunięcie istniejącego indeksu jeśli istnieje
    DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat;
    
    -- Obliczenie optymalnej liczby list dla IVFFlat
    DECLARE
        optimal_lists INTEGER;
        docs_count INTEGER;
    BEGIN
        SELECT COUNT(*) INTO docs_count FROM document_embeddings;
        
        -- Heurystyka: sqrt(n) list dla n dokumentów, min 10, max 1000
        optimal_lists := GREATEST(10, LEAST(1000, CEIL(SQRT(docs_count))));
        
        IF lists_count IS NULL THEN
            lists_count := optimal_lists;
        END IF;
        
        RAISE NOTICE 'Używam % list dla indeksu IVFFlat (zalecane: %)', lists_count, optimal_lists;
    END;
    
    -- Utworzenie nowego indeksu IVFFlat z optymalnymi parametrami
    EXECUTE format('CREATE INDEX idx_embeddings_vector_ivfflat ON document_embeddings 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = %s)', lists_count);
    
    RAISE NOTICE 'Utworzono indeks IVFFlat dla embeddingów z % listami', lists_count;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do aktualizacji wymiaru wektorowego dla modelu
CREATE OR REPLACE FUNCTION update_embedding_dimension(model_name VARCHAR, new_dimension INTEGER)
RETURNS VOID AS $$
DECLARE
    old_dimension INTEGER;
BEGIN
    -- Pobranie obecnego wymiaru
    SELECT dimension INTO old_dimension 
    FROM embedding_models 
    WHERE name = model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % nie istnieje', model_name;
    END IF;
    
    -- Aktualizacja wymiaru w tabeli modeli
    UPDATE embedding_models 
    SET dimension = new_dimension 
    WHERE name = model_name;
    
    RAISE NOTICE 'Zaktualizowano wymiar dla modelu % z % na %', 
                 model_name, old_dimension, new_dimension;
    
    -- UWAGA: Istniejące embeddingi muszą być regenerowane!
END;
$$ LANGUAGE plpgsql;

-- === FUNKCJE WYSZUKIWANIA ===

-- Funkcja do pełnotekstowego wyszukiwania
CREATE OR REPLACE FUNCTION text_search(
    search_query TEXT,
    limit_results INTEGER DEFAULT 10,
    min_score FLOAT DEFAULT 0.1
) 
RETURNS TABLE (
    id INTEGER,
    title TEXT,
    content TEXT,
    metadata JSONB,
    score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        d.metadata,
        ts_rank(d.search_vector, to_tsquery('polish', search_query)) AS score,
        d.created_at
    FROM 
        documents d
    WHERE 
        d.search_vector @@ to_tsquery('polish', search_query)
        AND ts_rank(d.search_vector, to_tsquery('polish', search_query)) >= min_score
    ORDER BY 
        score DESC
    LIMIT 
        limit_results;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do semantycznego wyszukiwania
CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding vector,
    model_name TEXT,
    limit_results INTEGER DEFAULT 10,
    min_score FLOAT DEFAULT 0.5
) 
RETURNS TABLE (
    id INTEGER,
    title TEXT,
    content TEXT,
    metadata JSONB,
    score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    model_id INTEGER;
BEGIN
    -- Znajdź ID modelu
    SELECT id INTO model_id FROM embedding_models WHERE name = model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % nie istnieje', model_name;
    END IF;
    
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        d.metadata,
        1 - (e.embedding <=> query_embedding) AS score, -- Cosine similarity (1 - distance)
        d.created_at
    FROM 
        document_embeddings e
    JOIN 
        documents d ON e.document_id = d.id
    WHERE 
        e.model_id = model_id
        AND (1 - (e.embedding <=> query_embedding)) >= min_score
    ORDER BY 
        score DESC
    LIMIT 
        limit_results;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do hybrydowego wyszukiwania
CREATE OR REPLACE FUNCTION hybrid_search(
    search_query TEXT,
    query_embedding vector,
    model_name TEXT,
    semantic_weight FLOAT DEFAULT 0.7,
    limit_results INTEGER DEFAULT 10,
    min_score FLOAT DEFAULT 0.3
) 
RETURNS TABLE (
    id INTEGER,
    title TEXT,
    content TEXT,
    metadata JSONB,
    score FLOAT,
    semantic_score FLOAT,
    text_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    model_id INTEGER;
BEGIN
    -- Znajdź ID modelu
    SELECT id INTO model_id FROM embedding_models WHERE name = model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % nie istnieje', model_name;
    END IF;
    
    -- Walidacja wagi
    IF semantic_weight < 0 OR semantic_weight > 1 THEN
        RAISE EXCEPTION 'Waga semantyczna musi być w zakresie 0-1';
    END IF;
    
    RETURN QUERY
    WITH semantic_results AS (
        SELECT 
            d.id,
            1 - (e.embedding <=> query_embedding) AS semantic_score
        FROM 
            document_embeddings e
        JOIN 
            documents d ON e.document_id = d.id
        WHERE 
            e.model_id = model_id
    ),
    text_results AS (
        SELECT 
            d.id,
            ts_rank(d.search_vector, to_tsquery('polish', search_query)) AS text_score
        FROM 
            documents d
        WHERE 
            d.search_vector @@ to_tsquery('polish', search_query)
    ),
    combined_results AS (
        SELECT 
            COALESCE(s.id, t.id) AS id,
            COALESCE(s.semantic_score, 0) AS semantic_score,
            COALESCE(t.text_score, 0) AS text_score,
            (COALESCE(s.semantic_score, 0) * semantic_weight + 
             COALESCE(t.text_score, 0) * (1 - semantic_weight)) AS combined_score
        FROM 
            semantic_results s
        FULL OUTER JOIN 
            text_results t ON s.id = t.id
        WHERE
            (COALESCE(s.semantic_score, 0) * semantic_weight + 
             COALESCE(t.text_score, 0) * (1 - semantic_weight)) >= min_score
    )
    SELECT 
        d.id,
        d.title,
        d.content,
        d.metadata,
        c.combined_score AS score,
        c.semantic_score,
        c.text_score,
        d.created_at
    FROM 
        combined_results c
    JOIN 
        documents d ON c.id = d.id
    ORDER BY 
        c.combined_score DESC
    LIMIT 
        limit_results;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do rekomendacji podobnych dokumentów
CREATE OR REPLACE FUNCTION recommend_similar(
    source_document_id INTEGER,
    model_name TEXT,
    limit_results INTEGER DEFAULT 10,
    min_score FLOAT DEFAULT 0.5
) 
RETURNS TABLE (
    id INTEGER,
    title TEXT,
    content TEXT,
    metadata JSONB,
    score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    model_id INTEGER;
    source_embedding vector;
BEGIN
    -- Znajdź ID modelu
    SELECT id INTO model_id FROM embedding_models WHERE name = model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % nie istnieje', model_name;
    END IF;
    
    -- Pobierz embedding dokumentu źródłowego
    SELECT embedding INTO source_embedding
    FROM document_embeddings
    WHERE document_id = source_document_id AND model_id = model_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Brak embeddingu dla dokumentu ID % w modelu %', 
                        source_document_id, model_name;
    END IF;
    
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        d.metadata,
        1 - (e.embedding <=> source_embedding) AS score,
        d.created_at
    FROM 
        document_embeddings e
    JOIN 
        documents d ON e.document_id = d.id
    WHERE 
        e.model_id = model_id
        AND e.document_id != source_document_id
        AND (1 - (e.embedding <=> source_embedding)) >= min_score
    ORDER BY 
        score DESC
    LIMIT 
        limit_results;
END;
$$ LANGUAGE plpgsql;

-- === PROCEDURY ADMINISTRACYJNE ===

-- Procedura do regeneracji indeksów wektorowych
CREATE OR REPLACE PROCEDURE rebuild_vector_indexes(
    use_hnsw BOOLEAN DEFAULT false,
    lists_count INTEGER DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF use_hnsw THEN
        PERFORM create_hnsw_index();
    ELSE
        PERFORM create_ivfflat_index(lists_count);
    END IF;
    
    ANALYZE document_embeddings;
    
    RAISE NOTICE 'Indeksy wektorowe zostały zregenerowane';
END;
$$;

-- Procedura do regeneracji wszystkich indeksów (full-text i wektorowych)
CREATE OR REPLACE PROCEDURE rebuild_all_indexes()
LANGUAGE plpgsql
AS $$
BEGIN
    -- Indeksy pełnotekstowe
    REINDEX INDEX idx_documents_title;
    REINDEX INDEX idx_documents_content_gin;
    REINDEX INDEX idx_documents_title_trigram;
    
    -- Indeksy wektorowe
    CALL rebuild_vector_indexes();
    
    -- Analiza dla optymalizatora zapytań
    ANALYZE documents;
    ANALYZE document_embeddings;
    ANALYZE embedding_models;
    
    RAISE NOTICE 'Wszystkie indeksy zostały zregenerowane';
END;
$$;

-- === NOTATKI ===
/*
Funkcje powyżej oferują:

1. Dynamiczne zarządzanie indeksami wektorowymi (create_hnsw_index, create_ivfflat_index)
2. Aktualizację wymiarów dla modeli embeddingów (update_embedding_dimension)
3. Trzy podstawowe metody wyszukiwania:
   - text_search: wyszukiwanie pełnotekstowe
   - semantic_search: wyszukiwanie semantyczne (wektorowe)
   - hybrid_search: wyszukiwanie hybrydowe łączące oba podejścia
4. Funkcję do rekomendacji podobnych dokumentów (recommend_similar)
5. Procedury administracyjne do regeneracji indeksów

WAŻNE:
- Aby te funkcje działały, tabele muszą zostać najpierw utworzone przez create_tables.sql
- Tworzone indeksy są optymalizowane pod kątem wydajności, ale mogą wymagać dostrojenia
  dla konkretnych zbiorów danych
- Funkcje powinny być kompatybilne z PostgreSQL 17+ i pgvector 0.8.0
*/