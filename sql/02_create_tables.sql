\echo 'Tworzenie tabel dla dokumentów naukowych...'

-- Tabela głównych dokumentów (artykułów naukowych)
CREATE TABLE semantic.documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    full_text TEXT,                 
    pub_year INTEGER,
    doi TEXT,
    pmid TEXT,
    issn TEXT,
    volume TEXT,
    first_page TEXT,
    last_page TEXT,
    document_type semantic.document_type DEFAULT 'article',
    language TEXT DEFAULT 'en',
    word_count INTEGER,
    asjc_codes INTEGER[],
    subject_areas TEXT[],
    keywords TEXT[],
    author_highlights TEXT[],
    status semantic.document_status DEFAULT 'pending',
    processing_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    title_tsvector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, ''))
    ) STORED,
    abstract_tsvector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(abstract, ''))
    ) STORED,
    full_text_tsvector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(full_text, ''))
    ) STORED
);

-- Tabela autorów
-- UWAGA: Ta tabela oraz tabela `document_authors` nie są aktywnie używane w obecnych skryptach,
-- ale stanowią fundament pod przyszły rozwój aplikacji. Umożliwią wdrożenie funkcji
-- takich jak wyszukiwanie po autorze, analiza współpracy czy znajdowanie ekspertów.
-- Poprawiają również integralność danych przez ich normalizację.
CREATE TABLE semantic.authors (
    author_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name TEXT,
    last_name TEXT NOT NULL,
    email TEXT,
    affiliation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela powiązań dokument-autor
CREATE TABLE semantic.document_authors (
    document_id UUID REFERENCES semantic.documents(document_id) ON DELETE CASCADE,
    author_id UUID REFERENCES semantic.authors(author_id) ON DELETE CASCADE,
    author_order INTEGER,
    PRIMARY KEY (document_id, author_id)
);

-- Tabela embeddingów (chunki dokumentów z wektorami)
CREATE TABLE semantic.document_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES semantic.documents(document_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type TEXT DEFAULT 'content',
    embedding VECTOR(384),
    embedding_model semantic.embedding_model DEFAULT 'all-MiniLM-L6-v2',
    start_position INTEGER,
    end_position INTEGER,
    word_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index, chunk_type)
);

-- Tabela bibliografii (referencji)
-- UWAGA: Ta tabela nie jest obecnie wypełniana danymi. Została zaprojektowana z myślą
-- o przyszłych funkcjonalnościach, takich jak analiza cytowań, odkrywanie kluczowych
-- prac w danej dziedzinie czy budowanie grafów powiązań między dokumentami.
CREATE TABLE semantic.bibliography (
    bib_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES semantic.documents(document_id) ON DELETE CASCADE,
    bib_ref_id TEXT NOT NULL,
    title TEXT,
    authors_text TEXT,
    journal TEXT,
    pub_year INTEGER,
    doi TEXT,
    pmid TEXT,    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(document_id, bib_ref_id)
);

-- Tabela historii wyszukiwań (do analiz i optymalizacji)
-- UWAGA: Ta tabela nie jest obecnie używana. Jej celem jest przyszłe gromadzenie
-- danych o zapytaniach użytkowników w celu analizy trendów, optymalizacji
-- wydajności najczęstszych zapytań oraz potencjalnego "uczenia się" przez system.
CREATE TABLE semantic.search_history (
    search_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    search_type TEXT NOT NULL,
    limit_results INTEGER,
    similarity_threshold REAL,
    embedding_model semantic.embedding_model,
    results_count INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indeksy dla wydajności

CREATE INDEX idx_documents_doc_id ON semantic.documents(doc_id);
CREATE INDEX idx_documents_pub_year ON semantic.documents(pub_year);
CREATE INDEX idx_documents_doi ON semantic.documents(doi);
CREATE INDEX idx_documents_status ON semantic.documents(status);
CREATE INDEX idx_documents_asjc ON semantic.documents USING GIN(asjc_codes);
CREATE INDEX idx_documents_keywords ON semantic.documents USING GIN(keywords);
CREATE INDEX idx_documents_title_fts ON semantic.documents USING GIN(title_tsvector);
CREATE INDEX idx_documents_abstract_fts ON semantic.documents USING GIN(abstract_tsvector);
CREATE INDEX idx_documents_fulltext_fts ON semantic.documents USING GIN(full_text_tsvector);

-- Indeksy na embeddings
CREATE INDEX idx_embeddings_document ON semantic.document_embeddings(document_id);
CREATE INDEX idx_embeddings_chunk_type ON semantic.document_embeddings(chunk_type);

-- Indeks wektorowy HNSW dla embeddings (najlepszy dla większości przypadków)
CREATE INDEX idx_embeddings_vector_hnsw ON semantic.document_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Indeksy na autorów
CREATE INDEX idx_authors_name ON semantic.authors(last_name, first_name);
CREATE INDEX idx_document_authors_doc ON semantic.document_authors(document_id);

-- Indeksy na bibliografię
CREATE INDEX idx_bibliography_document ON semantic.bibliography(document_id);
CREATE INDEX idx_bibliography_doi ON semantic.bibliography(doi);

-- Indeksy na historię wyszukiwań
CREATE INDEX idx_search_history_created ON semantic.search_history(created_at);
CREATE INDEX idx_search_history_type ON semantic.search_history(search_type);

-- Trigger do automatycznej aktualizacji timestamp
CREATE OR REPLACE FUNCTION semantic.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON semantic.documents
    FOR EACH ROW EXECUTE FUNCTION semantic.update_updated_at_column();

-- Potwierdzenie utworzenia tabel
\echo 'Tabele zostały pomyślnie utworzone!'
\echo '- documents: główne dokumenty naukowe'
\echo '- authors: autorzy dokumentów'
\echo '- document_authors: powiązania dokument-autor'
\echo '- document_embeddings: chunki z embeddingami'
\echo '- bibliography: referencje/bibliografia'
\echo '- search_history: historia wyszukiwań'
\echo ''
\echo 'Indeksy zostały utworzone dla optymalnej wydajności!'
\echo '- Indeksy GIN dla full-text search'
\echo '- Indeks HNSW dla wyszukiwania wektorowego'
\echo '- Indeksy B-tree dla standardowych zapytań'