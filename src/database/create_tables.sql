-- Połącz z bazą danych
\c semantic_search

-- Ustaw domyślną ścieżkę wyszukiwania
SET search_path TO semantic_docs, public;

-- Tabela kategorii
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela dokumentów
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category_id INTEGER REFERENCES categories(id),
    metadata JSONB DEFAULT '{}',
    document_vector vector(1536), -- Wymiar wektora zależy od modelu embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indeks full-text search dla dokumentów
CREATE INDEX IF NOT EXISTS idx_documents_content_gin ON documents USING gin(to_tsvector('polish', content));
CREATE INDEX IF NOT EXISTS idx_documents_title_gin ON documents USING gin(to_tsvector('polish', title));

-- Indeks dla wektorów embeddings (dla szybkiego wyszukiwania)
CREATE INDEX IF NOT EXISTS idx_documents_vector ON documents USING ivfflat (document_vector vector_cosine_ops) WITH (lists = 100);

-- Funkcja do aktualizacji pola updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggery dla aktualizacji pola updated_at
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_categories_updated_at
    BEFORE UPDATE ON categories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Dodaj przykładowe kategorie
INSERT INTO categories (name, description)
VALUES 
    ('Artykuły naukowe', 'Publikacje akademickie i badawcze'),
    ('Dokumentacja techniczna', 'Dokumentacja techniczna produktów i usług'),
    ('Raporty biznesowe', 'Raporty i analizy biznesowe'),
    ('Artykuły prasowe', 'Publikacje z mediów'),
    ('Inne', 'Inne dokumenty')
ON CONFLICT (name) DO NOTHING;

-- Komunikat
SELECT 'Tabele i indeksy zostały utworzone pomyślnie.' as info;