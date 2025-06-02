-- Script to create tables for semantic document search
-- This should be run after init_db.sql

\c semantic_docs;

-- Documents table to store document metadata and content
CREATE TABLE IF NOT EXISTS doc_search.documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255),
    author VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for text search
CREATE INDEX IF NOT EXISTS idx_documents_content_gin ON doc_search.documents USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_documents_title ON doc_search.documents(title);

-- Embeddings table to store vector representations of documents
CREATE TABLE IF NOT EXISTS doc_search.embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES doc_search.documents(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    embedding_vector vector(384) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON doc_search.embeddings USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);

-- Add a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION doc_search.update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add a trigger to automatically update the updated_at column
CREATE TRIGGER update_documents_modtime
BEFORE UPDATE ON doc_search.documents
FOR EACH ROW
EXECUTE FUNCTION doc_search.update_modified_column(); 