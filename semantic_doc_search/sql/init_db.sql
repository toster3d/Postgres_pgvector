-- Script to initialize PostgreSQL database with pgvector extension
-- This should be run as a PostgreSQL superuser

-- Create pgvector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a schema for our application
CREATE SCHEMA IF NOT EXISTS doc_search;

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA doc_search TO postgres; 