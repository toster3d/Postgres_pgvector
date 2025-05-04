-- Usuń bazę danych, jeśli istnieje (uwaga - użyj tylko w środowisku developerskim)
DROP DATABASE IF EXISTS semantic_search;

-- Utwórz nową bazę danych
CREATE DATABASE semantic_search;

-- Połącz z nową bazą danych
\c semantic_search

-- Zainstaluj rozszerzenie pgvector, jeśli nie jest zainstalowane
CREATE EXTENSION IF NOT EXISTS vector;

-- Utwórz schemat dla naszej aplikacji
CREATE SCHEMA IF NOT EXISTS semantic_docs;

-- Ustaw domyślną ścieżkę wyszukiwania
SET search_path TO semantic_docs, public;

-- Komunikat
SELECT 'Baza danych semantic_search została utworzona pomyślnie z rozszerzeniem pgvector.' as info;