# Poprawiony Dockerfile dla Semantycznego Wyszukiwania Dokumentów
# Ten Dockerfile naprawia problemy z psycopg_pool i psycopg3

# Multi-stage build dla optymalizacji rozmiaru obrazu
FROM python:3.11-slim as builder

# Ustawienie zmiennych środowiskowych
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalacja zależności systemowych potrzebnych do budowania
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Utworzenie katalogu roboczego
WORKDIR /app

# Kopiowanie plików konfiguracyjnych
COPY pyproject.toml README.md ./

# Upgrade pip i instalacja build tools
RUN pip install --upgrade pip setuptools wheel

# Instalacja zależności Python (z psycopg3 i pooling)
RUN pip install -e .

# === STAGE 2: Runtime ===
FROM python:3.11-slim as runtime

# Ustawienie zmiennych środowiskowych
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

# Instalacja tylko runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Utworzenie użytkownika bez uprawnień root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie zainstalowanych pakietów z builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Kopiowanie kodu aplikacji
COPY semantic_doc_search/ ./semantic_doc_search/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Instalacja w trybie edytowalnym (dla development)
RUN pip install -e .

# Utworzenie potrzebnych katalogów
RUN mkdir -p /app/data /app/exports /app/logs /app/temp

# Zmiana właściciela plików na appuser
RUN chown -R appuser:appuser /app

# Przełączenie na użytkownika appuser
USER appuser

# Test działania instalacji - z poprawką dla psycopg3
RUN python -c "import semantic_doc_search; print('✅ Package installed successfully')" && \
    python -c "import psycopg; print('✅ psycopg3 available')" && \
    python -c "import psycopg_pool; print('✅ psycopg_pool available')" && \
    python -c "import sentence_transformers; print('✅ sentence_transformers available')"

# Sprawdzenie dostępności CLI
RUN semantic-docs --help > /dev/null && echo "✅ CLI working"

# Eksponowanie portu (opcjonalne, dla przyszłych rozszerzeń)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import semantic_doc_search; print('OK')" || exit 1

# Entry point
ENTRYPOINT ["semantic-docs"]
CMD ["--help"]