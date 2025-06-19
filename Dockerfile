FROM python:3.13-slim

# Ustawienie zmiennych środowiskowych
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.local/bin:$PATH" \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tworzenie użytkownika non-root
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Zapewnienie poprawnych uprawnień do katalogu domowego
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

# Instalacja zależności systemowych w jednej warstwie
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Zależności dla kompilacji Python packages
    build-essential \
    gcc \
    g++ \
    # Narzędzia PostgreSQL
    postgresql-client \
    libpq-dev \
    # Narzędzia systemowe (minimalne)
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie plików konfiguracyjnych (cache layer optimization)
COPY pyproject.toml ./

# Instalacja zależności Python (jako root dla system packages)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Tworzenie potrzebnych katalogów i ustawienie uprawnień
RUN mkdir -p /app/data /app/scripts && \
    chown -R appuser:appuser /app

# Przełączenie na użytkownika non-root
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Domyślna komenda
CMD ["/bin/bash"]