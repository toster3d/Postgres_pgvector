# docker/Dockerfile
# Dockerfile dla aplikacji semantycznego wyszukiwania dokumentów
# Kompatybilny z Python 3.13.3 i najnowszymi wersjami bibliotek

# === MULTI-STAGE BUILD ===

# Etap 1: Baza dla kodu Python
FROM python:3.13.3-slim-bookworm as python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    PYTHONPATH="/app:$PYTHONPATH" \
    UV_PIP_DEFAULT_INDEX_CLIENT=pypi-json

# Etap 2: Zależności budowania
FROM python-base as builder

WORKDIR /app

# Instalacja narzędzi budowania i uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://astral.sh/uv/install.sh | sh

# Kopiowanie plików projektowych
COPY pyproject.toml README.md ./

# Instalacja zależności
RUN . ~/.cargo/env && uv pip install --system -e .

# Etap 3: Obraz aplikacji
FROM python-base as app

WORKDIR /app

# Instalacja wymaganych pakietów
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie zależności z etapu budowania
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Kopiowanie kodu aplikacji
COPY . /app/

# Tworzenie użytkownika bez uprawnień roota
RUN groupadd -r semantic && useradd -r -g semantic semantic \
    && chown -R semantic:semantic /app

# Przełączenie na użytkownika bez uprawnień roota
USER semantic

# Eksponowanie portu (jeśli będzie używany serwer)
EXPOSE 8000

# Ustawienie zmiennych środowiskowych dla produkcji
ENV PYTHONPATH="/app" \
    SEMANTIC_DOCS_ENV="production"

# Utworzenie wolumenu dla danych
VOLUME ["/app/data"]

# Uruchomienie aplikacji
ENTRYPOINT ["semantic-docs"]
CMD ["help"]