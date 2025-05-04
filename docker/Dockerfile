FROM python:3.10-slim

WORKDIR /app

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Skopiuj pliki projektu
COPY requirements.txt .

# Zainstaluj zależności Pythona
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę kodu
COPY . .

# Zmienne środowiskowe
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Uruchom aplikację
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 