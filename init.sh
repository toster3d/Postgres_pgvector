#!/bin/bash
# docker/init-scripts/init.sh
# Skrypt inicjalizacji systemu semantycznego wyszukiwania dokumentów w Dockerze

set -e

# Kolory dla lepszej czytelności
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Inicjalizacja systemu semantycznego wyszukiwania dokumentów${NC}"

# Funkcja logowania
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Sprawdzenie zmiennych środowiskowych
check_env_vars() {
    log "Sprawdzanie zmiennych środowiskowych..."
    
    required_vars=("DB_HOST" "DB_NAME" "DB_USER" "DB_PASSWORD")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            error "Brak wymaganej zmiennej środowiskowej: $var"
            exit 1
        fi
    done
    
    log "Wszystkie wymagane zmienne środowiskowe są ustawione"
}

# Oczekiwanie na bazę danych
wait_for_database() {
    log "Oczekiwanie na uruchomienie bazy danych..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if pg_isready -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" > /dev/null 2>&1; then
            log "Baza danych jest dostępna"
            return 0
        fi
        
        warning "Próba $attempt/$max_attempts - baza danych jeszcze niedostępna"
        sleep 2
        ((attempt++))
    done
    
    error "Baza danych nie została uruchomiona w czasie $max_attempts prób"
    exit 1
}

# Sprawdzenie rozszerzenia pgvector
check_pgvector() {
    log "Sprawdzanie rozszerzenia pgvector..."
    
    if psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';" | grep -q "vector"; then
        log "Rozszerzenie pgvector jest dostępne"
    else
        error "Rozszerzenie pgvector nie jest dostępne"
        exit 1
    fi
}

# Uruchomienie skryptów SQL
run_sql_scripts() {
    log "Uruchamianie skryptów SQL..."
    
    # Lista skryptów w kolejności wykonania
    scripts=(
        "/docker-entrypoint-initdb.d/init_db.sql"
        "/docker-entrypoint-initdb.d/create_tables.sql"
        "/docker-entrypoint-initdb.d/functions.sql"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            log "Wykonywanie skryptu: $(basename $script)"
            psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "$script" > /dev/null 2>&1 || {
                error "Błąd wykonania skryptu: $script"
                exit 1
            }
        else
            warning "Skrypt nie istnieje: $script"
        fi
    done
    
    log "Wszystkie skrypty SQL zostały wykonane pomyślnie"
}

# Inicjalizacja podstawowych indeksów
initialize_indexes() {
    log "Inicjalizacja indeksów wektorowych..."
    
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "CALL rebuild_vector_indexes(false, 10);" > /dev/null 2>&1 || {
        warning "Nie udało się utworzyć indeksów wektorowych - prawdopodobnie brak danych"
    }
}

# Wstawienie przykładowych danych (opcjonalne)
insert_sample_data() {
    if [ "$LOAD_SAMPLE_DATA" = "true" ]; then
        log "Ładowanie przykładowych danych..."
        
        # Sprawdzenie czy już istnieją jakieś dokumenty
        count=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM documents;" | xargs)
        
        if [ "$count" -eq 0 ]; then
            # Wstawienie przykładowych dokumentów
            psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" << EOF
INSERT INTO documents (title, content, metadata) VALUES
    ('Wprowadzenie do AI', 'Sztuczna inteligencja to dziedzina informatyki zajmująca się tworzeniem systemów zdolnych do wykonywania zadań wymagających inteligencji.', '{"category": "AI", "source": "tutorial"}'),
    ('Podstawy ML', 'Machine Learning jest podzbiorem sztucznej inteligencji, który pozwala komputerom uczyć się bez jawnego programowania.', '{"category": "ML", "source": "tutorial"}'),
    ('Sieci neuronowe', 'Sieci neuronowe to modele matematyczne inspirowane biologicznymi sieciami neuronowymi w mózgu.', '{"category": "neural-networks", "source": "tutorial"}'),
    ('Przetwarzanie języka naturalnego', 'NLP to dziedzina AI zajmująca się interakcją między komputerami a ludzkim językiem naturalnym.', '{"category": "NLP", "source": "tutorial"}'),
    ('Wyszukiwanie semantyczne', 'Wyszukiwanie semantyczne wykorzystuje znaczenie tekstu, a nie tylko dopasowanie słów kluczowych.', '{"category": "search", "source": "tutorial"}');
EOF
            log "Załadowano przykładowe dokumenty"
        else
            log "Dokumenty już istnieją w bazie danych ($count dokumentów)"
        fi
    else
        log "Pominięcie ładowania przykładowych danych (LOAD_SAMPLE_DATA != true)"
    fi
}

# Sprawdzenie poprawności instalacji
verify_installation() {
    log "Weryfikacja instalacji..."
    
    # Sprawdzenie tabel
    tables=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    
    if [ "$tables" -ge 4 ]; then
        log "Wszystkie tabele zostały utworzone ($tables tabel)"
    else
        error "Nie wszystkie tabele zostały utworzone (znaleziono: $tables)"
        exit 1
    fi
    
    # Sprawdzenie funkcji
    functions=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = 'public';" | xargs)
    
    if [ "$functions" -ge 5 ]; then
        log "Wszystkie funkcje zostały utworzone ($functions funkcji)"
    else
        warning "Nie wszystkie funkcje zostały utworzone (znaleziono: $functions)"
    fi
}

# Wyświetlenie podsumowania
show_summary() {
    log "🎉 Inicjalizacja zakończona pomyślnie!"
    
    echo ""
    echo -e "${BLUE}=== PODSUMOWANIE SYSTEMU ===${NC}"
    echo -e "Host bazy danych: ${GREEN}$DB_HOST${NC}"
    echo -e "Nazwa bazy: ${GREEN}$DB_NAME${NC}"
    echo -e "Użytkownik: ${GREEN}$DB_USER${NC}"
    
    # Statystyki
    doc_count=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM documents;" | xargs)
    echo -e "Dokumenty w systemie: ${GREEN}$doc_count${NC}"
    
    echo ""
    echo -e "${YELLOW}Następne kroki:${NC}"
    echo "1. Użyj 'semantic-docs docs add' aby dodać dokumenty"
    echo "2. Użyj 'semantic-docs search semantic \"zapytanie\"' do wyszukiwania"
    echo "3. Sprawdź status: 'semantic-docs status'"
    echo ""
}

# === GŁÓWNA FUNKCJA ===
main() {
    check_env_vars
    wait_for_database
    check_pgvector
    run_sql_scripts
    initialize_indexes
    insert_sample_data
    verify_installation
    show_summary
}

# Uruchomienie skryptu
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi