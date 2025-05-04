"""
Skrypt do generowania przykładowych danych.
"""
import os
import sys
import logging
import random
from typing import List, Dict
from contextlib import contextmanager

# Dodaj katalog nadrzędny do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from src.database.db import get_db, init_db
from src.database.models import Category, Document, DocumentEmbedding
from src.embeddings.embeddings import get_embedding_generator

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Przykładowe treści dokumentów dla każdej kategorii
SAMPLE_DOCUMENTS = {
    "Ogólne": [
        {
            "title": "Wprowadzenie do baz danych",
            "content": """
            Baza danych to uporządkowany zbiór danych przechowywanych i przetwarzanych w komputerze. 
            Bazy danych są projektowane tak, aby efektywnie przechowywać, pobierać i zarządzać danymi.
            Systemy zarządzania bazami danych (DBMS) umożliwiają użytkownikom interakcję z bazami danych.
            Najpopularniejsze typy baz danych to relacyjne, dokumentowe, grafowe i klucz-wartość.
            """
        },
        {
            "title": "Historia Internetu",
            "content": """
            Internet rozpoczął się jako ARPANET w latach 60. XX wieku, projekt badawczy sponsorowany przez amerykańską agencję DARPA.
            World Wide Web został wynaleziony przez Tima Bernersa-Lee w 1989 roku w CERN.
            W latach 90. nastąpił gwałtowny rozwój komercyjny Internetu.
            Dzisiaj Internet łączy miliardy urządzeń i ludzi na całym świecie.
            """
        }
    ],
    "Technologia": [
        {
            "title": "Wprowadzenie do uczenia maszynowego",
            "content": """
            Uczenie maszynowe to dziedzina sztucznej inteligencji, która umożliwia systemom uczenie się i doskonalenie na podstawie doświadczenia.
            Algorytmy uczenia maszynowego budują model na podstawie danych treningowych, aby podejmować decyzje lub przewidywania bez wyraźnego programowania.
            Główne typy uczenia maszynowego to uczenie nadzorowane, nienadzorowane i uczenie przez wzmacnianie.
            Popularne zastosowania obejmują rozpoznawanie obrazów, przetwarzanie języka naturalnego i rekomendacje.
            """
        },
        {
            "title": "Architektura mikroserwisów",
            "content": """
            Architektura mikroserwisów to podejście do tworzenia aplikacji jako kolekcji luźno powiązanych usług.
            Każdy mikroserwis koncentruje się na konkretnej funkcjonalności biznesowej i może być rozwijany, wdrażany i skalowany niezależnie.
            Komunikacja między mikroserwisami odbywa się zazwyczaj przez API, często przy użyciu protokołów HTTP/REST lub gRPC.
            Korzyści obejmują elastyczność, skalowalność i odporność na awarie, ale kosztem zwiększonej złożoności operacyjnej.
            """
        },
        {
            "title": "Wprowadzenie do konteneryzacji i Docker",
            "content": """
            Konteneryzacja to lekka forma wirtualizacji, która pakuje aplikację i jej zależności w izolowane środowisko.
            Docker to wiodąca platforma konteneryzacji, która upraszcza tworzenie, wdrażanie i uruchamianie aplikacji w kontenerach.
            Obrazy Docker definiują środowisko aplikacji, a kontenery są uruchomionymi instancjami tych obrazów.
            Kontenery zapewniają spójność środowiska, efektywne wykorzystanie zasobów i łatwiejsze zarządzanie infrastrukturą.
            """
        }
    ],
    "Finanse": [
        {
            "title": "Podstawy inwestowania",
            "content": """
            Inwestowanie to alokowanie zasobów z oczekiwaniem na przyszłe zyski.
            Główne klasy aktywów to akcje, obligacje, nieruchomości i towary.
            Dywersyfikacja to strategia redukowania ryzyka poprzez rozłożenie inwestycji na różne aktywa.
            Compound interest to potężna koncepcja, w której zyski są ponownie inwestowane, generując dodatkowe zwroty w czasie.
            """
        },
        {
            "title": "Analiza finansowa firm",
            "content": """
            Analiza finansowa to ocena rentowności, wypłacalności i stabilności firmy.
            Kluczowe sprawozdania finansowe to bilans, rachunek zysków i strat oraz rachunek przepływów pieniężnych.
            Wskaźniki finansowe, takie jak wskaźnik P/E, ROI i wskaźnik zadłużenia, dostarczają wglądu w kondycję finansową.
            Analiza fundamentalna bada czynniki ekonomiczne i finansowe wpływające na wartość firmy.
            """
        }
    ],
    "Nauka": [
        {
            "title": "Wprowadzenie do fizyki kwantowej",
            "content": """
            Fizyka kwantowa to gałąź fizyki, która zajmuje się zachowaniem materii i energii na poziomie atomowym i subatomowym.
            Kluczowe zasady obejmują dualizm korpuskularno-falowy, zasadę nieoznaczoności Heisenberga i superpozycję kwantową.
            Mechanika kwantowa zrewolucjonizowała nasze zrozumienie natury i umożliwiła rozwój technologii takich jak lasery i tranzystory.
            Interpretacja kopenhaaska, teoria wielu światów i dekoherencja kwantowa to różne interpretacje mechaniki kwantowej.
            """
        },
        {
            "title": "Zmiana klimatu i jej wpływ",
            "content": """
            Zmiana klimatu odnosi się do długoterminowych zmian w temperaturach i wzorcach pogodowych na Ziemi.
            Główne przyczyny to emisje gazów cieplarnianych, wylesianie i industrializacja.
            Skutki obejmują rosnące temperatury, podnoszenie się poziomu morza, ekstremalne zjawiska pogodowe i zakłócenia ekosystemów.
            Łagodzenie skutków i adaptacja to dwa główne podejścia do radzenia sobie ze zmianą klimatu.
            """
        },
        {
            "title": "Rozwój genetyki i jej zastosowania",
            "content": """
            Genetyka to badanie genów, dziedziczenia cech i zmienności organizmów.
            DNA, nośnik informacji genetycznej, został odkryty w latach 50. XX wieku przez Watsona i Cricka.
            Projekt Ludzkiego Genomu zakończony w 2003 roku zmapował cały ludzki genom.
            Nowoczesne zastosowania genetyki obejmują medycynę spersonalizowaną, modyfikacje genetyczne i badania ewolucyjne.
            """
        }
    ]
}


def create_sample_categories(db: Session) -> Dict[str, int]:
    """Tworzy przykładowe kategorie w bazie danych."""
    # Sprawdź, czy kategorie już istnieją
    existing_categories = {cat.name: cat.id for cat in db.query(Category).all()}
    if all(cat in existing_categories for cat in SAMPLE_DOCUMENTS.keys()):
        logger.info("Kategorie już istnieją w bazie danych.")
        return existing_categories

    # Dodaj brakujące kategorie
    category_ids = {}
    for category_name in SAMPLE_DOCUMENTS.keys():
        if category_name in existing_categories:
            category_ids[category_name] = existing_categories[category_name]
            continue

        # Utwórz nową kategorię
        category = Category(
            name=category_name,
            description=f"Kategoria {category_name}"
        )
        db.add(category)
        db.commit()
        db.refresh(category)
        
        category_ids[category_name] = category.id
        logger.info(f"Utworzono kategorię: {category_name} (ID: {category.id})")
    
    return category_ids


def create_sample_documents(db: Session, category_ids: Dict[str, int]) -> List[int]:
    """Tworzy przykładowe dokumenty w bazie danych."""
    # Inicjalizuj generator embeddings
    embedding_generator = get_embedding_generator()
    
    # Sprawdź, czy dokumenty już istnieją
    existing_count = db.query(Document).count()
    if existing_count > 0:
        logger.info(f"W bazie danych już istnieje {existing_count} dokumentów.")
        return [doc.id for doc in db.query(Document).all()]

    # Dodaj dokumenty dla każdej kategorii
    document_ids = []
    for category_name, documents in SAMPLE_DOCUMENTS.items():
        category_id = category_ids[category_name]
        
        for doc_data in documents:
            # Utwórz nowy dokument
            document = Document(
                title=doc_data["title"],
                content=doc_data["content"].strip(),
                category_id=category_id,
                metadata={"source": "sample_data.py", "language": "pl"}
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Generuj embedding dla dokumentu
            content_text = f"{document.title} {document.content}"
            embedding_vector = embedding_generator.get_embedding(content_text)
            
            # Utwórz nowy embedding
            embedding = DocumentEmbedding(
                document_id=document.id,
                model_name=embedding_generator.model_name
            )
            embedding.set_embedding_vector(embedding_vector)
            
            # Dodaj embedding do bazy
            db.add(embedding)
            db.commit()
            
            document_ids.append(document.id)
            logger.info(f"Utworzono dokument: {document.title} (ID: {document.id}, Kategoria: {category_name})")
    
    return document_ids


def main():
    """Funkcja główna generująca przykładowe dane."""
    try:
        # Inicjalizuj bazę danych (jeśli jeszcze nie istnieje)
        init_db()
        
        # Dodaj przykładowe dane
        with get_db() as db:
            logger.info("Generowanie przykładowych kategorii...")
            category_ids = create_sample_categories(db)
            
            logger.info("Generowanie przykładowych dokumentów...")
            document_ids = create_sample_documents(db, category_ids)
            
            logger.info(f"Wygenerowano {len(document_ids)} przykładowych dokumentów.")
        
    except Exception as e:
        logger.error(f"Błąd podczas generowania przykładowych danych: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 