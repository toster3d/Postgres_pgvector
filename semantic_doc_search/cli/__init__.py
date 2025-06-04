# semantic_doc_search/cli/__init__.py
"""
Moduł CLI dla systemu semantycznego wyszukiwania dokumentów.

Zawiera interfejs wiersza poleceń zbudowany na Click 8.2.0 z Rich
dla kolorowego i interaktywnego interfejsu użytkownika.
"""

from semantic_doc_search.cli.main import cli
from semantic_doc_search.cli.document_manager import document_group
from semantic_doc_search.cli.search_commands import search_group

__all__ = ["cli", "document_group", "search_group"]