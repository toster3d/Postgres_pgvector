# semantic_doc_search/cli/__init__.py
"""
Moduł CLI dla systemu semantycznego wyszukiwania dokumentów.

Zawiera interfejs wiersza poleceń zbudowany na Click 8.2.0 z Rich
dla kolorowego i interaktywnego interfejsu użytkownika.
"""

from semantic_doc_search.cli.main import cli
from semantic_doc_search.cli.commands.docs import docs_group
from semantic_doc_search.cli.commands.search import search_group

__all__ = ["cli", "docs_group", "search_group"]