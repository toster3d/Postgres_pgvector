"""
Komendy CLI dla systemu semantycznego wyszukiwania dokumentów.
"""

from .docs import docs_group
from .search import search_group

__all__ = ['docs_group', 'search_group'] 