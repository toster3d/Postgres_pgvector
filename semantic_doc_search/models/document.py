"""
Modele dokumentów - re-export z core.models dla backward compatibility.
"""

from semantic_doc_search.core.models import Document, DocumentCreate

__all__ = ['Document', 'DocumentCreate'] 