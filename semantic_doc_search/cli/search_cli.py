"""
CLI for semantic search functionality
"""
import argparse
import logging
import sys
import json
from typing import List, Dict, Any

from semantic_doc_search.database.connection import test_connection
from semantic_doc_search.database.documents import search_documents_by_text
from semantic_doc_search.embeddings.search import (
    semantic_search, hybrid_search, document_recommendations
)

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def format_result(doc: Dict[str, Any], show_content: bool = False, max_content_length: int = 200) -> str:
    """Format a search result for display"""
    result = f"ID: {doc['id']}, Title: {doc['title']}"
    
    if 'similarity' in doc:
        result += f", Similarity: {doc['similarity']:.4f}"
    
    if 'semantic_score' in doc:
        result += f", Semantic: {doc['semantic_score']:.4f}"
    
    if 'text_score' in doc:
        result += f", Text: {doc['text_score']:.4f}"
    
    if 'combined_score' in doc:
        result += f", Combined: {doc['combined_score']:.4f}"
    
    if doc.get('source'):
        result += f"\n  Source: {doc['source']}"
    
    if doc.get('author'):
        result += f"\n  Author: {doc['author']}"
    
    if show_content and 'content' in doc:
        content = doc['content']
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        result += f"\n  Preview: {content}"
    
    return result

def text_search_command(args):
    """Handle text search command"""
    results = search_documents_by_text(args.query, args.limit)
    
    if not results:
        print("No documents found matching the query")
        return True
    
    print(f"\nFound {len(results)} documents matching text query: '{args.query}'\n")
    for doc in results:
        print(format_result(doc, args.show_content))
        print("")
    
    return True

def semantic_search_command(args):
    """Handle semantic search command"""
    results = semantic_search(args.query, args.model, args.limit, args.threshold)
    
    if not results:
        print("No documents found semantically similar to the query")
        return True
    
    print(f"\nFound {len(results)} documents semantically similar to: '{args.query}'\n")
    for doc in results:
        print(format_result(doc, args.show_content))
        print("")
    
    return True

def hybrid_search_command(args):
    """Handle hybrid search command"""
    results = hybrid_search(args.query, args.model, args.limit, args.semantic_weight)
    
    if not results:
        print("No documents found matching the hybrid query")
        return True
    
    print(f"\nFound {len(results)} documents matching hybrid query: '{args.query}'\n")
    for doc in results:
        print(format_result(doc, args.show_content))
        print("")
    
    return True

def recommend_command(args):
    """Handle document recommendation command"""
    results = document_recommendations(args.id, args.model, args.limit)
    
    if not results:
        print(f"No similar documents found for document ID: {args.id}")
        return True
    
    print(f"\nFound {len(results)} documents similar to document ID: {args.id}\n")
    for doc in results:
        print(format_result(doc, args.show_content))
        print("")
    
    return True

def export_results(results: List[Dict[str, Any]], format_type: str, output_file: str = None):
    """Export results to various formats"""
    if format_type == "json":
        # Convert results to JSON-serializable format (remove binary data)
        clean_results = []
        for doc in results:
            clean_doc = {k: v for k, v in doc.items() if isinstance(v, (str, int, float, bool, list, dict))}
            clean_results.append(clean_doc)
        
        json_str = json.dumps(clean_results, indent=2)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"Results exported to {output_file}")
        else:
            print(json_str)
    else:
        print("Unsupported export format")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-connection", action="store_true", help="Test database connection")
    
    subparsers = parser.add_subparsers(dest="command", help="Search command to execute")
    
    # Text search command
    text_parser = subparsers.add_parser("text", help="Perform text-based search")
    text_parser.add_argument("query", help="Search query")
    text_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of results")
    text_parser.add_argument("--show-content", "-s", action="store_true", help="Show content preview")
    text_parser.add_argument("--export", "-e", choices=["json"], help="Export format")
    text_parser.add_argument("--output", "-o", help="Output file for export")
    
    # Semantic search command
    semantic_parser = subparsers.add_parser("semantic", help="Perform semantic search")
    semantic_parser.add_argument("query", help="Search query")
    semantic_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    semantic_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of results")
    semantic_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Similarity threshold (0-1)")
    semantic_parser.add_argument("--show-content", "-s", action="store_true", help="Show content preview")
    semantic_parser.add_argument("--export", "-e", choices=["json"], help="Export format")
    semantic_parser.add_argument("--output", "-o", help="Output file for export")
    
    # Hybrid search command
    hybrid_parser = subparsers.add_parser("hybrid", help="Perform hybrid search (semantic + text)")
    hybrid_parser.add_argument("query", help="Search query")
    hybrid_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    hybrid_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of results")
    hybrid_parser.add_argument("--semantic-weight", "-w", type=float, default=0.7, 
                              help="Weight for semantic search (0-1)")
    hybrid_parser.add_argument("--show-content", "-s", action="store_true", help="Show content preview")
    hybrid_parser.add_argument("--export", "-e", choices=["json"], help="Export format")
    hybrid_parser.add_argument("--output", "-o", help="Output file for export")
    
    # Recommend similar documents command
    recommend_parser = subparsers.add_parser("recommend", help="Recommend similar documents")
    recommend_parser.add_argument("id", type=int, help="Document ID")
    recommend_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    recommend_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of recommendations")
    recommend_parser.add_argument("--show-content", "-s", action="store_true", help="Show content preview")
    recommend_parser.add_argument("--export", "-e", choices=["json"], help="Export format")
    recommend_parser.add_argument("--output", "-o", help="Output file for export")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Test connection if requested
    if args.test_connection:
        if test_connection():
            print("Database connection successful")
            return 0
        else:
            print("Database connection failed")
            return 1
    
    # Execute the requested command
    results = []
    
    if args.command == "text":
        success = text_search_command(args)
        if hasattr(args, 'export') and args.export:
            results = search_documents_by_text(args.query, args.limit)
    elif args.command == "semantic":
        success = semantic_search_command(args)
        if hasattr(args, 'export') and args.export:
            results = semantic_search(args.query, args.model, args.limit, args.threshold)
    elif args.command == "hybrid":
        success = hybrid_search_command(args)
        if hasattr(args, 'export') and args.export:
            results = hybrid_search(args.query, args.model, args.limit, args.semantic_weight)
    elif args.command == "recommend":
        success = recommend_command(args)
        if hasattr(args, 'export') and args.export:
            results = document_recommendations(args.id, args.model, args.limit)
    else:
        parser.print_help()
        return 1
    
    # Export results if requested
    if hasattr(args, 'export') and args.export and results:
        export_results(results, args.export, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 