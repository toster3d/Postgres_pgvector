"""
CLI commands for document management
"""
import argparse
import logging
import sys
import os
from typing import List, Optional

from semantic_doc_search.database.connection import test_connection
from semantic_doc_search.database.documents import (
    add_document, get_document, update_document, 
    delete_document, list_documents, search_documents_by_text
)
from semantic_doc_search.embeddings.document_processor import (
    process_document, process_multiple_documents,
    delete_document_embeddings, list_document_embeddings
)

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def read_file_content(file_path: str) -> Optional[str]:
    """Read content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def add_document_command(args):
    """Handle add document command"""
    # Try to read content from file if path provided
    content = args.content
    if args.file and not content:
        content = read_file_content(args.file)
        if content is None:
            print(f"Error: Could not read content from file '{args.file}'")
            return False
    
    # Add document to database
    doc_id = add_document(args.title, content, args.source, args.author)
    if doc_id is None:
        print("Error: Failed to add document")
        return False
    
    # If embeddings should be generated
    if args.embed:
        if process_document(doc_id, args.model):
            print(f"Document added with ID: {doc_id} and embedding generated")
        else:
            print(f"Document added with ID: {doc_id}, but embedding generation failed")
    else:
        print(f"Document added with ID: {doc_id}")
    
    return True

def show_document_command(args):
    """Handle show document command"""
    doc = get_document(args.id)
    if not doc:
        print(f"Error: Document with ID {args.id} not found")
        return False
    
    print("\n" + "="*80)
    print(f"Document ID: {doc['id']}")
    print(f"Title: {doc['title']}")
    if doc.get('source'):
        print(f"Source: {doc['source']}")
    if doc.get('author'):
        print(f"Author: {doc['author']}")
    print(f"Created: {doc['created_at']}")
    print(f"Updated: {doc['updated_at']}")
    print("-"*80)
    print("Content:")
    print(doc['content'])
    print("="*80 + "\n")
    
    # Show embeddings if requested
    if args.embeddings:
        embeddings = list_document_embeddings(args.id)
        if embeddings:
            print("Embeddings:")
            for emb in embeddings:
                print(f"  - ID: {emb['id']}, Model: {emb['model_name']}, Created: {emb['created_at']}")
        else:
            print("No embeddings found for this document")
    
    return True

def update_document_command(args):
    """Handle update document command"""
    update_fields = {}
    
    if args.title:
        update_fields['title'] = args.title
    
    # Update content from file or argument
    if args.file:
        content = read_file_content(args.file)
        if content is None:
            print(f"Error: Could not read content from file '{args.file}'")
            return False
        update_fields['content'] = content
    elif args.content:
        update_fields['content'] = args.content
    
    if args.source:
        update_fields['source'] = args.source
    
    if args.author:
        update_fields['author'] = args.author
    
    if not update_fields:
        print("Error: No fields specified for update")
        return False
    
    # Update document
    if update_document(args.id, **update_fields):
        print(f"Document {args.id} updated successfully")
        
        # Update embeddings if requested
        if args.regenerate_embeddings:
            if delete_document_embeddings(args.id):
                print("Old embeddings deleted")
            
            if process_document(args.id, args.model):
                print("New embeddings generated successfully")
            else:
                print("Failed to generate new embeddings")
        
        return True
    else:
        print(f"Error: Failed to update document {args.id}")
        return False

def delete_document_command(args):
    """Handle delete document command"""
    # Delete embeddings first (due to foreign key constraints)
    if not args.keep_embeddings:
        delete_document_embeddings(args.id)
    
    # Delete document
    if delete_document(args.id):
        print(f"Document {args.id} deleted successfully")
        return True
    else:
        print(f"Error: Failed to delete document {args.id}")
        return False

def list_documents_command(args):
    """Handle list documents command"""
    documents = list_documents(args.limit, args.offset)
    
    if not documents:
        print("No documents found")
        return True
    
    print(f"\nFound {len(documents)} documents:\n")
    for doc in documents:
        print(f"ID: {doc['id']}, Title: {doc['title']}")
        if doc.get('source'):
            print(f"  Source: {doc['source']}")
        if doc.get('author'):
            print(f"  Author: {doc['author']}")
        print(f"  Created: {doc['created_at']}")
        print("")
    
    return True

def embeddings_command(args):
    """Handle embeddings command"""
    if args.list:
        embeddings = list_document_embeddings(args.id)
        if not embeddings:
            print(f"No embeddings found for document {args.id}")
        else:
            print(f"Embeddings for document {args.id}:")
            for emb in embeddings:
                print(f"  - ID: {emb['id']}, Model: {emb['model_name']}, Created: {emb['created_at']}")
    
    if args.generate:
        if process_document(args.id, args.model):
            print(f"Embeddings generated for document {args.id} using model {args.model}")
        else:
            print(f"Failed to generate embeddings for document {args.id}")
    
    if args.delete:
        if delete_document_embeddings(args.id):
            print(f"Embeddings deleted for document {args.id}")
        else:
            print(f"No embeddings found for document {args.id}")
    
    return True

def process_documents_command(args):
    """Handle batch processing of documents"""
    doc_ids = args.ids
    results = process_multiple_documents(doc_ids, args.model)
    
    success = [doc_id for doc_id, success in results.items() if success]
    failed = [doc_id for doc_id, success in results.items() if not success]
    
    if success:
        print(f"Successfully processed {len(success)} documents: {', '.join(map(str, success))}")
    
    if failed:
        print(f"Failed to process {len(failed)} documents: {', '.join(map(str, failed))}")
    
    return len(failed) == 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Document Management CLI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-connection", action="store_true", help="Test database connection")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add document command
    add_parser = subparsers.add_parser("add", help="Add a new document")
    add_parser.add_argument("--title", "-t", required=True, help="Document title")
    add_parser.add_argument("--content", "-c", help="Document content")
    add_parser.add_argument("--file", "-f", help="Path to file containing document content")
    add_parser.add_argument("--source", "-s", help="Document source")
    add_parser.add_argument("--author", "-a", help="Document author")
    add_parser.add_argument("--embed", "-e", action="store_true", help="Generate embeddings for document")
    add_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    
    # Show document command
    show_parser = subparsers.add_parser("show", help="Show document details")
    show_parser.add_argument("id", type=int, help="Document ID")
    show_parser.add_argument("--embeddings", "-e", action="store_true", help="Show document embeddings")
    
    # Update document command
    update_parser = subparsers.add_parser("update", help="Update document")
    update_parser.add_argument("id", type=int, help="Document ID")
    update_parser.add_argument("--title", "-t", help="Document title")
    update_parser.add_argument("--content", "-c", help="Document content")
    update_parser.add_argument("--file", "-f", help="Path to file containing document content")
    update_parser.add_argument("--source", "-s", help="Document source")
    update_parser.add_argument("--author", "-a", help="Document author")
    update_parser.add_argument("--regenerate-embeddings", "-r", action="store_true", 
                               help="Regenerate embeddings after update")
    update_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    
    # Delete document command
    delete_parser = subparsers.add_parser("delete", help="Delete document")
    delete_parser.add_argument("id", type=int, help="Document ID")
    delete_parser.add_argument("--keep-embeddings", "-k", action="store_true", 
                              help="Keep embeddings (not recommended, may cause foreign key errors)")
    
    # List documents command
    list_parser = subparsers.add_parser("list", help="List documents")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of documents to list")
    list_parser.add_argument("--offset", "-o", type=int, default=0, help="Offset for pagination")
    
    # Embeddings command
    embeddings_parser = subparsers.add_parser("embeddings", help="Manage document embeddings")
    embeddings_parser.add_argument("id", type=int, help="Document ID")
    embeddings_parser.add_argument("--list", "-l", action="store_true", help="List embeddings")
    embeddings_parser.add_argument("--generate", "-g", action="store_true", help="Generate embeddings")
    embeddings_parser.add_argument("--delete", "-d", action="store_true", help="Delete embeddings")
    embeddings_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    
    # Process documents in batch
    process_parser = subparsers.add_parser("process", help="Process multiple documents")
    process_parser.add_argument("ids", type=int, nargs="+", help="Document IDs to process")
    process_parser.add_argument("--model", "-m", default="sklearn", help="Embedding model to use")
    
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
    if args.command == "add":
        success = add_document_command(args)
    elif args.command == "show":
        success = show_document_command(args)
    elif args.command == "update":
        success = update_document_command(args)
    elif args.command == "delete":
        success = delete_document_command(args)
    elif args.command == "list":
        success = list_documents_command(args)
    elif args.command == "embeddings":
        success = embeddings_command(args)
    elif args.command == "process":
        success = process_documents_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 