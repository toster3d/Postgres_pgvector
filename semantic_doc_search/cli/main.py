"""
Main entry point for the semantic document search CLI
"""
import argparse
import logging
import sys

from semantic_doc_search.cli.document_manager import main as doc_main
from semantic_doc_search.cli.search_cli import main as search_main

def main():
    """Main CLI entry point with subcommands for docs and search"""
    parser = argparse.ArgumentParser(
        description="Semantic Document Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  docs    - Document management commands
  search  - Search and recommendation commands

Examples:
  # Add a document
  semantic-docs docs add --title "Sample Document" --content "This is a sample document content" --embed
  
  # Search for similar documents 
  semantic-docs search semantic "What is semantic search?"
  
  # Find similar documents to document #5
  semantic-docs search recommend 5
"""
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Document management subcommand
    docs_parser = subparsers.add_parser("docs", help="Document management commands")
    docs_parser.set_defaults(func=lambda: doc_main())
    
    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search and recommendation commands")
    search_parser.set_defaults(func=lambda: search_main())
    
    # Parse args
    args = parser.parse_args()
    
    # If no subcommand is provided, show help
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Execute the subcommand's main function
    return args.func()

if __name__ == "__main__":
    sys.exit(main()) 