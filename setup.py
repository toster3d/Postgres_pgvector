from setuptools import setup, find_packages

setup(
    name="semantic-doc-search",
    version="0.1.0",
    description="Semantic document search and recommendation using PostgreSQL and pgvector",
    author="Semantic Search Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "psycopg2-binary>=2.9.5",
        "sentence-transformers>=2.2.2",
        "numpy>=1.22.0",
        "scikit-learn>=1.0.2",
        "openai>=0.27.0",
        "pgvector>=0.1.6",
    ],
    entry_points={
        "console_scripts": [
            "semantic-docs=semantic_doc_search.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
) 