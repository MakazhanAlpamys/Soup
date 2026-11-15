"""Offline documentation search and indexing for Soup CLI."""

from soup_cli.docs.bm25 import PureBM25
from soup_cli.docs.chunker import split_markdown_by_headings
from soup_cli.docs.retriever import search_docs
from soup_cli.docs.tokenizer import tokenize

__all__ = [
    "PureBM25",
    "search_docs",
    "split_markdown_by_headings",
    "tokenize",
]
