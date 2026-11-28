"""Document indexer, safe cache manager, and retrieval engine for Soup CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

from soup_cli.docs.bm25 import PureBM25
from soup_cli.docs.chunker import split_markdown_by_headings
from soup_cli.docs.tokenizer import tokenize

SCHEMA_VERSION = 1
INDEX_CONFIG = {
    "k1": 1.5,
    "b": 0.75,
    "eps": 0.25,
    "tokenizer": "domain_aware_v1",
}

_ALIASES = {
    ("continue", "training"): "resume",
    ("batch", "size"): "batch-size",
}
_REFERENCE_PHRASES = (
    ("without", "a", "reference", "model"),
    ("without", "reference", "model"),
    ("no", "reference", "model"),
    ("reference-free",),
)
_REFERENCE_HEADING = re.compile(
    r"\b(?:no reference model|without (?:a )?reference model|reference[- ]free)\b"
)


def _query_expansions(query: str) -> tuple[dict[str, float], bool]:
    """Infer terms from prose without treating flag/path components as prose."""
    words = re.findall(r"[\w./:\\+-]+", query.casefold())
    literal = set(tokenize(query))
    inferred: dict[str, float] = {}
    if {"mac", "macs"}.intersection(words):
        inferred.update({term: 0.7 for term in {"mac", "macs"} - literal})
        if {"train", "training", "fine-tune"}.intersection(words):
            inferred["mlx"] = 0.65

    def contains(phrase: tuple[str, ...]) -> bool:
        return any(tuple(words[pos : pos + len(phrase)]) == phrase for pos in range(len(words)))

    inferred.update({term: 0.65 for phrase, term in _ALIASES.items() if contains(phrase)})
    return (
        {term: weight for term, weight in inferred.items() if term not in literal},
        any(contains(phrase) for phrase in _REFERENCE_PHRASES),
    )


def _rank_chunks(
    query: str, index: PureBM25, chunks: Sequence[dict[str, Any]], top_k: int
) -> list[tuple[int, float]]:
    """Apply query-side lexical signals before the result cutoff."""
    tokens = tokenize(query)
    scores = index.score(tokens)
    inferred, reference_free = _query_expansions(query)
    for term, weight in sorted(inferred.items()):
        for doc_id, score in enumerate(index.score([term])):
            scores[doc_id] += weight * score

    prose_terms = {term for term in re.findall(r"[\w./:\\+-]+", query.casefold()) if term.isalpha()}
    literal_idf = {term: max(0.0, index.idf.get(term, 0.0)) for term in sorted(prose_terms)}
    for doc_id, chunk in enumerate(chunks):
        if scores[doc_id] <= 0:
            continue
        section = chunk["section"]
        if reference_free and _REFERENCE_HEADING.search(section.casefold()):
            scores[doc_id] += 8.0
        if section != "Overview":
            heading_tokens = set(tokenize(section))
            heading_score = sum(idf for term, idf in literal_idf.items() if term in heading_tokens)
            scores[doc_id] += min(0.25 * scores[doc_id], 0.3 * heading_score)

    order = sorted(range(len(scores)), key=lambda doc_id: (-scores[doc_id], doc_id))
    return [(doc_id, scores[doc_id]) for doc_id in order[:top_k]]


def get_default_cache_dir() -> Path:
    """Return default cache directory for documentation search index."""
    return Path.home() / ".soup" / "cache" / "docs"


def load_docs() -> dict[str, str]:
    """Load Markdown documentation files from source checkout docs/ or package data."""
    doc_files: dict[str, str] = {}

    # 1. Source checkout: locate repository root docs/ directory
    try:
        root = Path(__file__).resolve().parents[3]
        repo_docs = root / "docs"
        if (root / "pyproject.toml").is_file() and (root / "src" / "soup_cli").is_dir():
            for entry in sorted(repo_docs.glob("*.md")):
                doc_files[entry.name] = entry.read_text(encoding="utf-8")
            if doc_files:
                return doc_files
    except (IndexError, OSError, UnicodeError):
        pass

    # 2. Installed wheel: read package data bundled at build time via force-include
    try:
        bundled_pkg = resources.files("soup_cli.docs").joinpath("bundled")
        if bundled_pkg.is_dir():
            for entry in sorted(bundled_pkg.iterdir(), key=lambda p: p.name):
                if entry.name.endswith(".md"):
                    doc_files[entry.name] = entry.read_text(encoding="utf-8")
    except Exception:
        pass

    if not doc_files:
        raise FileNotFoundError("No documentation files found in docs/ or package data")

    return doc_files


def compute_corpus_fingerprint(doc_files: dict[str, str]) -> str:
    """Compute deterministic SHA-256 fingerprint over sorted document contents."""
    sorted_hashes = [
        (name, hashlib.sha256(content.encode("utf-8")).hexdigest())
        for name, content in sorted(doc_files.items())
    ]
    fingerprint_bytes = json.dumps(sorted_hashes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(fingerprint_bytes).hexdigest()


def build_corpus_chunks(doc_files: dict[str, str]) -> list[dict[str, Any]]:
    """Parse and chunk all documentation files."""
    chunks: list[dict[str, Any]] = []
    for filename, content in sorted(doc_files.items()):
        doc_name = filename[:-3] if filename.endswith(".md") else filename
        doc_chunks = split_markdown_by_headings(content, doc_name)
        chunks.extend(doc_chunks)
    return chunks


def build_index(chunks: Sequence[dict[str, Any]]) -> PureBM25:
    """Tokenize passage texts and fit PureBM25 inverted index."""
    tokenized_docs: list[list[str]] = []
    for c in chunks:
        passage_text = f"{c.get('heading_prefix', '')}\n\n{c.get('text', '')}"
        tokenized_docs.append(tokenize(passage_text))
    return PureBM25(
        tokenized_docs,
        k1=INDEX_CONFIG["k1"],
        b=INDEX_CONFIG["b"],
        eps=INDEX_CONFIG["eps"],
    )


def save_cache(
    cache_path: Path,
    fingerprint: str,
    chunks: list[dict[str, Any]],
    index: PureBM25,
) -> None:
    """Atomically save versioned documentation index cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_fingerprint": fingerprint,
        "config": INDEX_CONFIG,
        "chunks": chunks,
        "index": index.to_dict(),
    }

    # Write to temporary file in same directory and atomic rename
    fd, tmp_path_str = tempfile.mkstemp(dir=cache_path.parent, prefix="docs_index_", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_cache(
    cache_path: Path, expected_fingerprint: str
) -> tuple[PureBM25, list[dict[str, Any]]] | None:
    """Load cached index if present, valid, and matching current fingerprint."""
    if not cache_path.is_file():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        if data.get("corpus_fingerprint") != expected_fingerprint:
            return None
        if data.get("config") != INDEX_CONFIG:
            return None

        chunks = data["chunks"]
        index = PureBM25.from_dict(data["index"])
        return index, chunks
    except Exception:
        return None


def get_or_build_index(cache_dir: Path | None = None) -> tuple[PureBM25, list[dict[str, Any]]]:
    """Get active search index, loading from cache or building lazily."""
    cache_directory = cache_dir or get_default_cache_dir()
    cache_file = cache_directory / "docs_index.json"

    doc_files = load_docs()
    fingerprint = compute_corpus_fingerprint(doc_files)

    cached = load_cache(cache_file, fingerprint)
    if cached is not None:
        return cached

    chunks = build_corpus_chunks(doc_files)
    index = build_index(chunks)
    try:
        save_cache(cache_file, fingerprint, chunks, index)
    except Exception:
        # Cache write failures (e.g. read-only filesystem) should not break search
        pass

    return index, chunks


def search_docs(
    query: str,
    top_k: int = 5,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Search documentation for query terms using BM25 retrieval."""
    if not query or not query.strip() or top_k < 1:
        return []

    index, chunks = get_or_build_index(cache_dir=cache_dir)
    ranked = _rank_chunks(query, index, chunks, top_k=top_k)

    results = []
    for idx, score in ranked:
        if score > 0:
            item = dict(chunks[idx])
            item["score"] = round(score, 4)
            results.append(item)
    return results
