"""Rebuild the v1 docs-search gate and migration audit without scoring queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from soup_cli.docs.chunker import _FENCE_RE, _HEADING_RE, split_markdown_by_headings

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "docs_search"


def _historical_owners(text: str, chunks: list[dict[str, Any]]) -> list[tuple[str | None, int]]:
    """Locate old headings, including code comments, in their clean owning sections."""
    lines = text.splitlines()
    owners: list[tuple[str | None, int]] = []
    first = next((pos for pos, line in enumerate(lines) if _HEADING_RE.match(line)), len(lines))
    if first:
        owners.append((None, 0))
    owner = 0 if "\n".join(lines[:first]).strip() else -1
    fence: str | None = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0] * len(marker)
            elif marker.startswith(fence[0]) and len(marker) >= len(fence):
                fence = None
            continue
        match = _HEADING_RE.match(line)
        if match:
            heading = match.group(2).strip()
            if fence is None:
                owner += 1
                if owner >= len(chunks) or chunks[owner]["section"] != heading:
                    raise ValueError("heading ownership differs from the shipped chunker")
            owners.append((heading, owner))
    return owners


def build_fixture(
    source: dict[str, Any], documents: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild only from pinned docs; historical passage bodies are never inputs."""
    hashes = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in documents.items()
    }
    if hashes != source["documents"]:
        raise ValueError("pinned document hashes differ")
    chunks = []
    owners = {}
    for name, text in sorted(documents.items()):
        doc_name = name.removesuffix(".md")
        sections = split_markdown_by_headings(text, doc_name)
        owners[doc_name] = _historical_owners(text, sections)
        chunks.extend(
            dict(section, chunk_id=f"{doc_name}::section_{number:03d}")
            for number, section in enumerate(sections)
        )
    mapping = {}
    known = {chunk["chunk_id"] for chunk in chunks}
    for passage in source["passages"]:
        heading, number = owners[passage["doc_name"]][passage["section_index"]]
        if heading != passage["heading"]:
            raise ValueError(f"historical heading differs: {passage['chunk_id']}")
        new_id = f"{passage['doc_name']}::section_{number:03d}"
        if new_id not in known or passage["chunk_id"] in mapping:
            raise ValueError("invalid historical passage mapping")
        mapping[passage["chunk_id"]] = new_id
    annotations = {row["query_id"]: row for row in source["annotations"]}
    query_ids = [row["query_id"] for row in source["queries"]]
    if (
        len(annotations) != len(source["annotations"])
        or len(set(query_ids)) != len(query_ids)
        or set(query_ids) != set(annotations)
    ):
        raise ValueError("query and annotation IDs differ or repeat")
    queries = []
    labels = []
    for query in source["queries"]:
        annotation = annotations[query["query_id"]]
        old_targets = annotation["relevant_chunk_ids"]
        targets = sorted({mapping[old_id] for old_id in old_targets})
        if not targets or annotation["status"] != "resolved":
            raise ValueError("queries must retain resolved, nonempty targets")
        queries.append({"id": query["query_id"], "query": query["query"], "targets": targets})
        labels.append(
            {
                "id": query["query_id"],
                "query": query["query"],
                "old_status": annotation["status"],
                "new_status": "resolved",
                "old_targets": old_targets,
                "new_targets": targets,
                "label_policy": "union of clean owning sections; deduplicate; "
                "no new relevance judgments",
            }
        )
    disputed = []
    for old_id in source["disputed_passages"]:
        new_id = mapping[old_id]
        old_queries = [row["id"] for row in labels if old_id in row["old_targets"]]
        new_queries = [row["id"] for row in labels if new_id in row["new_targets"]]
        disputed.append(
            {
                "old_chunk_id": old_id,
                "new_chunk_id": new_id,
                "old_target_queries": old_queries,
                "new_target_queries": new_queries,
                "old_label_status": "positive target only for listed queries; otherwise unjudged",
                "new_label_status": "positive target only for listed queries; otherwise unjudged",
                "text_status": "old body excluded; section rebuilt entirely from pinned Soup docs",
            }
        )
    corpus_hash = hashlib.sha256(
        json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    fixture = {
        "cohort": "original28-pinned-docs-v1",
        "fixture_version": 1,
        "docs_commit": source["docs_commit"],
        "corpus_sha256": corpus_hash,
        "chunks": chunks,
        "queries": queries,
    }
    audit = {
        "fixture_version": 1,
        "docs_commit": source["docs_commit"],
        "documents": dict(sorted(hashes.items())),
        "historical_passage_count": len(source["passages"]),
        "rebuilt_section_count": len(chunks),
        "policy": "All historical passage IDs retired; every new section rebuilt by Soup's "
        "fence-aware chunker. Model-based sub-splitting and normalization are removed.",
        "passage_mapping": mapping,
        "disputed_passages": disputed,
        "query_labels": labels,
    }
    return fixture, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write instead of checking artifacts")
    args = parser.parse_args()
    console = Console()
    try:
        source = json.loads((FIXTURES / "original28-source.json").read_bytes())
        revision = source["docs_commit"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("invalid pinned docs commit")
        documents = {}
        for name in source["documents"]:
            if Path(name).name != name or not name.endswith(".md"):
                raise ValueError("invalid pinned docs filename")
            documents[name] = subprocess.check_output(
                ["git", "show", f"{revision}:docs/{name}"], cwd=ROOT
            ).decode("utf-8")
        fixture, audit = build_fixture(source, documents)
        outputs = {"pinned_docs_v1.json": fixture, "pinned_docs_v1-migration.json": audit}
        for name, value in outputs.items():
            data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            path = FIXTURES / name
            if args.write:
                path.write_bytes(data)
            elif path.read_bytes() != data:
                raise ValueError(f"rebuilt artifact differs: {name}")
        console.print(
            f"{'Wrote' if args.write else 'Verified'} pinned-docs v1: "
            f"{len(documents)} docs, {len(fixture['chunks'])} sections, "
            f"{len(fixture['queries'])} queries; no queries scored."
        )
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as error:
        console.print(f"Invalid fixture rebuild: {error}", markup=False)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
