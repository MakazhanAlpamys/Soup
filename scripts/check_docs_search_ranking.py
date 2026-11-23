"""Compare deterministic docs-search rankings on a frozen, repository-local fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from rich.console import Console

from soup_cli.docs.retriever import _rank_chunks, build_corpus_chunks, build_index, load_docs
from soup_cli.docs.tokenizer import tokenize


def compare(fixture: dict[str, Any], *, require_hit5_gain: bool = False) -> dict[str, Any]:
    """Enforce option B with exact arithmetic and disclose individual rank losses."""
    if "documents" in fixture:
        docs = load_docs()
        hashes = {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in docs.items()
        }
        if hashes != fixture["documents"]:
            raise ValueError("documentation snapshot differs from the frozen fixture")
        chunks = [
            dict(chunk, chunk_id=str(number))
            for number, chunk in enumerate(build_corpus_chunks(docs))
        ]
    else:
        chunks = fixture["chunks"]
    corpus_hash = hashlib.sha256(
        json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if "corpus_sha256" in fixture and corpus_hash != fixture["corpus_sha256"]:
        raise ValueError("chunking differs from the frozen fixture")
    queries = fixture["queries"]
    if not chunks or not queries:
        raise ValueError("ranking fixtures require nonempty chunks and queries")
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("duplicate chunk IDs")
    query_ids = [query["id"] for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query IDs")
    known = set(chunk_ids)
    for query in queries:
        if not isinstance(query["query"], str) or not query["query"].strip():
            raise ValueError("queries must contain nonempty text")
        if not query["targets"] or not set(query["targets"]) <= known:
            raise ValueError(f"invalid targets for {query['id']}")

    index = build_index(chunks)
    rows = []
    totals = {
        arm: dict.fromkeys(["hit1", "hit5", "mrr10", "recall5"], Fraction())
        for arm in ["baseline", "candidate"]
    }
    for query in queries:
        targets = set(query["targets"])
        row = {"id": query["id"], "query": query["query"], "targets": sorted(targets)}
        rankings = {
            "baseline": index.rank(tokenize(query["query"]), len(chunks)),
            "candidate": _rank_chunks(query["query"], index, chunks, len(chunks)),
        }
        for arm, ranked in rankings.items():
            ordered = [chunk_ids[doc_id] for doc_id, score in ranked if score > 0]
            if arm == "baseline" and "baseline_top10" in query:
                if ordered[:10] != query["baseline_top10"]:
                    raise ValueError(f"baseline ranking changed for {query['id']}")
            rank = next(
                (pos for pos, chunk_id in enumerate(ordered, 1) if chunk_id in targets), None
            )
            values = {
                "hit1": Fraction(rank == 1),
                "hit5": Fraction(rank is not None and rank <= 5),
                "mrr10": Fraction(1, rank) if rank is not None and rank <= 10 else Fraction(),
                "recall5": Fraction(len(targets.intersection(ordered[:5])), len(targets)),
            }
            for name, value in values.items():
                totals[arm][name] += value
            row[arm] = {"rank": rank, "top10": ordered[:10]}
        rows.append(row)

    aggregates_pass = all(
        totals["candidate"][key] >= totals["baseline"][key] for key in totals["baseline"]
    )
    fresh_gain = totals["candidate"]["hit5"] > totals["baseline"]["hit5"]
    losses = [
        {
            "id": row["id"],
            "query": row["query"],
            "old_rank": row["baseline"]["rank"],
            "new_rank": row["candidate"]["rank"],
        }
        for row in rows
        if (row["candidate"]["rank"] or float("inf")) > (row["baseline"]["rank"] or float("inf"))
    ]
    return {
        "cohort": fixture["cohort"],
        "corpus_sha256": corpus_hash,
        "query_count": len(queries),
        "gate": "option B: aggregate preservation, all individual losses disclosed",
        "passed": aggregates_pass and (fresh_gain or not require_hit5_gain),
        "require_hit5_gain": require_hit5_gain,
        "metrics": {
            arm: {key: float(value / len(queries)) for key, value in values.items()}
            for arm, values in totals.items()
        },
        "rank_losses": losses,
        "per_query": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-hit5-gain", action="store_true")
    args = parser.parse_args()
    console = Console()
    try:
        data = args.fixture.read_bytes()
        if args.output and args.output.exists() and os.path.samefile(args.fixture, args.output):
            raise ValueError("output must not replace the frozen fixture")
        report = compare(json.loads(data), require_hit5_gain=args.require_hit5_gain)
        report["fixture_sha256"] = hashlib.sha256(data).hexdigest()
        report["ranker_sha256"] = hashlib.sha256(
            Path(__file__).parents[1].joinpath("src/soup_cli/docs/retriever.py").read_bytes()
        ).hexdigest()
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        console.print_json(data=report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        console.print(f"Invalid ranking comparison: {error}", markup=False)
        raise SystemExit(2) from error
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
