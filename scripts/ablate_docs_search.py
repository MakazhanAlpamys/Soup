"""Development-only ablation of #3192's three lexical signals."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from check_docs_search_ranking import compare
from rich.console import Console

from soup_cli.docs.retriever import (
    _REFERENCE_HEADING,
    _query_expansions,
    _rank_chunks,
    build_corpus_chunks,
    build_index,
    load_docs,
)
from soup_cli.docs.tokenizer import tokenize


def main() -> None:
    root = Path(__file__).parents[1]
    fixture = json.loads(
        (root / "tests/fixtures/docs_search/maintainer_dev.json").read_text(encoding="utf-8")
    )
    compare(fixture)  # Validate the frozen docs, chunks, labels and baseline first.
    chunks = build_corpus_chunks(load_docs())
    index = build_index(chunks)
    output = {}
    for inflection, aliases, heading in itertools.product([False, True], repeat=3):
        rows = []
        for query in fixture["queries"]:
            text = query["query"]
            tokens = tokenize(text)
            scores = index.score(tokens)
            inferred, reference_free = _query_expansions(text)
            for term, weight in sorted(inferred.items()):
                if (weight == 0.7 and inflection) or (weight != 0.7 and aliases):
                    for doc_id, score in enumerate(index.score([term])):
                        scores[doc_id] += weight * score
            prose = {
                term for term in re.findall(r"[\w./:\\+-]+", text.casefold()) if term.isalpha()
            }
            for doc_id, chunk in enumerate(chunks):
                if scores[doc_id] <= 0:
                    continue
                section = chunk["section"]
                if aliases and reference_free and _REFERENCE_HEADING.search(section.casefold()):
                    scores[doc_id] += 8.0
                if heading and section != "Overview":
                    matches = prose.intersection(tokenize(section))
                    increment = sum(max(0, index.idf.get(term, 0)) for term in sorted(matches))
                    scores[doc_id] += min(0.25 * scores[doc_id], 0.3 * increment)
            order = sorted(range(len(scores)), key=lambda doc_id: (-scores[doc_id], doc_id))
            ranking = [(doc_id, scores[doc_id]) for doc_id in order]
            if inflection and aliases and heading:
                assert ranking == _rank_chunks(text, index, chunks, len(chunks))
            elif not any([inflection, aliases, heading]):
                assert ranking == index.rank(tokens, len(chunks))
            rank = next(
                (
                    pos
                    for pos, (doc_id, score) in enumerate(ranking, 1)
                    if score > 0 and str(doc_id) in query["targets"]
                ),
                None,
            )
            rows.append({"id": query["id"], "query": text, "rank": rank})
        output[f"{int(inflection)}{int(aliases)}{int(heading)}"] = {
            "hit5": sum(row["rank"] is not None and row["rank"] <= 5 for row in rows),
            "mrr10": sum(
                1 / row["rank"] if row["rank"] is not None and row["rank"] <= 10 else 0
                for row in rows
            )
            / len(rows),
            "rows": rows,
        }
    Console().print_json(data=output)


if __name__ == "__main__":
    main()
