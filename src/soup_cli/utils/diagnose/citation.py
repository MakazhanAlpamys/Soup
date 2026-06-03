"""Citation-recall regression probe (v0.71.10 #202).

A RAFT-trained model is supposed to cite the supporting ``[doc-id]`` in its
answer. This probe runs each RAFT-shaped row (``{query, golden_doc,
distractor_docs, answer}``) through the adapter generator, extracts the cited
document ids, and measures recall against the ground-truth golden-doc id.

The ``FailureScore`` follows the same OK/MINOR/MAJOR taxonomy as every other
diagnose mode (``classify_score`` on the mean recall). The training-config
``citation_recall_threshold`` is a separate eval-gate knob — the diagnose
badge uses the shared 0.85/0.60 bands so it reads consistently next to the
other six probes.

Heavy imports are avoided here: ``utils.raft`` + ``citation_faithful`` are
pure-Python; only the caller's generator closure touches a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from soup_cli.utils.diagnose._common import (
    GeneratorFn,
    call_generator,
    merge_evidence,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score

_MAX_CITATION_ROWS = 256


def is_raft_row(row: object) -> bool:
    """True when ``row`` carries the RAFT fields the citation probe needs."""
    if not isinstance(row, Mapping):
        return False
    return (
        isinstance(row.get("query"), str)
        and isinstance(row.get("golden_doc"), str)
        and isinstance(row.get("answer"), str)
    )


def score_citation(
    rows: Sequence[Mapping[str, object]],
    generator: GeneratorFn,
    *,
    citation_style: str = "bracket",
    shuffle_seed: int | None = None,
    max_rows: int = _MAX_CITATION_ROWS,
) -> FailureScore:
    """Measure citation recall of ``generator`` over RAFT rows.

    For each row the prompt is composed (with deterministic ``[doc-N]`` ids),
    the generator produces an answer, and recall = ``1.0`` when the golden
    doc id is among the cited ids (else ``0.0``). The mean recall becomes the
    score; the verdict follows ``classify_score``.

    Raises ``ValueError`` when no usable RAFT row is supplied (so the caller
    falls back to a neutral score rather than reporting a misleading 0.0).
    """
    from soup_cli.utils.citation_faithful import extract_citation_ids
    from soup_cli.utils.raft import build_raft_prompt

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mappings")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive int")

    raft_rows = [r for r in rows if is_raft_row(r)][:max_rows]
    if not raft_rows:
        raise ValueError("no RAFT-shaped rows for the citation probe")

    recalls: list[float] = []
    cited_total = 0
    for index, row in enumerate(raft_rows):
        composed = build_raft_prompt(row, shuffle_seed=shuffle_seed, row_index=index)
        output = call_generator(generator, composed.prompt)
        cited = set(extract_citation_ids(output, style=citation_style))
        cited_total += len(cited)
        recalls.append(1.0 if composed.golden_doc_id in cited else 0.0)

    mean_recall = sum(recalls) / len(recalls)
    score = max(0.0, min(1.0, mean_recall))
    evidence = merge_evidence(
        {
            "rows": len(raft_rows),
            "mean_recall": mean_recall,
            "cited_total": cited_total,
            "style": citation_style,
        }
    )
    return FailureScore(
        mode="citation",
        score=score,
        verdict=classify_score(score),
        evidence=evidence,
    )
