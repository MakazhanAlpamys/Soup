"""v0.71.10 #202 — `soup eval citation`.

Scores citation precision / recall / F1 per row + an aggregate, routed through
``citation_faithful.score_citations``. Attached to the existing ``soup eval``
Typer app via :func:`register` (mirrors the v0.55.0 / v0.61.0 / v0.65.0
registration pattern so ``commands/eval.py`` stays under length cap).

Input JSONL accepts two row shapes:

* ``{"predicted": str, "expected_ids": [str, ...]}`` — scored directly.
* RAFT rows ``{"query", "golden_doc", "distractor_docs", "answer"}`` — the
  ``answer`` is treated as the prediction and the ground-truth citation is the
  golden document's deterministic ``[doc-N]`` id (the same id the RAFT trainer
  assigns), so a model trained with ``citation_faithful`` can be scored against
  its own training data.
"""

from __future__ import annotations

import json
import os
import stat
from typing import List, Mapping, Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

_MAX_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB
_MAX_ROWS = 1_000_000


def _load_jsonl_rows(path: str, console: Console) -> List[dict]:
    """Load JSONL rows with cwd containment + O_NOFOLLOW + size/row caps.

    Mirrors the v0.65.0 ``irt.load_response_rows`` TOCTOU-safe reader: the
    cwd-containment helper rejects out-of-tree / symlink paths, then the open
    uses ``O_NOFOLLOW`` (POSIX) so a symlink swapped in after the check still
    fails. Malformed JSON lines are skipped (counted toward the row cap).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(path, "citation data")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_RDONLY | no_follow)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"citation data not found: {os.path.basename(path)}"
        ) from exc
    except OSError as exc:
        raise typer.BadParameter(
            f"citation data cannot be opened (symlink?): {type(exc).__name__}"
        ) from exc
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise typer.BadParameter("citation data must be a regular file")
        if st.st_size > _MAX_FILE_BYTES:
            raise typer.BadParameter(
                f"citation data exceeds {_MAX_FILE_BYTES} bytes"
            )
        rows: List[dict] = []
        skipped = 0
        with os.fdopen(fd, "r", encoding="utf-8-sig") as handle:
            for count, line in enumerate(handle):
                if count >= _MAX_ROWS:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    skipped += 1
    except Exception:
        # fdopen may not have taken ownership of fd on early raise.
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    if skipped:
        console.print(f"[dim]Skipped {skipped} malformed / non-object row(s).[/]")
    return rows


def _row_predicted_expected(
    row: Mapping[str, object], *, row_index: int, shuffle_seed: Optional[int] = None
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Resolve a row into ``(predicted_text, expected_ids)`` or None to skip.

    For a RAFT-shaped row the golden ``[doc-N]`` id is derived from
    ``build_raft_prompt`` — which shuffles document order by
    ``(shuffle_seed, row_index)``. To score a model against its OWN training
    data, ``shuffle_seed`` MUST match the ``data.raft_shuffle_seed`` used at
    train time (code-review L4); otherwise the golden id won't line up.
    """
    predicted = row.get("predicted")
    expected = row.get("expected_ids")
    if isinstance(predicted, str) and isinstance(expected, list):
        ids = tuple(e for e in expected if isinstance(e, str) and e)
        return predicted, ids
    # RAFT-shaped row → answer is the prediction, golden id the expectation.
    if (
        isinstance(row.get("query"), str)
        and isinstance(row.get("golden_doc"), str)
        and isinstance(row.get("answer"), str)
    ):
        from soup_cli.utils.raft import build_raft_prompt

        composed = build_raft_prompt(
            row, shuffle_seed=shuffle_seed, row_index=row_index
        )
        return composed.answer, (composed.golden_doc_id,)
    return None


def register(app: typer.Typer, console: Console) -> None:
    """Attach ``soup eval citation`` to the eval Typer app."""

    @app.command(name="citation")
    def citation(
        data: str = typer.Argument(
            ...,
            help=(
                "JSONL of {predicted, expected_ids} rows OR RAFT rows "
                "{query, golden_doc, distractor_docs, answer}."
            ),
        ),
        style: str = typer.Option(
            "bracket", "--style",
            help="Citation style: bracket / inline / footnote.",
        ),
        shuffle_seed: Optional[int] = typer.Option(
            None, "--shuffle-seed",
            help=(
                "For RAFT-shaped rows: the data.raft_shuffle_seed used at "
                "train time, so the golden [doc-N] id matches what the model "
                "saw. Ignored for {predicted, expected_ids} rows."
            ),
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Write the per-row + aggregate CitationScore JSON here.",
        ),
    ) -> None:
        """Score citation precision / recall / F1 over a JSONL (#202)."""
        from soup_cli.utils.citation_faithful import (
            score_citations,
            validate_citation_style,
        )

        try:
            canonical_style = validate_citation_style(style)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Invalid --style:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        rows = _load_jsonl_rows(data, console)
        if not rows:
            console.print("[red]No usable rows in citation data.[/]")
            raise typer.Exit(2)

        table = Table(title="Citation scores", border_style="cyan")
        table.add_column("Row", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")

        scored: List[dict] = []
        prec_sum = rec_sum = f1_sum = 0.0
        for index, row in enumerate(rows):
            resolved = _row_predicted_expected(
                row, row_index=index, shuffle_seed=shuffle_seed
            )
            if resolved is None:
                continue
            predicted, expected_ids = resolved
            try:
                cs = score_citations(
                    predicted=predicted,
                    expected_ids=list(expected_ids),
                    style=canonical_style,
                )
            except (TypeError, ValueError) as exc:
                console.print(
                    f"[yellow]Row {index} skipped:[/] {escape(str(exc))}"
                )
                continue
            prec_sum += cs.precision
            rec_sum += cs.recall
            f1_sum += cs.f1
            scored.append({
                "row": index,
                "precision": cs.precision,
                "recall": cs.recall,
                "f1": cs.f1,
                "predicted_count": cs.predicted_count,
                "expected_count": cs.expected_count,
            })
            if index < 50:  # keep the table bounded
                table.add_row(
                    str(index),
                    f"{cs.precision:.3f}",
                    f"{cs.recall:.3f}",
                    f"{cs.f1:.3f}",
                )

        if not scored:
            console.print(
                "[red]No scorable rows[/] — need {predicted, expected_ids} "
                "or RAFT {query, golden_doc, answer}."
            )
            raise typer.Exit(2)

        n = len(scored)
        aggregate = {
            "precision": prec_sum / n,
            "recall": rec_sum / n,
            "f1": f1_sum / n,
        }
        console.print(table)
        console.print(
            Panel(
                f"Rows scored: [bold]{n}[/]\n"
                f"Style: [bold]{escape(canonical_style)}[/]\n"
                f"Mean precision: [bold]{aggregate['precision']:.3f}[/]\n"
                f"Mean recall:    [bold]{aggregate['recall']:.3f}[/]\n"
                f"Mean F1:        [bold]{aggregate['f1']:.3f}[/]",
                title="Citation aggregate",
                border_style="green",
            )
        )

        if output is not None:
            from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

            try:
                enforce_under_cwd_and_no_symlink(output, "citation output")
            except (TypeError, ValueError) as exc:
                console.print(f"[red]Invalid --output:[/] {escape(str(exc))}")
                raise typer.Exit(2) from exc
            payload = {
                "style": canonical_style,
                "n_rows": n,
                "aggregate": aggregate,
                "rows": scored,
            }
            with open(output, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, allow_nan=False)
            console.print(f"Wrote citation report -> {escape(output)}")
