"""v0.53.6 #106 — Data Recipe DAG runner (stub-then-live).

Schema-only this release. Per-node-kind handlers (seed / llm_text / code /
judge / validator / sampler), checkpoint-between-nodes, resume-on-failure
+ ``soup data recipe --execute`` wire-up land in v0.53.7. Mirrors the
project stub-then-live pattern (v0.27.0 MII / v0.37.0 multipack /
v0.50.0 GRPO Plus).

The validator surface ships now so callers can target the schema and
type-check against ``run_recipe`` before live execution exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover — type-only import.
    from soup_cli.utils.recipe_dag import RecipeDAG


def run_recipe(
    dag: "RecipeDAG",
    *,
    output_dir: str,
    checkpoint_dir: str | None = None,
    resume: bool = False,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> Mapping[str, Any]:
    """Execute a validated :class:`RecipeDAG` end-to-end.

    Deferred-live stub. Per-node-kind handlers + checkpoint/resume +
    ``--execute`` plumbing land in v0.53.7. Schema-only contract:

    - ``dag``: a :class:`soup_cli.utils.recipe_dag.RecipeDAG` (validated)
    - ``output_dir``: cwd-contained directory for sampler output JSONL
    - ``checkpoint_dir``: optional intermediate-node checkpoint dir
    - ``resume``: if True, skip nodes whose checkpoint already exists
    - ``judge_provider`` / ``judge_model``: routed into the v0.40.3
      ``JudgeEvaluator`` for ``judge`` nodes

    Raises:
        TypeError: if ``dag`` is not a ``RecipeDAG``.
        NotImplementedError: always — live runner ships in v0.53.7.
    """
    # Late import so test code can patch the module without forcing a
    # heavy recipe_dag import at module load.
    from soup_cli.utils.recipe_dag import RecipeDAG

    if not isinstance(dag, RecipeDAG):
        raise TypeError("dag must be a RecipeDAG")
    if not isinstance(output_dir, str) or not output_dir:
        raise TypeError("output_dir must be a non-empty string")
    if checkpoint_dir is not None and not isinstance(checkpoint_dir, str):
        raise TypeError("checkpoint_dir must be a string or None")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if judge_provider is not None and not isinstance(judge_provider, str):
        raise TypeError("judge_provider must be a string or None")
    if judge_model is not None and not isinstance(judge_model, str):
        raise TypeError("judge_model must be a string or None")
    raise NotImplementedError(
        "Data Recipe DAG runner is deferred to v0.53.7. The schema + "
        "validator surface in soup_cli.utils.recipe_dag is live; only the "
        "per-node execution loop is missing."
    )


__all__ = ["run_recipe"]
