#!/usr/bin/env python3
"""Regenerate tests/fixtures/recipe_config_snapshots.json (#621, #638).

Deliberately a separate, explicit step rather than auto-updating on a failed
test: run this and review the diff, don't silently regenerate.

#638: the fixture no longer stores each recipe's full resolved config (162
copies of the same ~296 schema defaults). It stores the shared defaults
ONCE as "baseline", and per recipe only the fields that differ from that
baseline. A schema-default change then moves the one committed baseline
value, not 149-odd identical copies of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "recipe_config_snapshots.json"

# Minimal config: only the fields every SoupConfig requires (base, task,
# data.train, data.format) are set, to placeholder values no real recipe
# shares. Everything else resolves to its schema default. Those four
# placeholder-driven fields end up in EVERY recipe's delta automatically —
# no special-casing needed to keep genuinely-per-recipe fields out of the
# shared baseline, since no real recipe's value ever matches a placeholder.
_BASELINE_YAML = (
    "base: __BASELINE_PLACEHOLDER_MODEL__\n"
    "task: sft\n"
    "data: {train: __baseline_placeholder__.jsonl, format: chatml}\n"
)

_MISSING = object()


def _ensure_src_on_path() -> None:
    """Make ``soup_cli`` importable, without piling up duplicate entries.

    Called from every function that needs the import — each is usable on
    its own — so membership is checked rather than inserting unconditionally.
    """
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a nested dict to ``{"training.lora.r": 16, ...}``."""
    out: dict[str, object] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def delta(resolved: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    """Fields in ``resolved`` whose value differs from ``baseline``.

    Public: this is part of the contract between the generator and the test
    that verifies the fixture, not an internal of either.

    A field present in ``resolved`` but absent from ``baseline`` (a nested
    object a recipe fills in that the baseline leaves ``None``) counts as
    differing — compared against a sentinel, never a value a real config
    could hold.
    """
    return {
        path: value
        for path, value in resolved.items()
        if value != baseline.get(path, _MISSING)
    }


def build_baseline() -> dict[str, object]:
    """The shared schema-default baseline every recipe's delta is relative to."""
    _ensure_src_on_path()
    from soup_cli.config.loader import load_config_from_string

    return flatten(load_config_from_string(_BASELINE_YAML).model_dump(mode="json"))


def build_snapshot() -> dict[str, dict]:
    _ensure_src_on_path()
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.recipes.catalog import RECIPES

    baseline = build_baseline()
    recipes = {
        name: delta(
            flatten(load_config_from_string(recipe.yaml_str).model_dump(mode="json")),
            baseline,
        )
        for name, recipe in sorted(RECIPES.items())
    }
    return {"baseline": baseline, "recipes": recipes}


def main() -> int:
    snapshot = build_snapshot()
    _FIXTURE_PATH.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote baseline ({len(snapshot['baseline'])} fields) and "
        f"{len(snapshot['recipes'])} recipe deltas to {_FIXTURE_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
