#!/usr/bin/env python3
"""Regenerate tests/fixtures/recipe_config_snapshots.json (#621).

Deliberately a separate, explicit step rather than auto-updating on a failed
test: run this and review the diff, don't silently regenerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "recipe_config_snapshots.json"


def build_snapshot() -> dict[str, object]:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.recipes.catalog import RECIPES

    return {
        name: load_config_from_string(recipe.yaml_str).model_dump(mode="json")
        for name, recipe in sorted(RECIPES.items())
    }


def main() -> int:
    snapshot = build_snapshot()
    _FIXTURE_PATH.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(snapshot)} recipe snapshots to {_FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
