"""v0.40.1 Part E — Recipe id fuzzy-match + UX papercuts.

Closes:
  - M3: `soup recipes show <typo>` suggests close matches via difflib
  - JSONL BOM auto-strip (Windows PowerShell users)
  - `soup data sample` filename includes strategy to prevent overwrite
"""

from __future__ import annotations

import json
from pathlib import Path

# --- M3: recipe fuzzy-match -----------------------------------------------


def test_suggest_recipes_returns_close_matches():
    from soup_cli.commands.recipes import _suggest_recipes
    from soup_cli.recipes.catalog import RECIPES

    if not RECIPES:
        return
    real_name = next(iter(RECIPES.keys()))
    # Mutate one char in the middle to simulate a typo.
    if len(real_name) >= 4:
        typo = real_name[:2] + "x" + real_name[3:]
    else:
        typo = real_name + "x"
    suggestions = _suggest_recipes(typo)
    assert real_name in suggestions or any(
        s in real_name or real_name in s for s in suggestions
    )


def test_suggest_recipes_empty_for_garbage():
    from soup_cli.commands.recipes import _suggest_recipes

    # Wholly unrelated query should return [] (cutoff=0.6).
    suggestions = _suggest_recipes("zzqqxxyyy_no_match_at_all")
    assert suggestions == []


# --- BOM auto-strip in JSONL loader ---------------------------------------


def test_jsonl_loader_strips_utf8_bom(tmp_path: Path):
    from soup_cli.data.loader import _load_jsonl

    f = tmp_path / "with_bom.jsonl"
    # Write BOM + valid JSONL via binary mode so we control the bytes.
    f.write_bytes(
        b"\xef\xbb\xbf"  # UTF-8 BOM
        + json.dumps({"prompt": "hi"}).encode("utf-8")
        + b"\n"
        + json.dumps({"prompt": "world"}).encode("utf-8")
        + b"\n"
    )
    rows = _load_jsonl(f)
    assert len(rows) == 2
    assert rows[0]["prompt"] == "hi"


def test_jsonl_loader_no_bom_still_works(tmp_path: Path):
    from soup_cli.data.loader import _load_jsonl

    f = tmp_path / "no_bom.jsonl"
    f.write_text('{"prompt": "alpha"}\n{"prompt": "beta"}\n', encoding="utf-8")
    rows = _load_jsonl(f)
    assert [r["prompt"] for r in rows] == ["alpha", "beta"]


# --- data sample default filename includes strategy -----------------------


def test_data_sample_default_filename_includes_strategy():
    """Source-level invariant: default ``out_path`` template names the strategy."""
    import inspect

    from soup_cli.commands.data import sample_data

    src = inspect.getsource(sample_data)
    assert '_sampled_{strategy}.jsonl' in src, (
        "default sampled filename must embed the strategy to prevent overwrite"
    )
