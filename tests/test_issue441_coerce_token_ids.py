"""Issue #441 — ``coerce_token_ids`` is the shared public helper, and it
accepts duck-typed mappings.

Two follow-ups from #430, neither in scope for that PR:

1. ``_coerce_token_ids`` was private-named but imported by ``data_doctor``.
   A leading underscore is this repo's marker for "private to the module";
   the import graph said otherwise. The helper *is* the anti-drift
   mechanism, so it is spelled as one: ``coerce_token_ids``.

2. The Mapping gate missed dict-like objects that are not registered as
   ``collections.abc.Mapping``. Those flowed into ``_coerce_int_list``,
   which iterated *keys* and raised ``input_ids[0]='input_ids'``. The
   mask path (``_apply_template_with_mask``) had the same gate and
   silently fell back, dropping ``assistant_masks``. Both sites now use
   one ``_is_mapping_like`` predicate (``Mapping`` or ``hasattr(..., "get")``),
   which is @emre155's shape from #438.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

from soup_cli.data.loss_mask import IGNORE_INDEX, coerce_token_ids


class _DuckMapping:
    """Dict-like, not registered as ``collections.abc.Mapping``.

    The object measured on ``main`` in #441: ``get`` / ``__getitem__`` /
    ``keys`` / ``__iter__`` holding ``{"input_ids": [...], ...}``.
    """

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def get(self, key, default=None):
        return self._payload.get(key, default)

    def __getitem__(self, key):
        return self._payload[key]

    def keys(self):
        return self._payload.keys()

    def __iter__(self):
        return iter(self._payload)


_USER_ASSISTANT = [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "A"},
]


def test_duck_mapping_is_not_a_collections_mapping():
    """Control: if this object IS a Mapping, the rest of the suite cannot
    see the bug #441 describes."""
    assert not isinstance(_DuckMapping({"input_ids": [1, 2, 3]}), Mapping)


# ---------------------------------------------------------------------------
# Part 1 — public name, one copy, no cross-module private import
# ---------------------------------------------------------------------------


class TestPublicSharedHelper:
    def test_helper_is_public(self):
        assert coerce_token_ids.__name__ == "coerce_token_ids"
        assert not coerce_token_ids.__name__.startswith("_")

    def test_data_doctor_imports_the_loss_mask_helper_not_a_copy(self):
        """Fails if data_doctor grows its own coerce, or goes back to
        importing the underscore-prefixed name — the drift the shared
        helper exists to prevent."""
        import soup_cli.utils.data_doctor as doctor

        tree = ast.parse(Path(doctor.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        defined: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "soup_cli.data.loss_mask"
            ):
                for alias in node.names:
                    imported.add(alias.name)
            if isinstance(node, ast.FunctionDef) and node.name in {
                "coerce_token_ids",
                "_coerce_token_ids",
            }:
                defined.add(node.name)
        assert "coerce_token_ids" in imported, (
            "data_doctor must import coerce_token_ids from loss_mask; "
            f"loss_mask names imported: {sorted(imported)}"
        )
        assert "_coerce_token_ids" not in imported, (
            "data_doctor must not import another module's private name"
        )
        assert not defined, f"data_doctor grew its own copy of the helper: {defined}"

    def test_no_module_imports_loss_mask_private_coerce(self):
        src_root = Path(__file__).resolve().parents[1] / "src" / "soup_cli"
        offenders: list[str] = []
        for path in src_root.rglob("*.py"):
            if path.name == "loss_mask.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "from soup_cli.data.loss_mask import" in text and "_coerce_token_ids" in text:
                offenders.append(str(path.relative_to(src_root.parent.parent)))
        assert offenders == [], (
            "underscore-prefixed coerce imported across modules: " + ", ".join(offenders)
        )


# ---------------------------------------------------------------------------
# Part 2 — duck-typed mapping: ids, mask path, missing-key error
# ---------------------------------------------------------------------------


class TestDuckTypedMapping:
    def test_duck_mapping_returns_input_ids_not_keys(self):
        """Reproduces the measured failure: pre-fix this raised
        ``input_ids[0]='input_ids'`` because ``list(duck)`` is the keys."""
        out = _DuckMapping(
            {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}
        )
        assert coerce_token_ids(out) == [1, 2, 3]

    def test_duck_mapping_with_assistant_masks_takes_the_mask_path(self):
        """Fails if only ``coerce_token_ids`` is widened and
        ``_apply_template_with_mask`` still returns None: the fallback
        unmasks the whole assistant span (two tokens here), the mask path
        keeps only the last token."""
        from soup_cli.data.loss_mask import build_assistant_only_labels

        class _DuckMaskTokenizer:
            chat_template = "x"

            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("return_assistant_tokens_mask"):
                    return _DuckMapping(
                        {
                            "input_ids": [10, 20, 30, 40],
                            "assistant_masks": [0, 0, 0, 1],
                        }
                    )
                if len(messages) == 1:
                    return [10, 20]
                return [10, 20, 30, 40]

        got = build_assistant_only_labels(_USER_ASSISTANT, _DuckMaskTokenizer())
        assert got["input_ids"] == [10, 20, 30, 40]
        assert got["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 40]

    def test_get_without_input_ids_raises_the_existing_error(self):
        out = _DuckMapping({"assistant_masks": [0, 1, 1]})
        with pytest.raises(ValueError, match="no 'input_ids'"):
            coerce_token_ids(out)

    def test_non_mapping_non_sequence_still_fails_loudly(self):
        with pytest.raises(ValueError, match="invalid input_ids"):
            coerce_token_ids(object())

    def test_plain_sequence_and_dict_are_unchanged(self):
        assert coerce_token_ids([7, 8, 9]) == [7, 8, 9]
        assert coerce_token_ids({"input_ids": [7, 8, 9]}) == [7, 8, 9]


class TestOneMappingPredicate:
    def test_both_call_sites_share_the_mapping_predicate(self):
        """Reverting the one-line widen in only one site would pass a
        unit test of the other. Pinning that both sites call the shared
        predicate — and that ``isinstance(out, Mapping)`` lives in exactly
        one place — is the mutation #441 asks for."""
        import soup_cli.data.loss_mask as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert src.count("isinstance(out, Mapping)") == 1
        assert src.count("_is_mapping_like(") >= 2
        assert "def _is_mapping_like(" in src
        tree = ast.parse(src)
        calls_in: dict[str, int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            count = sum(
                1
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_is_mapping_like"
            )
            if count:
                calls_in[node.name] = count
        assert calls_in.get("coerce_token_ids", 0) >= 1
        assert calls_in.get("_apply_template_with_mask", 0) >= 1
