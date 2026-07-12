"""v0.71.33 — `soup draft`: train-your-own speculative-decoding draft.

Covers:
* ``utils/adapter_fuse.py`` — the shared LoRA -> dense fuse extracted from
  ``commands/shrink.py`` (v0.71.29) so ``shrink`` and ``draft`` share one path.
* ``utils/draft.py`` — pure acceptance kernel + tokenizer-equality guard +
  local draft registry + torch-lazy measurement.
* ``utils/spec_pairing.py`` — local-registry lookup before the static map.
* ``commands/draft.py`` — ``soup draft distill / measure / list``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Task 1 — shared adapter fuse
# ---------------------------------------------------------------------------
class TestAdapterFuse:
    def test_module_exports_fuse_and_release(self):
        from soup_cli.utils.adapter_fuse import fuse_adapter_into, release_cuda

        assert callable(fuse_adapter_into)
        assert callable(release_cuda)

    def test_shrink_reuses_the_shared_implementation(self):
        """shrink must not keep a second copy of the fuse (no behavioural drift)."""
        from soup_cli.commands import shrink
        from soup_cli.utils.adapter_fuse import fuse_adapter_into, release_cuda

        assert shrink._fuse_adapter is fuse_adapter_into
        assert shrink._release_cuda is release_cuda

    def test_no_top_level_torch(self):
        import soup_cli

        path = Path(soup_cli.__file__).parent / "utils" / "adapter_fuse.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        ("torch", "transformers", "peft")
                    ), f"top-level import of {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    ("torch", "transformers", "peft")
                ), f"top-level import from {node.module}"

    def test_fuse_revalidates_base_dir_before_swapping(self, tmp_path, monkeypatch):
        """The subprocess ran for hours — the swap target is re-checked."""
        from soup_cli.utils import adapter_fuse

        seen: list[tuple[str, str]] = []

        def _fake_enforce(path: str, field: str) -> str:
            seen.append((path, field))
            raise ValueError("refused")

        monkeypatch.setattr(adapter_fuse, "enforce_under_cwd_and_no_symlink", _fake_enforce)
        with pytest.raises(ValueError, match="refused"):
            adapter_fuse.fuse_adapter_into(
                base_dir=str(tmp_path / "base"), adapter_dir=str(tmp_path / "ad")
            )
        # Refused BEFORE any model load.
        assert seen and seen[0][0] == str(tmp_path / "base")


def _pytest_marker() -> None:  # pragma: no cover - import sanity
    return None
