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
import os
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


# ---------------------------------------------------------------------------
# Task 2 — pure acceptance kernel
# ---------------------------------------------------------------------------
class _FakeTok:
    """Duck-typed tokenizer: a fixed text -> ids table plus a vocab_size."""

    def __init__(self, vocab_size: int, table: dict[str, list[int]] | None = None):
        self.vocab_size = vocab_size
        self._table = table or {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self._table:
            return list(self._table[text])
        # Deterministic default: codepoint sum, so two tokenizers with the same
        # table agree everywhere.
        return [sum(ord(ch) for ch in text) % max(self.vocab_size, 1)]


class TestComputeAcceptance:
    def test_all_match(self):
        from soup_cli.utils.draft import compute_acceptance

        assert compute_acceptance([1, 2, 3], [1, 2, 3]) == 1.0

    def test_none_match(self):
        from soup_cli.utils.draft import compute_acceptance

        assert compute_acceptance([9, 9, 9], [1, 2, 3]) == 0.0

    def test_partial(self):
        from soup_cli.utils.draft import compute_acceptance

        assert compute_acceptance([1, 2, 9], [1, 2, 3]) == pytest.approx(2 / 3)

    def test_empty_is_zero(self):
        from soup_cli.utils.draft import compute_acceptance

        assert compute_acceptance([], []) == 0.0

    def test_length_mismatch_raises(self):
        from soup_cli.utils.draft import compute_acceptance

        with pytest.raises(ValueError, match="same length"):
            compute_acceptance([1, 2], [1])


class TestClassify:
    def test_boundary_exact(self):
        from soup_cli.utils.draft import classify_acceptance

        assert classify_acceptance(0.70) == "STRONG"
        assert classify_acceptance(0.6999) == "MODERATE"
        assert classify_acceptance(0.50) == "MODERATE"
        assert classify_acceptance(0.4999) == "WEAK"
        assert classify_acceptance(0.0) == "WEAK"
        assert classify_acceptance(1.0) == "STRONG"

    def test_rejects_bool(self):
        from soup_cli.utils.draft import classify_acceptance

        with pytest.raises(TypeError, match="bool"):
            classify_acceptance(True)

    def test_rejects_nonfinite(self):
        from soup_cli.utils.draft import classify_acceptance

        with pytest.raises(ValueError, match="finite"):
            classify_acceptance(float("nan"))

    def test_rejects_out_of_range(self):
        from soup_cli.utils.draft import classify_acceptance

        with pytest.raises(ValueError, match="between 0 and 1"):
            classify_acceptance(1.5)


class TestSameTokenizer:
    def test_identical_is_true(self):
        from soup_cli.utils.draft import same_tokenizer

        tok = _FakeTok(32000)
        assert same_tokenizer(tok, tok) is True
        assert same_tokenizer(_FakeTok(32000), _FakeTok(32000)) is True

    def test_equal_vocab_size_but_different_ids_is_false(self):
        """The test that makes the guard meaningful.

        Two tokenizers can both report vocab_size 32000 and still disagree on
        every single token — a vocab_size check alone would wave that through
        and the draft's proposals would be pure noise.
        """
        from soup_cli.utils.draft import same_tokenizer

        probe = "Hello, world!"
        a = _FakeTok(32000, {probe: [1, 2, 3]})
        b = _FakeTok(32000, {probe: [7, 8, 9]})
        assert same_tokenizer(a, b) is False

    def test_different_vocab_size_is_false(self):
        from soup_cli.utils.draft import same_tokenizer

        assert same_tokenizer(_FakeTok(32000), _FakeTok(49152)) is False

    def test_probe_corpus_is_non_trivial(self):
        """A single-ASCII-word probe would miss most real tokenizer splits."""
        from soup_cli.utils.draft import PROBE_CORPUS

        assert len(PROBE_CORPUS) >= 4
        joined = "".join(PROBE_CORPUS)
        assert any(ord(ch) > 127 for ch in joined), "probe must include non-ASCII"
        assert any(ch.isdigit() for ch in joined), "probe must include digits"

    def test_broken_tokenizer_encode_is_false_not_raise(self):
        from soup_cli.utils.draft import same_tokenizer

        class _Broken:
            vocab_size = 32000

            def encode(self, text, add_special_tokens=False):
                raise RuntimeError("boom")

        assert same_tokenizer(_FakeTok(32000), _Broken()) is False


class TestReport:
    def _report(self, **kw):
        from soup_cli.utils.draft import AcceptanceReport

        defaults = dict(
            target="t",
            draft="d",
            n_prompts=2,
            n_generated_tokens=100,
            acceptance_rate=0.75,
            verdict="STRONG",
            tok_s_plain=10.0,
            tok_s_assisted=12.0,
            speedup=1.2,
            num_assistant_tokens=5,
            soup_version="0.71.33",
        )
        defaults.update(kw)
        return AcceptanceReport(**defaults)

    def test_frozen(self):
        import dataclasses

        report = self._report()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.acceptance_rate = 0.1  # type: ignore[misc]

    def test_to_dict_round_trip(self):
        from soup_cli.utils.draft import draft_report_to_dict

        data = draft_report_to_dict(self._report())
        assert data["acceptance_rate"] == 0.75
        assert data["verdict"] == "STRONG"
        assert data["target"] == "t"
        assert data["speedup"] == 1.2

    def test_panel_names_the_verdict_and_rate(self):
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.draft import render_draft_panel

        buf = StringIO()
        Console(file=buf, width=100).print(render_draft_panel(self._report()))
        out = buf.getvalue()
        assert "STRONG" in out
        assert "75" in out  # the acceptance rate as a percentage

    def test_panel_renders_when_throughput_unmeasured(self):
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.draft import render_draft_panel

        buf = StringIO()
        report = self._report(tok_s_plain=None, tok_s_assisted=None, speedup=None)
        Console(file=buf, width=100).print(render_draft_panel(report))
        assert "STRONG" in buf.getvalue()


# ---------------------------------------------------------------------------
# Task 3 — local draft registry + spec_pairing lookup
# ---------------------------------------------------------------------------
@pytest.fixture()
def draft_registry(tmp_path, monkeypatch):
    """Point the draft registry at a temp file (never touch the real ~/.soup)."""
    path = tmp_path / "drafts.json"
    monkeypatch.setenv("SOUP_DRAFT_REGISTRY_PATH", str(path))
    return path


class TestDraftRegistry:
    def test_round_trip(self, draft_registry, tmp_path):
        from soup_cli.utils.draft import list_drafts, lookup_draft, register_draft

        draft = tmp_path / "mydraft"
        draft.mkdir()
        register_draft("HF/Target-7B", str(draft), 0.62)

        assert lookup_draft("HF/Target-7B") == os.path.realpath(str(draft))
        entries = list_drafts()
        assert len(entries) == 1
        assert entries[0]["acceptance_rate"] == 0.62
        assert entries[0]["target"] == "hf/target-7b"

    def test_lookup_is_case_insensitive(self, draft_registry, tmp_path):
        """Must match pick_draft_model's .lower() normalisation."""
        from soup_cli.utils.draft import lookup_draft, register_draft

        draft = tmp_path / "d"
        draft.mkdir()
        register_draft("Qwen/Qwen2.5-72B", str(draft))
        assert lookup_draft("qwen/qwen2.5-72b") is not None
        assert lookup_draft("QWEN/QWEN2.5-72B") is not None

    def test_stale_dir_is_skipped(self, draft_registry, tmp_path):
        """A moved/deleted draft degrades to 'no draft', never a crash in serve."""
        import shutil

        from soup_cli.utils.draft import lookup_draft, register_draft

        draft = tmp_path / "gone"
        draft.mkdir()
        register_draft("hf/target", str(draft))
        shutil.rmtree(draft)
        assert lookup_draft("hf/target") is None

    def test_reregister_replaces(self, draft_registry, tmp_path):
        from soup_cli.utils.draft import list_drafts, lookup_draft, register_draft

        first = tmp_path / "one"
        first.mkdir()
        second = tmp_path / "two"
        second.mkdir()
        register_draft("hf/t", str(first))
        register_draft("hf/t", str(second))
        assert len(list_drafts()) == 1
        assert lookup_draft("hf/t") == os.path.realpath(str(second))

    def test_malformed_json_reads_as_empty(self, draft_registry):
        from soup_cli.utils.draft import list_drafts, lookup_draft

        draft_registry.write_text("{ not json", encoding="utf-8")
        assert list_drafts() == []
        assert lookup_draft("anything") is None

    def test_non_dict_payload_reads_as_empty(self, draft_registry):
        from soup_cli.utils.draft import list_drafts

        draft_registry.write_text('["nope"]', encoding="utf-8")
        assert list_drafts() == []

    def test_missing_file_reads_as_empty(self, draft_registry):
        from soup_cli.utils.draft import list_drafts, lookup_draft

        assert not draft_registry.exists()
        assert list_drafts() == []
        assert lookup_draft("x") is None

    def test_entry_cap(self, draft_registry, tmp_path):
        from soup_cli.utils.draft import _MAX_REGISTRY_ENTRIES, list_drafts, register_draft

        draft = tmp_path / "d"
        draft.mkdir()
        for i in range(_MAX_REGISTRY_ENTRIES + 5):
            register_draft(f"hf/target-{i}", str(draft))
        assert len(list_drafts()) == _MAX_REGISTRY_ENTRIES

    def test_atomic_write_leaves_no_temp_file(self, draft_registry, tmp_path):
        from soup_cli.utils.draft import register_draft

        draft = tmp_path / "d"
        draft.mkdir()
        register_draft("hf/t", str(draft))
        leftovers = [p.name for p in draft_registry.parent.glob(".soup.*")]
        assert leftovers == []

    def test_rejects_empty_target_and_bad_rate(self, draft_registry, tmp_path):
        from soup_cli.utils.draft import register_draft

        draft = tmp_path / "d"
        draft.mkdir()
        with pytest.raises(ValueError, match="non-empty"):
            register_draft("  ", str(draft))
        with pytest.raises(ValueError, match="between 0 and 1"):
            register_draft("hf/t", str(draft), 1.5)


class TestSpecPairingLocal:
    def test_local_registry_beats_static_map(self, draft_registry, tmp_path):
        """A draft you trained yourself wins over the built-in pairing."""
        from soup_cli.utils.draft import register_draft
        from soup_cli.utils.spec_pairing import pick_draft_model

        draft = tmp_path / "mydraft"
        draft.mkdir()
        register_draft("qwen/qwen2.5-72b", str(draft))  # also in the static map
        assert pick_draft_model("Qwen/Qwen2.5-72B") == os.path.realpath(str(draft))

    def test_falls_back_to_static_map(self, draft_registry):
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("Qwen/Qwen2.5-72B") == "Qwen/Qwen2.5-0.5B"

    def test_unknown_target_still_none(self, draft_registry):
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("nobody/nothing-1b") is None

    def test_url_target_still_rejected(self, draft_registry):
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("https://evil.example/model") is None

    def test_registry_blowup_does_not_crash_serve(self, draft_registry, monkeypatch):
        """pick_draft_model is on serve's startup path — it must never raise."""
        from soup_cli.utils import spec_pairing

        def _boom(target):
            raise RuntimeError("registry on fire")

        monkeypatch.setattr(spec_pairing, "_lookup_local_draft", _boom)
        assert spec_pairing.pick_draft_model("Qwen/Qwen2.5-72B") == "Qwen/Qwen2.5-0.5B"


class TestDraftNoTopLevelTorch:
    def test_utils_draft_is_torch_free(self):
        import soup_cli

        path = Path(soup_cli.__file__).parent / "utils" / "draft.py"
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
