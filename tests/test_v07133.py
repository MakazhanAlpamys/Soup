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

    def test_failed_save_does_not_orphan_the_staging_dir(self, tmp_path, monkeypatch):
        """python-review HIGH: a save that raises mid-write used to leave a
        full model's worth of bytes in a .fuse_* dir next to the model."""
        import sys
        import types

        from soup_cli.utils import adapter_fuse

        base = tmp_path / "base"
        base.mkdir()
        (base / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        class _Merged:
            def save_pretrained(self, path):
                # The staging dir already exists at this point — this is exactly
                # where a disk-full / interrupted write lands.
                Path(path).joinpath("partial.bin").write_bytes(b"xxxx")
                raise RuntimeError("disk full")

        class _Base:
            @staticmethod
            def from_pretrained(*a, **kw):
                return _Base()

        class _Peft:
            @staticmethod
            def from_pretrained(*a, **kw):
                obj = _Peft()
                obj.merge_and_unload = lambda: _Merged()  # noqa: E731
                return obj

        class _Tok:
            @staticmethod
            def from_pretrained(*a, **kw):
                return _Tok()

            def save_pretrained(self, path):
                pass

        fake_tf = types.ModuleType("transformers")
        fake_tf.AutoModelForCausalLM = _Base
        fake_tf.AutoTokenizer = _Tok
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = _Peft
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)

        with pytest.raises(RuntimeError, match="disk full"):
            adapter_fuse.fuse_adapter_into(
                base_dir="base", adapter_dir=str(tmp_path / "ad")
            )
        assert list(tmp_path.glob(".fuse_*")) == [], "staging dir was orphaned"
        assert base.exists(), "the base model must survive a failed fuse"

    def test_fuse_adapter_into_is_the_in_place_special_case(self, monkeypatch):
        """shrink's in-place fuse == merge(base=out=base_dir)."""
        from soup_cli.utils import adapter_fuse

        seen: dict = {}
        monkeypatch.setattr(
            adapter_fuse,
            "merge_adapter_to_dense",
            lambda **kw: seen.update(kw),
        )
        adapter_fuse.fuse_adapter_into(base_dir="model", adapter_dir="ad", trc=True)
        assert seen["base_model"] == "model"
        assert seen["out_dir"] == "model"
        assert seen["adapter_dir"] == "ad"
        assert seen["trc"] is True

    def test_merge_rejects_pickle_weights_in_adapter_dir(self, tmp_path, monkeypatch):
        """security CRITICAL: the adapter dir may be attacker-produced; refuse
        pickle weights before PEFT torch.load's them."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "out" / "_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.bin").write_bytes(b"\x80\x04pickle")  # unsafe ext

        with pytest.raises(ValueError, match="(?i)unsafe|pickle|safetensors"):
            adapter_fuse.merge_adapter_to_dense(
                base_model="org/tiny", adapter_dir=str(adapter), out_dir="out"
            )

    def test_merge_loads_base_weights_from_the_base_model_not_out_dir(
        self, tmp_path, monkeypatch
    ):
        """The whole point of the CRITICAL fix: an adapter-only dir has no base
        weights, so the base must be loaded from a separate model id/path."""
        import sys
        import types

        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "out" / "_adapter"
        adapter.mkdir(parents=True)
        loaded: dict = {}

        class _Merged:
            def save_pretrained(self, path):
                Path(path).joinpath("model.safetensors").write_bytes(b"dense")

        class _Base:
            @staticmethod
            def from_pretrained(model_id, **kw):
                loaded["base"] = model_id
                return _Base()

        class _Peft:
            @staticmethod
            def from_pretrained(base, adapter_dir, **kw):
                loaded["adapter"] = adapter_dir
                obj = _Peft()
                obj.merge_and_unload = lambda: _Merged()  # noqa: E731
                return obj

        class _Tok:
            @staticmethod
            def from_pretrained(model_id, **kw):
                return _Tok()

            def save_pretrained(self, path):
                Path(path).joinpath("tokenizer.json").write_text("{}", encoding="utf-8")

        fake_tf = types.ModuleType("transformers")
        fake_tf.AutoModelForCausalLM = _Base
        fake_tf.AutoTokenizer = _Tok
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = _Peft
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)

        adapter_fuse.merge_adapter_to_dense(
            base_model="org/tiny", adapter_dir=str(adapter), out_dir="out"
        )
        assert loaded["base"] == "org/tiny"
        assert loaded["adapter"] == str(adapter)
        # out/ is now the DENSE model, and the adapter subdir is gone.
        assert (tmp_path / "out" / "model.safetensors").exists()
        assert not (tmp_path / "out" / "_adapter").exists()
        assert list(tmp_path.glob(".fuse_*")) == []

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


class TestCountAcceptedAndRate:
    def test_count_accepted(self):
        from soup_cli.utils.draft import count_accepted

        assert count_accepted([1, 2, 9], [1, 2, 3]) == 2
        assert count_accepted([], []) == 0

    def test_count_accepted_length_mismatch_raises(self):
        from soup_cli.utils.draft import count_accepted

        with pytest.raises(ValueError, match="same length"):
            count_accepted([1], [1, 2])

    def test_acceptance_rate(self):
        from soup_cli.utils.draft import acceptance_rate

        assert acceptance_rate(75, 100) == 0.75
        assert acceptance_rate(0, 0) == 0.0

    def test_acceptance_rate_rejects_impossible_counts(self):
        from soup_cli.utils.draft import acceptance_rate

        with pytest.raises(ValueError, match="exceeds"):
            acceptance_rate(5, 3)
        with pytest.raises(ValueError, match="non-negative"):
            acceptance_rate(-1, 3)


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

    def test_concurrent_registration_does_not_lose_an_entry(
        self, draft_registry, tmp_path
    ):
        """code-review MEDIUM: read-modify-write under a cross-process lock."""
        import threading

        from soup_cli.utils.draft import list_drafts, register_draft

        draft = tmp_path / "d"
        draft.mkdir()
        errors: list[BaseException] = []

        def _register(idx: int) -> None:
            try:
                register_draft(f"hf/target-{idx}", str(draft))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_register, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        targets = {entry["target"] for entry in list_drafts()}
        assert targets == {f"hf/target-{i}" for i in range(12)}

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


# ---------------------------------------------------------------------------
# Task 4 — torch-lazy measurement
# ---------------------------------------------------------------------------
class _TensorTok:
    """Minimal tokenizer over torch tensors."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, prompt_ids: list[int]):
        self._prompt_ids = prompt_ids

    def __call__(self, text, return_tensors=None):
        import torch

        ids = torch.tensor([self._prompt_ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _FakeTarget:
    """Greedy-generates a fixed continuation."""

    def __init__(self, full_ids: list[int], n_prompt: int):
        self._full_ids = full_ids
        self._n_prompt = n_prompt
        self.calls: list[dict] = []

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def generate(self, **kwargs):
        import torch

        self.calls.append(kwargs)
        return torch.tensor([self._full_ids])


class _FakeDraft:
    """Returns logits whose per-position argmax is a scripted sequence."""

    def __init__(self, argmax_per_position: list[int], vocab: int = 100):
        self._argmax = argmax_per_position
        self._vocab = vocab

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, input_ids=None, **kwargs):
        import torch

        seq = int(input_ids.shape[1])
        logits = torch.zeros(1, seq, self._vocab)
        for pos in range(seq):
            logits[0, pos, self._argmax[pos]] = 10.0

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


class TestMeasureAcceptance:
    # prompt = [5, 6] (len 2); target generates [10, 11, 12] at positions 2,3,4.
    FULL_IDS = [5, 6, 10, 11, 12]
    N_PROMPT = 2
    # Draft argmax per position. Causal alignment: logits[p-1] predicts token p,
    # so the proposals for the generated positions 2,3,4 are ARGMAX[1:4].
    #   correct     -> ARGMAX[1:4] = [10, 99, 12]  vs [10, 11, 12] -> 2 matches
    #   off-by-one+ -> ARGMAX[2:5] = [99, 12, 99]  vs [10, 11, 12] -> 0 matches
    #   off-by-one- -> ARGMAX[0:3] = [99, 10, 99]  vs [10, 11, 12] -> 0 matches
    # The three alignments give DIFFERENT answers, so this test can actually
    # fail on an off-by-one (an unfalsifiable test is worse than none).
    ARGMAX = [99, 10, 99, 12, 99]

    def test_shift_convention_is_pinned(self):
        from soup_cli.utils.draft import measure_acceptance

        accepted, total = measure_acceptance(
            _FakeTarget(self.FULL_IDS, self.N_PROMPT),
            _FakeDraft(self.ARGMAX),
            _TensorTok([5, 6]),
            ["hello"],
            max_new_tokens=8,
        )
        assert (accepted, total) == (2, 3)

    def test_off_by_one_alternatives_would_score_differently(self):
        """Guards the guard: the fixture must discriminate, not just pass."""
        from soup_cli.utils.draft import compute_acceptance

        generated = [10, 11, 12]
        correct = self.ARGMAX[1:4]
        off_plus = self.ARGMAX[2:5]
        off_minus = self.ARGMAX[0:3]
        assert compute_acceptance(correct, generated) == pytest.approx(2 / 3)
        assert compute_acceptance(off_plus, generated) == 0.0
        assert compute_acceptance(off_minus, generated) == 0.0

    def test_multiple_prompts_accumulate(self):
        from soup_cli.utils.draft import measure_acceptance

        accepted, total = measure_acceptance(
            _FakeTarget(self.FULL_IDS, self.N_PROMPT),
            _FakeDraft(self.ARGMAX),
            _TensorTok([5, 6]),
            ["a", "b", "c"],
            max_new_tokens=8,
        )
        assert (accepted, total) == (6, 9)

    def test_empty_generation_is_skipped_not_counted(self):
        from soup_cli.utils.draft import measure_acceptance

        # Target returns the prompt unchanged -> nothing was generated.
        accepted, total = measure_acceptance(
            _FakeTarget([5, 6], 2),
            _FakeDraft([99, 99]),
            _TensorTok([5, 6]),
            ["a"],
            max_new_tokens=8,
        )
        assert (accepted, total) == (0, 0)

    def test_generation_is_greedy(self):
        """Sampling would make the acceptance rate non-deterministic."""
        from soup_cli.utils.draft import measure_acceptance

        target = _FakeTarget(self.FULL_IDS, self.N_PROMPT)
        measure_acceptance(
            target, _FakeDraft(self.ARGMAX), _TensorTok([5, 6]), ["a"],
            max_new_tokens=8,
        )
        assert target.calls[0]["do_sample"] is False
        assert target.calls[0]["max_new_tokens"] == 8


class TestMeasureThroughput:
    def test_discards_a_warmup_generate(self):
        from soup_cli.utils.draft import measure_throughput

        target = _FakeTarget([5, 6, 10, 11, 12], 2)
        rate = measure_throughput(
            target, _TensorTok([5, 6]), ["a", "b"], max_new_tokens=8
        )
        # 2 prompts + 1 warm-up = 3 generate calls; only 2 are timed.
        assert len(target.calls) == 3
        assert rate > 0

    def test_empty_prompts_is_zero(self):
        from soup_cli.utils.draft import measure_throughput

        target = _FakeTarget([5, 6], 2)
        assert measure_throughput(target, _TensorTok([5, 6]), []) == 0.0
        assert target.calls == []

    def test_assistant_model_is_forwarded(self):
        from soup_cli.utils.draft import measure_throughput

        target = _FakeTarget([5, 6, 10, 11, 12], 2)
        draft = _FakeDraft([0, 0, 0, 0, 0])
        measure_throughput(
            target,
            _TensorTok([5, 6]),
            ["a"],
            assistant_model=draft,
            num_assistant_tokens=7,
        )
        assert target.calls[0]["assistant_model"] is draft
        assert target.calls[0]["num_assistant_tokens"] == 7

    def test_plain_run_passes_no_assistant(self):
        from soup_cli.utils.draft import measure_throughput

        target = _FakeTarget([5, 6, 10, 11, 12], 2)
        measure_throughput(target, _TensorTok([5, 6]), ["a"])
        assert "assistant_model" not in target.calls[0]


class TestDraftNoTopLevelTorch:
    @pytest.mark.parametrize(
        "relpath",
        [("utils", "draft.py"), ("commands", "draft.py")],
    )
    def test_is_torch_free(self, relpath):
        import soup_cli

        path = Path(soup_cli.__file__).parent.joinpath(*relpath)
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


# ---------------------------------------------------------------------------
# Task 5 — soup draft CLI
# ---------------------------------------------------------------------------
class _Cfg:
    """Stand-in for transformers AutoConfig output."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size


@pytest.fixture()
def runner():
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture()
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    import json as _json

    path.write_text(
        "\n".join(_json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return str(path)


class TestDraftCliPlumbing:
    def test_registered_on_main_app(self, runner):
        from soup_cli.cli import app

        result = runner.invoke(app, ["draft", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        for sub in ("distill", "measure", "list"):
            assert sub in result.output

    def test_subcommand_help(self, runner):
        from soup_cli.commands.draft import app

        for sub in ("distill", "measure", "list"):
            result = runner.invoke(app, [sub, "--help"])
            assert result.exit_code == 0, (sub, result.output)

    @pytest.mark.parametrize("sub", ["distill", "measure"])
    def test_trust_remote_code_flag_exists(self, runner, sub):
        """code-review MEDIUM: without it, any auto_map model is a dead end."""
        import re

        from soup_cli.commands.draft import app

        result = runner.invoke(app, [sub, "--help"])
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")
        assert "--trust-remote-code" in re.sub(r"\s+", " ", plain)

    def test_trust_flag_is_threaded_into_resolution(self, monkeypatch):
        from soup_cli.commands import draft as draft_cmd

        seen: dict = {}

        def _fake_resolve(model_id, requested, console, requires_remote_code):
            seen["requested"] = requested
            return requested

        monkeypatch.setattr(
            "soup_cli.utils.trust_remote.resolve_trust_remote_code", _fake_resolve
        )
        monkeypatch.setattr(
            "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
            lambda mid: False,
        )
        assert draft_cmd._resolve_trust("org/x", True) is True
        assert seen["requested"] is True


class TestDraftDistillCli:
    def _data(self, tmp_path, rows: int = 200):
        return _write_jsonl(
            tmp_path / "d.jsonl",
            [{"messages": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "yo"}]}] * rows,
        )

    def _patch_configs(self, monkeypatch, target_vocab, draft_vocab):
        from soup_cli.commands import draft as draft_cmd

        def _fake_vocab(model_id: str, trc: bool = False) -> int:
            return target_vocab if "target" in model_id else draft_vocab

        monkeypatch.setattr(draft_cmd, "_vocab_size_of", _fake_vocab)

    def test_plan_only_writes_nothing(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "draftout", "--plan-only"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "task: distill" in result.output
        assert not (in_tmp_cwd / "draftout").exists()

    def test_vocab_mismatch_rejected(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 151936)
        data = self._data(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "draftout"],
        )
        assert result.exit_code == 1
        assert "tokenizer" in result.output.lower()
        assert "uld_strategy" in result.output

    def test_data_outside_cwd_rejected(self, runner, in_tmp_cwd, tmp_path_factory,
                                       monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        outside = tmp_path_factory.mktemp("outside")
        data = _write_jsonl(outside / "d.jsonl", [{"text": "x"}])
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "draftout", "--plan-only"],
        )
        assert result.exit_code == 1
        assert "cwd" in result.output.lower() or "outside" in result.output.lower()

    def test_output_outside_cwd_rejected(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "../escape", "--plan-only"],
        )
        assert result.exit_code == 1

    def test_output_is_cwd_rejected(self, runner, in_tmp_cwd, monkeypatch):
        """security CRITICAL: -o . would rmtree the whole working dir on success."""
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", ".", "--plan-only"],
        )
        assert result.exit_code == 1
        assert "current directory" in result.output.lower()

    def test_output_preexisting_nondraft_dir_rejected(
        self, runner, in_tmp_cwd, monkeypatch
    ):
        """security CRITICAL: overwriting an unrelated dir needs --force."""
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)
        victim = in_tmp_cwd / "important"
        victim.mkdir()
        (victim / "notes.txt").write_text("keep me", encoding="utf-8")
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "important", "--plan-only"],
        )
        assert result.exit_code == 1
        assert "--force" in result.output

    def test_output_preexisting_draft_dir_allowed(
        self, runner, in_tmp_cwd, monkeypatch
    ):
        """Re-distilling into a prior draft (has config.json) is fine."""
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)
        prior = in_tmp_cwd / "draftout"
        prior.mkdir()
        (prior / "config.json").write_text("{}", encoding="utf-8")
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "draftout", "--plan-only"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_missing_data_file_rejected(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_configs(monkeypatch, 49152, 49152)
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", "nope.jsonl", "-o", "draftout", "--plan-only"],
        )
        assert result.exit_code == 1

    def test_generated_config_is_schema_valid(self, in_tmp_cwd):
        """The rendered YAML must actually load as a SoupConfig."""
        from soup_cli.commands.draft import _build_distill_config_yaml
        from soup_cli.config.loader import load_config_from_string

        data = self._data(in_tmp_cwd)
        yaml_text = _build_distill_config_yaml(
            draft_base="org/tiny",
            target="org/target",
            data=data,
            out_dir="draftout",
            steps=100,
            data_rows=200,
        )
        cfg = load_config_from_string(yaml_text)
        assert cfg.task == "distill"
        assert cfg.base == "org/tiny"
        assert cfg.training.teacher_model == "org/target"

    def test_trainer_output_goes_to_a_nested_adapter_dir(self, in_tmp_cwd):
        """code-review CRITICAL: the distill trainer only ever writes a LoRA
        adapter (never dense base weights), so it must NOT train straight into
        -o — the merge needs a separate adapter dir plus the base weights."""
        import yaml

        from soup_cli.commands.draft import _ADAPTER_SUBDIR, _build_distill_config_yaml

        yaml_text = _build_distill_config_yaml(
            draft_base="org/tiny",
            target="org/target",
            data="d.jsonl",
            out_dir=os.path.join("draftout", _ADAPTER_SUBDIR),
            steps=100,
            data_rows=200,
        )
        parsed = yaml.safe_load(yaml_text)
        assert parsed["output"].endswith(_ADAPTER_SUBDIR)
        assert parsed["base"] == "org/tiny"  # student = the draft base

    def test_run_distill_merges_base_plus_adapter_not_out_dir_into_itself(
        self, in_tmp_cwd, monkeypatch
    ):
        """The merge must load base weights from --draft-base and the adapter
        from the nested dir. Fusing out_dir into itself (the original bug) can
        never work: out_dir holds no base weights."""
        from soup_cli.commands import draft as draft_cmd

        calls: dict = {}

        class _Result:
            returncode = 0
            stdout = b""
            stderr = b""

        def _fake_run(argv, **kwargs):
            # Simulate the trainer writing ONLY an adapter into its output dir.
            adapter = Path("draftout") / draft_cmd._ADAPTER_SUBDIR
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            return _Result()

        def _fake_merge(*, base_model, adapter_dir, out_dir, trc=False):
            calls.update(
                base_model=base_model, adapter_dir=adapter_dir, out_dir=out_dir
            )

        import subprocess as _sp

        monkeypatch.setattr(_sp, "run", _fake_run)
        monkeypatch.setattr(draft_cmd, "merge_adapter_to_dense", _fake_merge)

        self._data(in_tmp_cwd)
        draft_cmd._run_distill(
            draft_base="org/tiny",
            target="org/target",
            data="d.jsonl",
            out_dir="draftout",
            steps=100,
            data_rows=200,
        )
        assert calls["base_model"] == "org/tiny", "base weights must come from --draft-base"
        assert calls["adapter_dir"].endswith(draft_cmd._ADAPTER_SUBDIR)
        assert calls["out_dir"] == "draftout"
        assert calls["adapter_dir"] != calls["out_dir"]

    def test_run_distill_fails_loudly_when_no_adapter_was_written(
        self, in_tmp_cwd, monkeypatch
    ):
        from soup_cli.commands import draft as draft_cmd

        class _Result:
            returncode = 0
            stdout = b""
            stderr = b""

        import subprocess as _sp

        monkeypatch.setattr(_sp, "run", lambda argv, **kw: _Result())
        monkeypatch.setattr(
            draft_cmd, "merge_adapter_to_dense", lambda **kw: None
        )
        with pytest.raises(RuntimeError, match="no adapter"):
            draft_cmd._run_distill(
                draft_base="org/tiny",
                target="org/target",
                data="d.jsonl",
                out_dir="draftout",
                steps=100,
                data_rows=200,
            )

    def test_newline_in_model_id_cannot_inject_yaml_keys(self, in_tmp_cwd):
        """python-review CRITICAL: raw interpolation let a crafted --target
        smuggle sibling keys into the training: block. Every user string is
        json.dumps'd, so a newline stays inside the scalar."""
        import yaml

        from soup_cli.commands.draft import _build_distill_config_yaml

        hostile = "org/x\ntraining:\n  reward_fn: ../../evil.py"
        yaml_text = _build_distill_config_yaml(
            draft_base="org/tiny",
            target=hostile,
            data="d.jsonl",
            out_dir="draftout",
            steps=100,
            data_rows=200,
        )
        parsed = yaml.safe_load(yaml_text)
        # The payload landed as a VALUE, not as new keys.
        assert parsed["training"]["teacher_model"] == hostile
        assert "reward_fn" not in parsed["training"]
        assert parsed["task"] == "distill"

    def test_absurd_steps_over_tiny_data_is_refused(self, in_tmp_cwd):
        """500 steps over 4 rows = 125 epochs — refuse rather than emit it."""
        from soup_cli.commands.draft import _build_distill_config_yaml

        with pytest.raises(ValueError, match="epochs"):
            _build_distill_config_yaml(
                draft_base="org/tiny",
                target="org/target",
                data="d.jsonl",
                out_dir="draftout",
                steps=500,
                data_rows=4,
            )

    def test_happy_path_trains_fuses_and_registers(
        self, runner, in_tmp_cwd, monkeypatch, draft_registry
    ):
        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app
        from soup_cli.utils.draft import lookup_draft

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)

        trained: dict = {}

        def _fake_train(**kwargs):
            trained.update(kwargs)
            # The real subprocess writes the adapter into out_dir; simulate the
            # fused dense draft landing in the output dir.
            Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(draft_cmd, "_run_distill", _fake_train)

        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "draftout"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert trained["target"] == "org/target"
        registered = lookup_draft("org/target")
        assert registered == os.path.realpath(str(in_tmp_cwd / "draftout"))

    def test_no_register_flag_skips_registry(
        self, runner, in_tmp_cwd, monkeypatch, draft_registry
    ):
        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app
        from soup_cli.utils.draft import lookup_draft

        self._patch_configs(monkeypatch, 49152, 49152)
        data = self._data(in_tmp_cwd)
        monkeypatch.setattr(
            draft_cmd,
            "_run_distill",
            lambda **kw: Path(kw["out_dir"]).mkdir(parents=True, exist_ok=True),
        )
        result = runner.invoke(
            app,
            ["distill", "--target", "org/target", "--draft-base", "org/tiny",
             "--data", data, "-o", "draftout", "--no-register"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert lookup_draft("org/target") is None


class TestDraftMeasureCli:
    def _prompts(self, tmp_path):
        return _write_jsonl(
            tmp_path / "p.jsonl", [{"prompt": "What is 2+2?"}, {"prompt": "Hi"}]
        )

    def _patch_load(self, monkeypatch, *, compatible=True):
        from soup_cli.commands import draft as draft_cmd

        tok_a = _FakeTok(49152)
        tok_b = _FakeTok(49152) if compatible else _FakeTok(151936)

        def _fake_load(model_id, **kwargs):
            tok = tok_a if "target" in model_id else tok_b
            return object(), tok, "cpu"

        monkeypatch.setattr(draft_cmd, "_load_pair_member", _fake_load)

    def test_happy_path_writes_report_and_exits_zero(
        self, runner, in_tmp_cwd, monkeypatch
    ):
        import json as _json

        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch)
        monkeypatch.setattr(draft_cmd, "measure_acceptance", lambda *a, **k: (75, 100))
        monkeypatch.setattr(draft_cmd, "measure_throughput", lambda *a, **k: 20.0)

        prompts = self._prompts(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", prompts, "-o", "report.json"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "STRONG" in result.output
        data = _json.loads((in_tmp_cwd / "report.json").read_text(encoding="utf-8"))
        assert data["acceptance_rate"] == 0.75
        assert data["verdict"] == "STRONG"
        assert data["n_generated_tokens"] == 100

    def test_below_min_acceptance_exits_two(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch)
        monkeypatch.setattr(draft_cmd, "measure_acceptance", lambda *a, **k: (40, 100))
        monkeypatch.setattr(draft_cmd, "measure_throughput", lambda *a, **k: 20.0)

        prompts = self._prompts(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", prompts, "--min-acceptance", "0.6"],
        )
        assert result.exit_code == 2
        assert "60.0%" in result.output
        assert "below" in result.output.lower()

    def test_mismatched_tokenizer_exits_one(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch, compatible=False)
        prompts = self._prompts(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", prompts],
        )
        assert result.exit_code == 1
        assert "tokenizer" in result.output.lower()

    def test_prompts_outside_cwd_rejected(
        self, runner, in_tmp_cwd, tmp_path_factory, monkeypatch
    ):
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch)
        outside = tmp_path_factory.mktemp("outside2")
        prompts = _write_jsonl(outside / "p.jsonl", [{"prompt": "hi"}])
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", prompts],
        )
        assert result.exit_code == 1

    def test_empty_prompts_rejected(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch)
        empty = in_tmp_cwd / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", str(empty)],
        )
        assert result.exit_code == 1

    def test_bad_min_acceptance_rejected(self, runner, in_tmp_cwd, monkeypatch):
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch)
        prompts = self._prompts(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", prompts, "--min-acceptance", "1.5"],
        )
        assert result.exit_code != 0

    def test_zero_generated_tokens_is_not_a_crash(
        self, runner, in_tmp_cwd, monkeypatch
    ):
        """A target that emits EOS immediately must not divide by zero."""
        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        self._patch_load(monkeypatch)
        monkeypatch.setattr(draft_cmd, "measure_acceptance", lambda *a, **k: (0, 0))
        monkeypatch.setattr(draft_cmd, "measure_throughput", lambda *a, **k: 5.0)

        prompts = self._prompts(in_tmp_cwd)
        result = runner.invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", prompts],
        )
        assert result.exit_code == 1
        assert "no tokens" in result.output.lower()


class TestDraftListCli:
    def test_lists_registered_drafts(self, runner, draft_registry, tmp_path):
        from soup_cli.commands.draft import app
        from soup_cli.utils.draft import register_draft

        draft = tmp_path / "d"
        draft.mkdir()
        register_draft("org/target-7b", str(draft), 0.71)

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "org/target-7b" in result.output
        assert "71" in result.output

    def test_empty_registry_is_not_an_error(self, runner, draft_registry):
        from soup_cli.commands.draft import app

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "no draft" in result.output.lower()
