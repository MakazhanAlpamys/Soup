"""Regression tests for #681.

``wasserstein`` / ``topk_align`` forwarded the student's own ``input_ids``
straight to the teacher, clamped into range. Clamping stops an index error
but does not translate tokens between vocabularies, so an incompatible pair
trained on logits the teacher computed for the wrong text, with a finite
loss and no visible failure. ``wasserstein_aligned`` (#258) already handles
mismatched tokenizers correctly by re-tokenizing and aligning; the fix here
is to fail early for the two strategies that don't, using the same
tokenizer-compatibility probe (#304's ``same_tokenizer``) this codebase
already trusts for the analogous speculative-decoding question.
"""

from __future__ import annotations

import types

import pytest


class _FakeTok:
    """Tiny stand-in tokenizer: reports a vocab size and encodes via a map."""

    def __init__(self, vocab_size: int, encode_map: dict | None = None):
        self.vocab_size = vocab_size
        self._encode_map = encode_map or {}
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._encode_map.get(text, [1, 2, 3]))


class _FakeModel:
    """Enough of an HF causal-LM to get DistillTrainerWrapper.setup() past
    the student/teacher load without touching a real model or the network."""

    def __init__(self, vocab_size: int):
        self.config = types.SimpleNamespace(vocab_size=vocab_size)

    def eval(self):
        return self

    def parameters(self):
        return iter([])


# ---------------------------------------------------------------------------
# PART 1: the pure compatibility gate
# ---------------------------------------------------------------------------
class TestRequireUldIdCompatibleTokenizers:
    def test_identical_tokenizer_passes_wasserstein(self):
        from soup_cli.trainer.distill import _require_uld_id_compatible_tokenizers

        tok = _FakeTok(32000)
        _require_uld_id_compatible_tokenizers("wasserstein", tok, tok)

    def test_identical_tokenizer_passes_topk_align(self):
        from soup_cli.trainer.distill import _require_uld_id_compatible_tokenizers

        tok = _FakeTok(32000)
        _require_uld_id_compatible_tokenizers("topk_align", tok, tok)

    def test_mismatched_ids_same_vocab_size_raises_for_wasserstein(self):
        """The reporter's own repro shape: same-sized vocabs, different ids."""
        from soup_cli.trainer.distill import _require_uld_id_compatible_tokenizers

        probe = "The quick brown fox jumps over 13 lazy dogs."
        student = _FakeTok(32000, {probe: [1, 6, 7]})
        teacher = _FakeTok(32000, {probe: [1, 4, 4]})
        with pytest.raises(ValueError, match="wasserstein_aligned"):
            _require_uld_id_compatible_tokenizers("wasserstein", student, teacher)

    def test_mismatched_ids_raises_for_topk_align(self):
        from soup_cli.trainer.distill import _require_uld_id_compatible_tokenizers

        probe = "The quick brown fox jumps over 13 lazy dogs."
        student = _FakeTok(32000, {probe: [1, 6, 7]})
        teacher = _FakeTok(32000, {probe: [1, 4, 4]})
        with pytest.raises(ValueError, match="topk_align"):
            _require_uld_id_compatible_tokenizers("topk_align", student, teacher)

    def test_different_vocab_size_raises(self):
        from soup_cli.trainer.distill import _require_uld_id_compatible_tokenizers

        with pytest.raises(ValueError, match="wasserstein_aligned"):
            _require_uld_id_compatible_tokenizers(
                "wasserstein", _FakeTok(32000), _FakeTok(49152)
            )

    def test_wasserstein_aligned_is_not_gated(self):
        """#258's own strategy re-tokenizes and aligns; it never reuses raw
        ids, so a mismatched pair here must not raise."""
        from soup_cli.trainer.distill import _require_uld_id_compatible_tokenizers

        _require_uld_id_compatible_tokenizers(
            "wasserstein_aligned", _FakeTok(32000), _FakeTok(49152)
        )


# ---------------------------------------------------------------------------
# PART 2: wired into DistillTrainerWrapper.setup(), fails before training
# starts rather than on the first batch.
# ---------------------------------------------------------------------------
class TestDistillSetupRejectsIncompatibleUld:
    def _config(self, uld_strategy: str) -> "object":
        from soup_cli.config.loader import load_config_from_string

        top_k_line = "  uld_top_k: 4\n" if uld_strategy == "topk_align" else ""
        yaml_text = (
            "base: fake/student\n"
            "task: distill\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            f"  teacher_model: fake/teacher\n"
            f"  uld_strategy: {uld_strategy}\n"
            f"{top_k_line}"
            "output: ./out\n"
        )
        return load_config_from_string(yaml_text)

    def _patch_model_loading(self, monkeypatch, student_tok, teacher_tok):
        import peft
        import transformers

        from soup_cli.utils import peft_wiring

        def _fake_tok(model_id, trust_remote_code=False, **kwargs):
            return {"fake/student": student_tok, "fake/teacher": teacher_tok}[model_id]

        def _fake_model(model_id, trust_remote_code=False, device_map=None, **kwargs):
            return _FakeModel(vocab_size=student_tok.vocab_size)

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", staticmethod(_fake_tok)
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            staticmethod(_fake_model),
        )
        monkeypatch.setattr(
            peft_wiring, "resolve_lora_target_modules", lambda model, configured: ["q_proj"]
        )
        monkeypatch.setattr(peft_wiring, "apply_pre_lora_patches", lambda *a, **k: None)
        monkeypatch.setattr(peft_wiring, "apply_post_lora_patches", lambda *a, **k: None)
        monkeypatch.setattr(peft, "get_peft_model", lambda model, config: model)

    @pytest.mark.parametrize("strategy", ["wasserstein", "topk_align"])
    def test_setup_raises_before_training_on_mismatched_tokenizers(
        self, monkeypatch, strategy
    ):
        pytest.importorskip("torch")
        from soup_cli.trainer.distill import DistillTrainerWrapper

        probe = "The quick brown fox jumps over 13 lazy dogs."
        student_tok = _FakeTok(32000, {probe: [1, 6, 7]})
        teacher_tok = _FakeTok(32000, {probe: [1, 4, 4]})
        self._patch_model_loading(monkeypatch, student_tok, teacher_tok)

        wrapper = DistillTrainerWrapper(self._config(strategy), device="cpu")
        with pytest.raises(ValueError, match="wasserstein_aligned"):
            wrapper.setup({})

    def test_setup_does_not_raise_id_compat_error_on_identical_tokenizer(
        self, monkeypatch
    ):
        """Same-tokenizer fast path (#258's own requirement): setup() must
        get past the compatibility gate without our new error firing."""
        pytest.importorskip("torch")
        from soup_cli.trainer.distill import DistillTrainerWrapper

        tok = _FakeTok(32000)
        self._patch_model_loading(monkeypatch, tok, tok)

        wrapper = DistillTrainerWrapper(self._config("wasserstein"), device="cpu")
        try:
            wrapper.setup({})
        except ValueError as exc:
            assert "wasserstein_aligned" not in str(exc)
        except Exception:
            # Dataset/Trainer construction needs a real dataset and output
            # dir; only the tokenizer-compatibility gate is under test here.
            pass
