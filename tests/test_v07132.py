"""v0.71.32 — ASR (Whisper) fine-tuning.

Covers pure-python WER/CER metrics (``utils/asr_metrics``), the ``task='asr'``
schema surface (task/format Literals + ``asr_language``/``asr_task`` +
cross-validators), the ``AsrTrainerWrapper`` (arch guard + row validation +
``Seq2SeqTrainer`` build), ``soup infer --task asr``, and the whisper/smolvlm
recipes.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Task 1 — WER / CER metrics
# ---------------------------------------------------------------------------


class TestAsrMetrics:
    def test_normalize_text_lower_punct_ws(self):
        from soup_cli.utils.asr_metrics import normalize_text

        assert normalize_text("The Cat, sat!") == "the cat sat"
        assert normalize_text("the   cat    sat") == "the cat sat"
        assert normalize_text("HELLO") == "hello"

    def test_normalize_text_opt_out(self):
        from soup_cli.utils.asr_metrics import normalize_text

        assert normalize_text("The Cat!", lower=False, strip_punct=False) == "The Cat!"

    def test_wer_identity_zero(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "the cat sat") == 0.0

    def test_wer_all_substitutions_one(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "a dog ran") == 1.0

    def test_wer_single_deletion(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "the cat") == pytest.approx(1 / 3)

    def test_wer_single_insertion(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat", "the cat sat") == pytest.approx(1 / 2)

    def test_wer_single_substitution(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)

    def test_wer_normalizes_by_default(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("The cat, sat.", "the cat sat") == 0.0

    def test_wer_empty_both_zero(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("", "") == 0.0

    def test_wer_empty_ref_nonempty_hyp_one(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("", "hello world") == 1.0

    def test_cer_char_level(self):
        from soup_cli.utils.asr_metrics import cer

        assert cer("abc", "abc") == 0.0
        assert cer("cat", "cot") == pytest.approx(1 / 3)

    def test_word_accuracy_is_one_minus_wer(self):
        from soup_cli.utils.asr_metrics import word_accuracy

        assert word_accuracy("the cat sat", "the cat sat") == 1.0
        # all-wrong -> wer 1.0 -> accuracy 0.0 (clamped, never negative)
        assert word_accuracy("the cat sat", "a dog ran") == 0.0

    def test_word_accuracy_clamps_at_zero(self):
        from soup_cli.utils.asr_metrics import word_accuracy

        # many insertions push wer > 1.0; accuracy must clamp to 0.0
        assert word_accuracy("cat", "a b c d e f") == 0.0

    def test_corpus_wer_is_not_mean(self):
        from soup_cli.utils.asr_metrics import corpus_wer

        # per-example wers are 0.0 and 1.0 -> mean would be 0.5.
        # corpus = (0 edits + 1 edit) / (3 + 1 ref words) = 0.25.
        val = corpus_wer(["the cat sat", "hello"], ["the cat sat", "world"])
        assert val == pytest.approx(0.25)

    def test_corpus_wer_length_mismatch_raises(self):
        from soup_cli.utils.asr_metrics import corpus_wer

        with pytest.raises(ValueError, match="same length"):
            corpus_wer(["a"], ["a", "b"])

    def test_seq_cap_raises(self):
        from soup_cli.utils.asr_metrics import _MAX_SEQ, wer

        long_ref = " ".join(["w"] * (_MAX_SEQ + 1))
        with pytest.raises(ValueError, match="too long"):
            wer(long_ref, "short", normalize=False)


class TestNoTopLevelTorch:
    def test_asr_metrics_has_no_heavy_top_level_import(self):
        import soup_cli.utils.asr_metrics as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = {"torch", "transformers", "peft", "datasets"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned
