"""v0.71.10 "RAG family" — RAFT / RA-DIT / steering / citation live wiring.

Closes #199 (RAFT span-mask trainer), #200 (RA-DIT auto-link), #201 (live
CAA/ITI/RepE steering + serve --steer decode hook), #202 (citation-span
loss-mask + soup eval citation + diagnose citation mode).

Pure-Python / CPU tests + tiny-tensor torch tests; the live model paths are
step-6 smoked on SmolLM2-135M (RTX 3050).
"""

from __future__ import annotations

import re
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Word-level fake tokenizer with offset-mapping support.

    Tokenises on non-whitespace runs; each token gets a deterministic id and
    its ``(start, end)`` char offset. Supports the subset of the HF tokenizer
    API that ``utils.raft`` uses.
    """

    eos_token_id = 99
    pad_token_id = 0
    chat_template = None

    _WORD = re.compile(r"\S+")

    def __init__(self, fast: bool = True):
        self._fast = fast
        self._vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab) + 1  # 1-based; 0 = pad
        return self._vocab[tok]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        toks = list(self._WORD.finditer(text))
        ids = [self._id(m.group(0)) for m in toks]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            if not self._fast:
                raise NotImplementedError("slow tokenizer has no offsets")
            out["offset_mapping"] = [(m.start(), m.end()) for m in toks]
        return out


class _FakeBaseTrainer:
    """Minimal stand-in for transformers.Trainer (compute_loss test)."""

    def __init__(self, *args, **kwargs):
        pass


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Returns fixed logits regardless of input (compute_loss test)."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **kwargs):
        return _FakeOutputs(self._logits)


def _raft_row(distractors: int = 2) -> dict:
    return {
        "query": "What is the capital of France?",
        "golden_doc": "Paris has been the capital of France since 987.",
        "distractor_docs": [f"Distractor number {i} about geese." for i in range(distractors)],
        "answer": "The capital is Paris [doc-0].",
    }


# ---------------------------------------------------------------------------
# #199 — build_raft_prompt
# ---------------------------------------------------------------------------


class TestBuildRaftPrompt:
    def test_basic_prompt_shape(self):
        from soup_cli.utils.raft import RAFT_INSTRUCTION, build_raft_prompt

        composed = build_raft_prompt(_raft_row(2))
        assert RAFT_INSTRUCTION in composed.prompt
        assert "Question: What is the capital of France?" in composed.prompt
        assert "Documents:" in composed.prompt
        assert composed.prompt.rstrip().endswith("Answer:")
        assert composed.answer == "The capital is Paris [doc-0]."

    def test_doc_ids_assigned_and_golden_tracked(self):
        from soup_cli.utils.raft import build_raft_prompt

        composed = build_raft_prompt(_raft_row(2))
        # 1 golden + 2 distractors → doc-0, doc-1, doc-2.
        assert composed.doc_ids == ("doc-0", "doc-1", "doc-2")
        assert composed.golden_doc_id in composed.doc_ids
        # The golden doc text must appear next to its assigned id.
        assert f"[{composed.golden_doc_id}] Paris has been the capital" in composed.prompt

    def test_shuffle_reproducible_for_same_seed(self):
        from soup_cli.utils.raft import build_raft_prompt

        a = build_raft_prompt(_raft_row(4), shuffle_seed=42, row_index=3)
        b = build_raft_prompt(_raft_row(4), shuffle_seed=42, row_index=3)
        assert a.prompt == b.prompt
        assert a.golden_doc_id == b.golden_doc_id

    def test_different_row_index_differs(self):
        from soup_cli.utils.raft import build_raft_prompt

        # Across a handful of indices the golden position should vary at least
        # once (deterministic shuffle keyed on index).
        positions = {
            build_raft_prompt(_raft_row(6), shuffle_seed=1, row_index=i).golden_doc_id
            for i in range(8)
        }
        assert len(positions) > 1

    def test_no_distractors_ok(self):
        from soup_cli.utils.raft import build_raft_prompt

        composed = build_raft_prompt(_raft_row(0))
        assert composed.doc_ids == ("doc-0",)
        assert composed.golden_doc_id == "doc-0"

    def test_missing_field_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises(ValueError, match="query"):
            build_raft_prompt({"golden_doc": "x", "answer": "y"})

    def test_shuffle_seed_bool_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises(TypeError, match="shuffle_seed"):
            build_raft_prompt(_raft_row(), shuffle_seed=True)

    def test_row_index_negative_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises(ValueError, match="row_index"):
            build_raft_prompt(_raft_row(), row_index=-1)

    def test_too_many_docs_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        row = _raft_row(0)
        row["distractor_docs"] = [f"d{i}" for i in range(65)]
        with pytest.raises(ValueError, match="documents"):
            build_raft_prompt(row)

    def test_non_mapping_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises(ValueError, match="mapping"):
            build_raft_prompt(["not", "a", "dict"])


# ---------------------------------------------------------------------------
# #199 — tokenize_raft_example
# ---------------------------------------------------------------------------


class TestTokenizeRaftExample:
    def _composed(self):
        from soup_cli.utils.raft import build_raft_prompt

        return build_raft_prompt(_raft_row(1))

    def test_answer_only_mask(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer()
        row = tokenize_raft_example(tok, self._composed(), max_length=512)
        labels = row["labels"]
        weights = row["loss_weights"]
        # The leading run (prompt) is masked -100 / weight 0; the answer tail
        # is unmasked / weight 1.
        assert any(x == -100 for x in labels)
        assert any(x != -100 for x in labels)
        # Prompt positions: label -100 ⇔ weight 0.0.
        for label, w in zip(labels, weights):
            if label == -100:
                assert w == 0.0
            else:
                assert w >= 1.0
        assert len(row["input_ids"]) == len(labels) == len(weights)
        assert row["attention_mask"] == [1] * len(row["input_ids"])

    def test_eos_appended(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer()
        row = tokenize_raft_example(tok, self._composed(), max_length=512)
        assert row["input_ids"][-1] == tok.eos_token_id

    def test_truncation_respects_max_length(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer()
        row = tokenize_raft_example(tok, self._composed(), max_length=8)
        assert len(row["input_ids"]) == 8
        assert len(row["labels"]) == 8
        assert len(row["loss_weights"]) == 8

    def test_citation_boost_applied(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer()
        # Answer with a [doc-0] citation token → that token's weight boosted.
        composed = self._composed()
        row = tokenize_raft_example(
            tok, composed, max_length=512, citation_faithful=True
        )
        # At least one answer token gets the boost (> 1.0).
        assert any(w > 1.0 for w in row["loss_weights"])

    def test_no_citation_boost_when_disabled(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer()
        row = tokenize_raft_example(
            tok, self._composed(), max_length=512, citation_faithful=False
        )
        assert all(w in (0.0, 1.0) for w in row["loss_weights"])

    def test_slow_tokenizer_degrades_to_flat_mask(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer(fast=False)  # no offset mapping
        row = tokenize_raft_example(
            tok, self._composed(), max_length=512, citation_faithful=True
        )
        # No offsets → flat answer weights (no boost), never raises.
        assert all(w in (0.0, 1.0) for w in row["loss_weights"])

    def test_bad_max_length_rejected(self):
        from soup_cli.utils.raft import tokenize_raft_example

        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="max_length"):
            tokenize_raft_example(tok, self._composed(), max_length=4)
        with pytest.raises(ValueError, match="max_length"):
            tokenize_raft_example(tok, self._composed(), max_length=True)


# ---------------------------------------------------------------------------
# #199 — citation_span_token_weights
# ---------------------------------------------------------------------------


class TestCitationSpanTokenWeights:
    def test_overlapping_tokens_boosted(self):
        from soup_cli.utils.raft import citation_span_token_weights

        answer = "Paris [doc-0] is."
        # offsets for: "Paris"(0-5) "[doc-0]"(6-13) "is."(14-17)
        offsets: List[Tuple[int, int]] = [(0, 5), (6, 13), (14, 17)]
        weights = citation_span_token_weights(answer, offsets, boost=5.0)
        assert weights == [1.0, 5.0, 1.0]

    def test_no_citation_all_one(self):
        from soup_cli.utils.raft import citation_span_token_weights

        weights = citation_span_token_weights("plain text", [(0, 5), (6, 10)])
        assert weights == [1.0, 1.0]

    def test_boost_below_one_rejected(self):
        from soup_cli.utils.raft import citation_span_token_weights

        with pytest.raises(ValueError, match="boost"):
            citation_span_token_weights("x", [(0, 1)], boost=0.5)

    def test_boost_bool_rejected(self):
        from soup_cli.utils.raft import citation_span_token_weights

        with pytest.raises(TypeError, match="boost"):
            citation_span_token_weights("x", [(0, 1)], boost=True)


# ---------------------------------------------------------------------------
# #199 — RaftDataCollator
# ---------------------------------------------------------------------------


class TestRaftDataCollator:
    def test_pads_ragged_batch(self):
        import torch

        from soup_cli.trainer.raft import RaftDataCollator

        collate = RaftDataCollator(_FakeTokenizer())
        batch = collate([
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
             "labels": [-100, 2, 3], "loss_weights": [0.0, 1.0, 1.0]},
            {"input_ids": [4, 5], "attention_mask": [1, 1],
             "labels": [-100, 5], "loss_weights": [0.0, 1.0]},
        ])
        assert batch["input_ids"].shape == (2, 3)
        # Row 2 padded with pad_id=0 in input, -100 in labels, 0.0 in weights.
        assert batch["input_ids"][1].tolist() == [4, 5, 0]
        assert batch["labels"][1].tolist() == [-100, 5, -100]
        assert batch["loss_weights"][1].tolist() == [0.0, 1.0, 0.0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 0]
        assert batch["loss_weights"].dtype == torch.float32

    def test_empty_batch_rejected(self):
        from soup_cli.trainer.raft import RaftDataCollator

        with pytest.raises(ValueError, match="empty batch"):
            RaftDataCollator(_FakeTokenizer())([])


# ---------------------------------------------------------------------------
# #199 — make_raft_trainer_class compute_loss
# ---------------------------------------------------------------------------


class TestRaftTrainerComputeLoss:
    def test_factory_caches(self):
        from soup_cli.trainer.raft import make_raft_trainer_class

        a = make_raft_trainer_class(_FakeBaseTrainer)
        b = make_raft_trainer_class(_FakeBaseTrainer)
        assert a is b
        assert "_RaftTrainer" in a.__name__

    def test_weighted_loss_finite(self):
        import torch

        from soup_cli.trainer.raft import make_raft_trainer_class

        cls = make_raft_trainer_class(_FakeBaseTrainer)
        trainer = cls()
        vocab = 10
        logits = torch.randn(1, 4, vocab)
        model = _FakeModel(logits)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 5, 6]]),
            "loss_weights": torch.tensor([[0.0, 0.0, 1.0, 5.0]]),
        }
        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_all_one_weights_equals_answer_only_ce(self):
        import torch
        from torch.nn.functional import cross_entropy

        from soup_cli.trainer.raft import make_raft_trainer_class

        cls = make_raft_trainer_class(_FakeBaseTrainer)
        trainer = cls()
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 10)
        model = _FakeModel(logits)
        labels = torch.tensor([[-100, -100, 5, 6]])
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": labels,
            "loss_weights": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        }
        weighted = trainer.compute_loss(model, inputs).item()
        # Reference answer-only CE.
        ref = cross_entropy(
            logits[:, :-1, :].reshape(-1, 10),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        ).item()
        assert weighted == pytest.approx(ref, abs=1e-5)


# ---------------------------------------------------------------------------
# #199 — schema raft_shuffle_seed
# ---------------------------------------------------------------------------


class TestRaftShuffleSeedSchema:
    def _yaml(self, seed_line: str) -> str:
        return (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: sft\n"
            "data:\n"
            "  train: ./data/raft.jsonl\n"
            "  format: raft\n"
            f"{seed_line}"
            "training:\n"
            "  epochs: 1\n"
            "output: ./output\n"
        )

    def test_accepts_int(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("  raft_shuffle_seed: 42\n"))
        assert cfg.data.raft_shuffle_seed == 42

    def test_default_none(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml(""))
        assert cfg.data.raft_shuffle_seed is None

    def test_bool_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="raft_shuffle_seed"):
            load_config_from_string(self._yaml("  raft_shuffle_seed: true\n"))

    def test_negative_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="raft_shuffle_seed"):
            load_config_from_string(self._yaml("  raft_shuffle_seed: -1\n"))


# ---------------------------------------------------------------------------
# #199 — SFT wiring source-grep regression guards
# ---------------------------------------------------------------------------


class TestSftRaftWiring:
    def _sft_src(self) -> str:
        import soup_cli.trainer.sft as sft

        with open(sft.__file__, encoding="utf-8") as fh:
            return fh.read()

    def test_sft_routes_raft(self):
        src = self._sft_src()
        assert "_prepare_raft_dataset" in src
        assert "make_raft_trainer_class" in src
        assert "RaftDataCollator" in src
        assert 'cfg.data.format == "raft"' in src

    def test_raft_modules_have_no_top_level_torch(self):
        import soup_cli.utils.raft as raft

        with open(raft.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


# ===========================================================================
# #202 — per-style citation extractors + eval citation + diagnose mode
# ===========================================================================


class TestPerStyleExtractors:
    def test_bracket_default(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        assert extract_citation_ids("see [doc-1] and [doc-2]") == ("doc-1", "doc-2")

    def test_inline_style(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        assert extract_citation_ids("see (doc-1) here", style="inline") == ("doc-1",)
        # bracket extractor must NOT match the parenthetical form.
        assert extract_citation_ids("see (doc-1) here") == ()

    def test_footnote_style(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        assert extract_citation_ids("fact[^3] more", style="footnote") == ("3",)
        # footnote `[^3]` must not be picked up by the bracket extractor
        # (starts with ^ after the bracket → fails alnum-leading).
        assert extract_citation_ids("fact[^3] more") == ()

    def test_citation_spans_cover_delimiters(self):
        from soup_cli.utils.citation_faithful import citation_spans

        spans = citation_spans("ab [doc-0] cd")
        assert spans == ((3, 10),)  # covers the full "[doc-0]"

    def test_score_citations_with_style(self):
        from soup_cli.utils.citation_faithful import score_citations

        cs = score_citations(
            predicted="answer (doc-0)", expected_ids=["doc-0"], style="inline"
        )
        assert cs.precision == 1.0
        assert cs.recall == 1.0

    def test_unknown_style_rejected(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        with pytest.raises(ValueError, match="citation_style"):
            extract_citation_ids("x", style="bogus")


class TestEvalCitationCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_citation_in_eval_help(self):
        from soup_cli.cli import app

        result = self._runner().invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "citation" in result.output

    def test_citation_predicted_expected(self):
        import json

        from soup_cli.cli import app

        runner = self._runner()
        with runner.isolated_filesystem():
            with open("c.jsonl", "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"predicted": "Paris [doc-0].", "expected_ids": ["doc-0"]})
                    + "\n"
                )
                fh.write(
                    json.dumps({"predicted": "Berlin [doc-2].", "expected_ids": ["doc-0"]})
                    + "\n"
                )
            result = runner.invoke(
                app, ["eval", "citation", "c.jsonl", "--output", "out.json"]
            )
            assert result.exit_code == 0, (result.output, result.exception)
            with open("out.json", encoding="utf-8") as fh:
                payload = json.load(fh)
            assert payload["n_rows"] == 2
            # First row recall 1.0, second 0.0 → mean 0.5.
            assert payload["aggregate"]["recall"] == pytest.approx(0.5)

    def test_citation_raft_rows(self):
        import json

        from soup_cli.cli import app

        runner = self._runner()
        with runner.isolated_filesystem():
            with open("raft.jsonl", "w", encoding="utf-8") as fh:
                # answer cites doc-0; with no shuffle (default seed 0) golden
                # id is deterministic — the row scores its own answer.
                fh.write(json.dumps({
                    "query": "q", "golden_doc": "g", "distractor_docs": [],
                    "answer": "see [doc-0]",
                }) + "\n")
            result = runner.invoke(app, ["eval", "citation", "raft.jsonl"])
            assert result.exit_code == 0, (result.output, result.exception)
            assert "Citation aggregate" in result.output

    def test_invalid_style_exit_2(self):
        from soup_cli.cli import app

        runner = self._runner()
        with runner.isolated_filesystem():
            with open("c.jsonl", "w", encoding="utf-8") as fh:
                fh.write('{"predicted": "x", "expected_ids": ["a"]}\n')
            result = runner.invoke(app, ["eval", "citation", "c.jsonl", "--style", "bogus"])
            assert result.exit_code == 2

    def test_missing_file_exit(self):
        from soup_cli.cli import app

        runner = self._runner()
        with runner.isolated_filesystem():
            result = runner.invoke(app, ["eval", "citation", "nope.jsonl"])
            assert result.exit_code != 0

    def test_no_scorable_rows_exit_2(self):
        from soup_cli.cli import app

        runner = self._runner()
        with runner.isolated_filesystem():
            with open("c.jsonl", "w", encoding="utf-8") as fh:
                fh.write('{"unrelated": "row"}\n')
            result = runner.invoke(app, ["eval", "citation", "c.jsonl"])
            assert result.exit_code == 2


class TestDiagnoseCitationMode:
    def test_citation_in_failure_modes(self):
        from soup_cli.utils.diagnose.report import FAILURE_MODES

        assert "citation" in FAILURE_MODES

    def test_is_raft_row(self):
        from soup_cli.utils.diagnose.citation import is_raft_row

        assert is_raft_row({"query": "q", "golden_doc": "g", "answer": "a"})
        assert not is_raft_row({"prompt": "p"})
        assert not is_raft_row("not a dict")

    def test_score_citation_recall_ok(self):
        from soup_cli.utils.diagnose.citation import score_citation

        rows = [_raft_row(2)]

        def gen(prompt: str) -> str:
            # Always cite the golden doc id present in the prompt's [doc-N].
            from soup_cli.utils.raft import build_raft_prompt

            golden = build_raft_prompt(rows[0]).golden_doc_id
            return f"the answer is correct [{golden}]"

        result = score_citation(rows, gen)
        assert result.mode == "citation"
        assert result.score == pytest.approx(1.0)
        assert result.verdict == "OK"

    def test_score_citation_no_citation_major(self):
        from soup_cli.utils.diagnose.citation import score_citation

        rows = [_raft_row(2), _raft_row(2)]
        result = score_citation(rows, lambda p: "no citation at all")
        assert result.score == 0.0
        assert result.verdict == "MAJOR"

    def test_score_citation_no_raft_rows_raises(self):
        from soup_cli.utils.diagnose.citation import score_citation

        with pytest.raises(ValueError, match="RAFT"):
            score_citation([{"prompt": "p"}], lambda p: "x")

    def test_build_report_fills_citation_neutral(self):
        from soup_cli.utils.diagnose.runner import build_report

        report = build_report(run_id="r", base="b", adapter="a", scores={})
        assert "citation" in report.scores
        assert report.scores["citation"].verdict == "OK"

    def test_citation_probe_torch_free(self):
        import soup_cli.utils.diagnose.citation as cit

        with open(cit.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "\nimport torch" not in src


# ===========================================================================
# #201 — live CAA / ITI / RepE steering + serve --steer decode hook
# ===========================================================================


class TestSteeringMath:
    def test_caa_mean_difference(self):
        import numpy as np

        from soup_cli.utils.steering import compute_caa_vector

        pos = np.array([[2.0, 0.0], [4.0, 0.0]])  # mean [3,0]
        neg = np.array([[0.0, 1.0], [0.0, 3.0]])  # mean [0,2]
        vec = compute_caa_vector(pos, neg)
        assert np.allclose(vec, [3.0, -2.0])
        assert vec.dtype == np.float32

    def test_caa_dim_mismatch_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import compute_caa_vector

        with pytest.raises(ValueError, match="mismatch"):
            compute_caa_vector(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_caa_empty_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import compute_caa_vector

        with pytest.raises(ValueError):
            compute_caa_vector(np.zeros((0, 3)), np.zeros((2, 3)))

    def test_repe_direction_aligned(self):
        import numpy as np

        from soup_cli.utils.steering import compute_repe_direction

        # Diffs scattered along the +x axis → top PC ≈ x, sign-aligned positive.
        diffs = np.array([[2.0, 0.1], [3.0, -0.1], [4.0, 0.05]])
        vec = compute_repe_direction(diffs)
        assert vec.shape == (2,)
        assert vec[0] > 0  # points along the dominant +x diff direction

    def test_iti_selects_top_heads(self):
        import numpy as np

        from soup_cli.utils.steering import compute_iti_directions

        # 3 heads; head 1 has the biggest pos/neg separation.
        pos = np.zeros((2, 3, 2))
        neg = np.zeros((2, 3, 2))
        pos[:, 1, :] = 5.0  # head 1 strongly separated
        pos[:, 0, :] = 0.5
        dirs, selected = compute_iti_directions(pos, neg, top_k=1)
        assert selected == (1,)
        assert np.allclose(dirs[1], [5.0, 5.0])
        assert np.allclose(dirs[0], [0.0, 0.0])  # non-selected zeroed

    def test_iti_top_k_bool_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import compute_iti_directions

        with pytest.raises(ValueError, match="top_k"):
            compute_iti_directions(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), top_k=True)


class TestLoadContrastivePairs:
    def test_loads_pairs(self, tmp_path, monkeypatch):
        import json

        from soup_cli.utils.steering import load_contrastive_pairs

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "pairs.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"positive": "be kind", "negative": "be rude"}) + "\n")
            fh.write(json.dumps({"positive": "help", "negative": "refuse"}) + "\n")
            fh.write("not json\n")  # skipped
            fh.write(json.dumps({"positive": "x"}) + "\n")  # incomplete, skipped
        pairs = load_contrastive_pairs("pairs.jsonl")
        assert pairs == [("be kind", "be rude"), ("help", "refuse")]

    def test_outside_cwd_rejected(self, tmp_path):
        from soup_cli.utils.steering import load_contrastive_pairs

        outside = tmp_path / "pairs.jsonl"
        outside.write_text('{"positive":"a","negative":"b"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="cwd"):
            load_contrastive_pairs(str(outside))

    def test_empty_file_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.steering import load_contrastive_pairs

        monkeypatch.chdir(tmp_path)
        (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no usable"):
            load_contrastive_pairs("empty.jsonl")


class TestBuildSteeringVectorValidation:
    def test_method_validated_first(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError, match="steering method"):
            build_steering_vector(method="nonsense", name="x", base="m", pairs_path="p")

    def test_name_validated(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError, match="steering name"):
            build_steering_vector(method="caa", name="bad/path", base="m", pairs_path="p")

    def test_base_required(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError, match="base"):
            build_steering_vector(method="caa", name="ok", pairs_path="p")

    def test_pairs_required(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError, match="pairs_path"):
            build_steering_vector(method="caa", name="ok", base="m")


class TestSteeringArtifactRoundtrip:
    def _write_artifact(self, dir_path, *, method="caa", intervention="residual", vec=None):
        import json
        import os

        import numpy as np
        from safetensors.numpy import save_file

        os.makedirs(dir_path, exist_ok=True)
        if vec is None:
            vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        save_file({"vector": vec}, os.path.join(dir_path, "steering_vector.safetensors"))
        config = {
            "method": method,
            "name": "safety-v1",
            "layer": 1,
            "hidden_dim": int(vec.shape[0]),
            "intervention_point": intervention,
            "base": "tiny",
            "default_strength": 1.0,
        }
        with open(os.path.join(dir_path, "steering_config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh)

    def test_load_roundtrip(self, tmp_path, monkeypatch):
        import numpy as np

        from soup_cli.utils.steering import load_steering_artifact

        monkeypatch.chdir(tmp_path)
        self._write_artifact("steering/safety-v1")
        loaded = load_steering_artifact("steering/safety-v1")
        assert loaded.method == "caa"
        assert loaded.layer == 1
        assert loaded.intervention_point == "residual"
        assert np.allclose(loaded.vector, [0.1, 0.2, 0.3, 0.4])

    def test_resolve_steering_dir_local_fallback(self, tmp_path, monkeypatch):
        from soup_cli.utils.steering import resolve_steering_dir

        monkeypatch.chdir(tmp_path)
        self._write_artifact("steering/safety-v1")
        resolved = resolve_steering_dir("safety-v1")
        assert resolved.replace("\\", "/").endswith("steering/safety-v1")

    def test_resolve_unknown_raises(self, tmp_path, monkeypatch):
        from soup_cli.utils.steering import resolve_steering_dir

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="no steering vector"):
            resolve_steering_dir("does-not-exist")

    def test_load_missing_files_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.steering import load_steering_artifact

        monkeypatch.chdir(tmp_path)
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="missing"):
            load_steering_artifact("empty")


class TestInstallSteeringHook:
    def _fake_model(self, d=4, n=2):
        import torch
        import torch.nn as nn

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.o_proj = nn.Linear(d, d, bias=False)
                with torch.no_grad():
                    self.self_attn.o_proj.weight.copy_(torch.eye(d))
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return (x,)

        class FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer() for _ in range(n)])

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeInner()

        return FakeModel()

    def _loaded(self, intervention, vec):
        from soup_cli.utils.steering import LoadedSteering

        return LoadedSteering(
            method="caa" if intervention == "residual" else "iti",
            name="t",
            layer=0,
            intervention_point=intervention,
            vector=vec,
            default_strength=1.0,
        )

    def test_residual_hook_adds_vector(self):
        import numpy as np
        import torch

        from soup_cli.utils.steering import install_steering_hook

        model = self._fake_model(d=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        handle = install_steering_hook(model, self._loaded("residual", vec), strength=2.0)
        x = torch.zeros(1, 3, 4)
        out = model.model.layers[0](x)[0]
        # residual hook adds strength*vec = [2,0,0,0] to every position.
        assert torch.allclose(out[0, 0], torch.tensor([2.0, 0.0, 0.0, 0.0]))
        handle.remove()

    def test_iti_pre_hook_shifts_o_proj_input(self):
        import numpy as np
        import torch

        from soup_cli.utils.steering import install_steering_hook

        model = self._fake_model(d=4)
        vec = np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)
        handle = install_steering_hook(model, self._loaded("attn_o_proj_input", vec), strength=1.0)
        o_proj = model.model.layers[0].self_attn.o_proj
        x = torch.zeros(1, 2, 4)
        # o_proj is identity → output == shifted input == x + vec.
        out = o_proj(x)
        assert torch.allclose(out[0, 0], torch.tensor([0.0, 3.0, 0.0, 0.0]))
        handle.remove()

    def test_non_loaded_rejected(self):
        from soup_cli.utils.steering import install_steering_hook

        with pytest.raises(TypeError, match="LoadedSteering"):
            install_steering_hook(self._fake_model(), {"not": "loaded"}, strength=1.0)


class TestServeSteerCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_steer_flag_in_serve_help(self):
        from soup_cli.cli import app

        result = self._runner().invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--steer" in result.output

    def test_steer_requires_transformers_backend(self):
        from soup_cli.cli import app

        result = self._runner().invoke(
            app, ["serve", "-m", "model", "--steer", "safety-v1", "--backend", "vllm"]
        )
        assert result.exit_code == 2
        assert "transformers" in result.output

    def test_steer_bad_name_rejected(self):
        from soup_cli.cli import app

        result = self._runner().invoke(
            app, ["serve", "-m", "model", "--steer", "bad/name"]
        )
        assert result.exit_code == 2


class TestSteerCommandPlumbing:
    def test_steer_apply_loads_artifact(self, tmp_path, monkeypatch):
        import json
        import os

        import numpy as np
        from safetensors.numpy import save_file
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        monkeypatch.chdir(tmp_path)
        d = "steering/safety-v1"
        os.makedirs(d, exist_ok=True)
        save_file(
            {"vector": np.array([0.1, 0.2], dtype=np.float32)},
            os.path.join(d, "steering_vector.safetensors"),
        )
        with open(os.path.join(d, "steering_config.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "method": "caa", "name": "safety-v1", "layer": 1,
                "hidden_dim": 2, "intervention_point": "residual",
                "base": "tiny", "default_strength": 1.0,
            }, fh)
        result = CliRunner().invoke(app, ["apply", "--name", "safety-v1"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "Vector loaded" in result.output

    def test_steer_train_help_has_output(self):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        result = CliRunner().invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--top-k" in result.output


# ===========================================================================
# #200 — live RA-DIT two-stage orchestrator + retriever auto-link
# ===========================================================================


def _seed_registry(db_path, *, embedding_output, with_generator=False):
    """Push a retriever (embedding) entry whose config marks ra_dit_stage."""
    import os

    os.environ["SOUP_REGISTRY_DB_PATH"] = str(db_path)
    from soup_cli.registry.store import RegistryStore

    with RegistryStore() as store:
        # A non-RA-DIT embedding run (should NOT be picked).
        store.push(
            name="plain-embed",
            tag="v1",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            task="embedding",
            run_id=None,
            config={"task": "embedding", "output": "./other-embed"},
        )
        # The RA-DIT retriever stage (SHOULD be picked).
        rid = store.push(
            name="ra-dit-retriever",
            tag="v1",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            task="embedding",
            run_id=None,
            config={
                "task": "embedding",
                "output": embedding_output,
                "training": {"ra_dit_stage": "retriever"},
            },
        )
        if with_generator:
            store.push(
                name="ra-dit-gen",
                tag="v1",
                base_model="meta-llama/Llama-3.1-8B",
                task="sft",
                run_id=None,
                config={
                    "task": "sft",
                    "output": "./gen-out",
                    "training": {"ra_dit_stage": "generator"},
                },
            )
    return rid


class TestDiscoverLatestRetriever:
    def test_finds_ra_dit_retriever_output(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import discover_latest_retriever

        db = tmp_path / "reg.db"
        _seed_registry(db, embedding_output="./retriever-out")
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        found = discover_latest_retriever()
        assert found == "./retriever-out"

    def test_returns_none_when_no_retriever(self, tmp_path, monkeypatch):
        import os

        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.ra_dit_run import discover_latest_retriever

        db = tmp_path / "reg.db"
        os.environ["SOUP_REGISTRY_DB_PATH"] = str(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        with RegistryStore() as store:
            store.push(
                name="plain", tag="v1",
                base_model="m", task="embedding", run_id=None,
                config={"task": "embedding", "output": "./x"},
            )
        assert discover_latest_retriever() is None

    def test_empty_registry_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import discover_latest_retriever

        db = tmp_path / "reg.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        assert discover_latest_retriever() is None


class TestResolveRetrieverForGenerator:
    def test_manual_override_wins(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import resolve_retriever_for_generator

        db = tmp_path / "reg.db"
        _seed_registry(db, embedding_output="./auto-retriever")
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        resolved, advisory = resolve_retriever_for_generator("my/manual-retriever")
        assert resolved == "my/manual-retriever"
        assert "override" in advisory.lower() or "manual" in advisory.lower()

    def test_autolinks_when_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import resolve_retriever_for_generator

        db = tmp_path / "reg.db"
        _seed_registry(db, embedding_output="./auto-retriever")
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        resolved, advisory = resolve_retriever_for_generator(None)
        assert resolved == "./auto-retriever"
        assert "auto" in advisory.lower()

    def test_not_found_advisory(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import resolve_retriever_for_generator

        db = tmp_path / "reg.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        resolved, advisory = resolve_retriever_for_generator(None)
        assert resolved is None
        assert "no" in advisory.lower() and "retriever" in advisory.lower()


class TestRunRaDit:
    def _write_configs(self, tmp_path):
        retr = tmp_path / "retriever.yaml"
        retr.write_text(
            "base: sentence-transformers/all-MiniLM-L6-v2\n"
            "task: embedding\n"
            "output: ./ra-out/retriever\n"
            "training:\n  ra_dit_stage: retriever\n"
            "data:\n  train: ./triples.jsonl\n  format: embedding\n",
            encoding="utf-8",
        )
        gen = tmp_path / "generator.yaml"
        gen.write_text(
            "base: meta-llama/Llama-3.1-8B\n"
            "task: sft\n"
            "output: ./ra-out/generator\n"
            "training:\n  ra_dit_stage: generator\n"
            "data:\n  train: ./raft.jsonl\n  format: raft\n",
            encoding="utf-8",
        )
        return retr, gen

    def test_chains_two_stages_with_autolink(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import run_ra_dit

        monkeypatch.chdir(tmp_path)
        retr, gen = self._write_configs(tmp_path)
        calls = []

        def fake_runner(config_path):
            calls.append(config_path)

        result = run_ra_dit(
            "retriever.yaml", "generator.yaml", _runner=fake_runner
        )
        # Two subprocess stages ran, retriever first.
        assert len(calls) == 2
        assert "retriever" in calls[0].replace("\\", "/")
        # Generator stage ran via a rewritten temp yaml carrying the link.
        assert result.retriever_output.replace("\\", "/").endswith("ra-out/retriever")
        # Auto-linked retriever model == retriever output dir.
        assert result.retriever_model_used.replace("\\", "/").endswith(
            "ra-out/retriever"
        )
        assert result.autolinked is True

    def test_manual_override_skips_autolink(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import run_ra_dit

        monkeypatch.chdir(tmp_path)
        self._write_configs(tmp_path)
        result = run_ra_dit(
            "retriever.yaml",
            "generator.yaml",
            retriever_model="my/explicit-retriever",
            _runner=lambda p: None,
        )
        assert result.retriever_model_used == "my/explicit-retriever"
        assert result.autolinked is False

    def test_rewrites_generator_with_retriever_model(self, tmp_path, monkeypatch):
        import yaml

        from soup_cli.utils.ra_dit_run import run_ra_dit

        monkeypatch.chdir(tmp_path)
        self._write_configs(tmp_path)
        seen_yaml = {}

        def fake_runner(config_path):
            # On the 2nd call (generator), capture the rewritten YAML.
            with open(config_path, encoding="utf-8") as fh:
                seen_yaml[config_path] = yaml.safe_load(fh)

        run_ra_dit("retriever.yaml", "generator.yaml", _runner=fake_runner)
        # The generator temp yaml must carry the retriever model.
        gen_cfgs = [
            c for c in seen_yaml.values()
            if c.get("task") == "sft"
        ]
        assert gen_cfgs, "generator config not captured"
        training = gen_cfgs[0].get("training", {})
        assert "ra_dit_retriever_model" in training

    def test_outside_cwd_config_rejected(self, tmp_path):
        from soup_cli.utils.ra_dit_run import run_ra_dit

        retr, gen = self._write_configs(tmp_path)
        with pytest.raises(ValueError, match="cwd"):
            run_ra_dit(str(retr), str(gen), _runner=lambda p: None)


class TestTrainAutolinkHook:
    def test_generator_stage_autolinks_in_train(self, tmp_path, monkeypatch):
        """`soup train` of a generator stage with no retriever model auto-links."""
        import os

        from soup_cli.config.loader import load_config_from_string

        db = tmp_path / "reg.db"
        _seed_registry(db, embedding_output="./linked-retriever")
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        os.environ["SOUP_REGISTRY_DB_PATH"] = str(db)

        cfg = load_config_from_string(
            "base: meta-llama/Llama-3.1-8B\n"
            "task: sft\n"
            "output: ./out\n"
            "training:\n  ra_dit_stage: generator\n"
            "data:\n  train: ./raft.jsonl\n  format: raft\n"
        )
        assert cfg.training.ra_dit_retriever_model is None
        from soup_cli.utils.ra_dit_run import autolink_generator_retriever

        advisory = autolink_generator_retriever(cfg)
        assert cfg.training.ra_dit_retriever_model == "./linked-retriever"
        assert advisory is not None and "auto" in advisory.lower()

    def test_no_autolink_when_not_generator(self, tmp_path, monkeypatch):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.ra_dit_run import autolink_generator_retriever

        db = tmp_path / "reg.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        cfg = load_config_from_string(
            "base: m\ntask: sft\noutput: ./out\n"
            "data:\n  train: ./x.jsonl\n  format: chatml\n"
        )
        advisory = autolink_generator_retriever(cfg)
        assert advisory is None

    def test_manual_retriever_model_not_overwritten(self, tmp_path, monkeypatch):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.ra_dit_run import autolink_generator_retriever

        db = tmp_path / "reg.db"
        _seed_registry(db, embedding_output="./auto")
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        cfg = load_config_from_string(
            "base: m\ntask: sft\noutput: ./out\n"
            "training:\n  ra_dit_stage: generator\n"
            "  ra_dit_retriever_model: my/explicit\n"
            "data:\n  train: ./x.jsonl\n  format: raft\n"
        )
        advisory = autolink_generator_retriever(cfg)
        assert cfg.training.ra_dit_retriever_model == "my/explicit"
        assert advisory is None or "explicit" in advisory or "manual" in advisory.lower()


class TestRaDitCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_help(self):
        from soup_cli.commands.ra_dit import app

        result = self._runner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "retriever" in result.output.lower()

    def test_cli_registered(self):
        from soup_cli.cli import app

        result = self._runner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ra-dit" in result.output.lower()

    def test_plan_only(self, tmp_path, monkeypatch):
        from soup_cli.commands.ra_dit import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "retriever.yaml").write_text(
            "base: st/mini\ntask: embedding\noutput: ./r\n"
            "training:\n  ra_dit_stage: retriever\n"
            "data:\n  train: ./t.jsonl\n  format: embedding\n",
            encoding="utf-8",
        )
        (tmp_path / "generator.yaml").write_text(
            "base: meta-llama/Llama-3.1-8B\ntask: sft\noutput: ./g\n"
            "training:\n  ra_dit_stage: generator\n"
            "data:\n  train: ./raft.jsonl\n  format: raft\n",
            encoding="utf-8",
        )
        result = self._runner().invoke(app, [
            "--retriever-config", "retriever.yaml",
            "--generator-config", "generator.yaml",
            "--plan-only",
        ])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "retriever" in result.output.lower()
        assert "generator" in result.output.lower()

    def test_missing_config_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.ra_dit import app

        monkeypatch.chdir(tmp_path)
        result = self._runner().invoke(app, [
            "--retriever-config", "nope.yaml",
            "--generator-config", "alsonope.yaml",
            "--plan-only",
        ])
        assert result.exit_code == 2


class TestRaDitRunTorchFree:
    def test_module_torch_free(self):
        import soup_cli.utils.ra_dit_run as rr

        with open(rr.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "\nimport torch" not in src


# ===========================================================================
# v0.71.10 review-fix coverage (TDD wave) + regression guards for code fixes
# ===========================================================================


class TestValidateSteeringStrengthBounds:
    """#201 — validate_steering_strength boundary suite (HIGH)."""

    def test_zero_ok(self):
        from soup_cli.utils.steering import validate_steering_strength

        assert validate_steering_strength(0.0) == 0.0

    def test_upper_bound_ok(self):
        from soup_cli.utils.steering import validate_steering_strength

        assert validate_steering_strength(10.0) == 10.0

    def test_lower_bound_ok(self):
        from soup_cli.utils.steering import validate_steering_strength

        assert validate_steering_strength(-10.0) == -10.0

    def test_just_over_upper_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(ValueError, match="<="):
            validate_steering_strength(10.0001)

    def test_just_under_lower_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(ValueError, match="<="):
            validate_steering_strength(-10.0001)

    def test_nan_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(ValueError, match="finite"):
            validate_steering_strength(float("nan"))

    def test_inf_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(ValueError, match="finite"):
            validate_steering_strength(float("inf"))

    def test_bool_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(TypeError, match="bool"):
            validate_steering_strength(True)

    def test_non_number_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(TypeError, match="number"):
            validate_steering_strength("2.0")


class TestRepeDirectionSign:
    """#201 — RepE sign-alignment + degenerate-input rejection (HIGH)."""

    def test_sign_aligned_with_mean(self):
        import numpy as np

        from soup_cli.utils.steering import compute_repe_direction

        # Variance + mean both along -x → returned vector points -x AND its
        # projection onto the mean diff is non-negative (sign-aligned). This
        # invariant holds regardless of SVD's arbitrary sign choice.
        diffs = np.array([[-2.0, 0.0], [-4.0, 0.0], [-3.0, 0.0]])
        vec = compute_repe_direction(diffs)
        assert vec[0] < 0.0
        assert float(vec @ diffs.mean(axis=0)) >= 0.0

    def test_empty_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import compute_repe_direction

        with pytest.raises(ValueError):
            compute_repe_direction(np.zeros((0, 3)))

    def test_non_2d_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import compute_repe_direction

        with pytest.raises(ValueError):
            compute_repe_direction(np.zeros((3,)))


class TestBuildSteeringVectorPreLoadValidation:
    """#201 — layer / top_k validated BEFORE the model load (HIGH)."""

    def test_layer_out_of_range_rejected_pre_load(self):
        from soup_cli.utils.steering import build_steering_vector

        # layer=99999 fails the bounds check before load_contrastive_pairs /
        # the model load (base + pairs_path are never touched).
        with pytest.raises(ValueError, match="layer"):
            build_steering_vector(
                method="caa", name="ok", base="m", pairs_path="p", layer=99999
            )

    def test_layer_bool_rejected(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(TypeError, match="layer"):
            build_steering_vector(
                method="caa", name="ok", base="m", pairs_path="p", layer=True
            )

    def test_top_k_zero_rejected_pre_load(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError, match="top_k"):
            build_steering_vector(
                method="caa", name="ok", base="m", pairs_path="p", top_k=0
            )

    def test_top_k_bool_rejected_pre_load(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError, match="top_k"):
            build_steering_vector(
                method="caa", name="ok", base="m", pairs_path="p", top_k=True
            )

    def test_repe_requires_two_pairs(self, tmp_path, monkeypatch):
        import json

        from soup_cli.utils.steering import build_steering_vector

        monkeypatch.chdir(tmp_path)
        (tmp_path / "one.jsonl").write_text(
            json.dumps({"positive": "a", "negative": "b"}) + "\n", encoding="utf-8"
        )
        # repe needs >= 2 pairs — fails AFTER load_contrastive_pairs but BEFORE
        # the model load (regression for code-review L5).
        with pytest.raises(ValueError, match=">="):
            build_steering_vector(
                method="repe", name="ok", base="m", pairs_path="one.jsonl"
            )


class TestCitationBoostCap:
    """#199 — citation boost upper bound (MEDIUM)."""

    def test_boost_100_ok(self):
        from soup_cli.utils.raft import citation_span_token_weights

        weights = citation_span_token_weights(
            "[doc-0] x", [(0, 7), (8, 9)], boost=100.0
        )
        assert weights[0] == 100.0

    def test_boost_over_100_rejected(self):
        from soup_cli.utils.raft import citation_span_token_weights

        with pytest.raises(ValueError, match="boost"):
            citation_span_token_weights("x", [(0, 1)], boost=100.1)

    def test_tokenize_citation_boost_applied(self):
        from soup_cli.utils.raft import RaftComposed, tokenize_raft_example

        composed = RaftComposed(
            prompt="Q", answer="Paris [doc-0]", golden_doc_id="doc-0",
            doc_ids=("doc-0",),
        )
        row = tokenize_raft_example(
            _FakeTokenizer(fast=True), composed, max_length=64,
            citation_faithful=True, citation_boost=7.0,
        )
        # Some answer token carries the boost weight.
        assert 7.0 in row["loss_weights"]

    def test_tokenize_citation_boost_below_one_rejected(self):
        from soup_cli.utils.raft import RaftComposed, tokenize_raft_example

        composed = RaftComposed(
            prompt="Q", answer="Paris [doc-0]", golden_doc_id="doc-0",
            doc_ids=("doc-0",),
        )
        with pytest.raises(ValueError, match="boost"):
            tokenize_raft_example(
                _FakeTokenizer(fast=True), composed, max_length=64,
                citation_faithful=True, citation_boost=0.5,
            )


class TestRaftDocCap:
    """#199 — _MAX_DOCS + _MAX_FIELD_LEN boundaries (MEDIUM)."""

    def test_64_distractors_ok(self):
        from soup_cli.utils.raft import build_raft_prompt

        composed = build_raft_prompt(_raft_row(64))  # 1 golden + 64 = 65 = cap
        assert len(composed.doc_ids) == 65

    def test_65_distractors_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises(ValueError, match="documents"):
            build_raft_prompt(_raft_row(65))  # 66 > 65 cap

    def test_oversize_field_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        row = _raft_row(1)
        row["golden_doc"] = "x" * 70_000  # > _MAX_FIELD_LEN
        with pytest.raises(ValueError, match="chars"):
            build_raft_prompt(row)


class TestV07110FrozenDataclasses:
    """LOW — frozen invariant on the new v0.71.10 dataclasses."""

    def test_raft_composed_frozen(self):
        import dataclasses

        from soup_cli.utils.raft import RaftComposed

        c = RaftComposed(
            prompt="p", answer="a", golden_doc_id="doc-0", doc_ids=("doc-0",)
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.answer = "x"

    def test_steering_artifact_frozen(self):
        import dataclasses

        from soup_cli.utils.steering import SteeringArtifact

        a = SteeringArtifact(
            method="caa", name="n", layer=1, hidden_dim=4,
            intervention_point="residual", output_dir="d", base="b", num_pairs=2,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.layer = 9

    def test_loaded_steering_frozen(self):
        import dataclasses

        import numpy as np

        from soup_cli.utils.steering import LoadedSteering

        loaded = LoadedSteering(
            method="caa", name="n", layer=0, intervention_point="residual",
            vector=np.zeros(4, dtype=np.float32), default_strength=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            loaded.layer = 1

    def test_radit_run_result_frozen(self):
        import dataclasses

        from soup_cli.utils.ra_dit_run import RaDitRunResult

        r = RaDitRunResult(
            retriever_output="r", generator_output="g",
            retriever_model_used="r", autolinked=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.autolinked = False


class TestRunRaDitValidation:
    """#200 — timeout + oversize-yaml validation (LOW)."""

    def _configs(self, tmp_path):
        retr = tmp_path / "retriever.yaml"
        retr.write_text(
            "base: st/mini\ntask: embedding\noutput: ./r\n"
            "training:\n  ra_dit_stage: retriever\n"
            "data:\n  train: ./t.jsonl\n  format: embedding\n",
            encoding="utf-8",
        )
        gen = tmp_path / "generator.yaml"
        gen.write_text(
            "base: m\ntask: sft\noutput: ./g\n"
            "training:\n  ra_dit_stage: generator\n"
            "data:\n  train: ./raft.jsonl\n  format: raft\n",
            encoding="utf-8",
        )
        return retr, gen

    def test_bad_timeout_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import run_ra_dit

        monkeypatch.chdir(tmp_path)
        self._configs(tmp_path)
        with pytest.raises(ValueError, match="timeout"):
            run_ra_dit(
                "retriever.yaml", "generator.yaml",
                timeout_seconds=5, _runner=lambda p: None,
            )

    def test_timeout_bool_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import run_ra_dit

        monkeypatch.chdir(tmp_path)
        self._configs(tmp_path)
        with pytest.raises(ValueError, match="bool"):
            run_ra_dit(
                "retriever.yaml", "generator.yaml",
                timeout_seconds=True, _runner=lambda p: None,
            )

    def test_oversize_yaml_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.ra_dit_run import run_ra_dit

        monkeypatch.chdir(tmp_path)
        retr, _gen = self._configs(tmp_path)
        # Pad the retriever config past the 256KB cap with a comment line.
        retr.write_text(
            retr.read_text(encoding="utf-8") + "\n# " + "x" * (256 * 1024 + 10),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="cap|exceeds|KB"):
            run_ra_dit("retriever.yaml", "generator.yaml", _runner=lambda p: None)


class TestInstallSteeringHookExtras:
    """#201 — strength cap / bad layer / post-remove revert (LOW)."""

    def _fake_model(self, d=4, n=2):
        import torch
        import torch.nn as nn

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return (x,)

        class FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer() for _ in range(n)])

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeInner()

        return FakeModel()

    def _loaded(self, layer, vec):
        from soup_cli.utils.steering import LoadedSteering

        return LoadedSteering(
            method="caa", name="t", layer=layer,
            intervention_point="residual", vector=vec, default_strength=1.0,
        )

    def test_strength_cap_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import install_steering_hook

        model = self._fake_model()
        vec = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="<="):
            install_steering_hook(model, self._loaded(0, vec), strength=11.0)

    def test_bad_layer_rejected(self):
        import numpy as np

        from soup_cli.utils.steering import install_steering_hook

        model = self._fake_model(n=2)
        vec = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="out of range"):
            install_steering_hook(model, self._loaded(99, vec), strength=1.0)

    def test_remove_reverts(self):
        import numpy as np
        import torch

        from soup_cli.utils.steering import install_steering_hook

        model = self._fake_model(d=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        handle = install_steering_hook(model, self._loaded(0, vec), strength=2.0)
        x = torch.zeros(1, 1, 4)
        shifted = model.model.layers[0](x)[0]
        assert torch.allclose(shifted[0, 0], torch.tensor([2.0, 0.0, 0.0, 0.0]))
        handle.remove()
        reverted = model.model.layers[0](x)[0]
        assert torch.allclose(reverted[0, 0], torch.zeros(4))


class TestSteeringNoTopLevelTorch:
    def test_steering_no_top_level_torch(self):
        import soup_cli.utils.steering as steering

        with open(steering.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


class TestPrepareRaftDatasetExecution:
    """#199 — _prepare_raft_dataset map+filter real execution (MEDIUM)."""

    def _call(self, rows, *, max_length, citation=False):
        import types

        from soup_cli.trainer.sft import SFTTrainerWrapper

        stub = types.SimpleNamespace(tokenizer=_FakeTokenizer(fast=True))
        cfg = types.SimpleNamespace(
            data=types.SimpleNamespace(
                raft_shuffle_seed=None, max_length=max_length
            )
        )
        tcfg = types.SimpleNamespace(
            citation_faithful=citation, citation_style="bracket"
        )
        return SFTTrainerWrapper._prepare_raft_dataset(
            stub, {"train": rows}, cfg, tcfg
        )

    def test_keeps_trainable_rows(self):
        train, eval_ds = self._call([_raft_row(1)], max_length=512)
        assert len(train) == 1
        assert eval_ds is None
        assert any(w > 0.0 for w in train[0]["loss_weights"])

    def test_drops_all_masked_rows(self):
        # max_length=8 truncates the answer away → all-masked → dropped (M4).
        train, _eval = self._call([_raft_row(1), _raft_row(1)], max_length=8)
        assert len(train) == 0


class TestRaftComputeLossEdges:
    """#199 — compute_loss degenerate-zero + citation-boost shift (MEDIUM)."""

    def test_all_masked_returns_finite_zero(self):
        import torch

        from soup_cli.trainer.raft import make_raft_trainer_class

        cls = make_raft_trainer_class(_FakeBaseTrainer)
        trainer = cls()
        model = _FakeModel(torch.randn(1, 4, 10))
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, -100, -100]]),
            "loss_weights": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        }
        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0)

    def test_nan_logits_returns_structural_zero_with_grad(self):
        import torch

        from soup_cli.trainer.raft import make_raft_trainer_class

        cls = make_raft_trainer_class(_FakeBaseTrainer)
        trainer = cls()
        logits = torch.full((1, 4, 10), float("nan"), requires_grad=True)
        model = _FakeModel(logits)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, 5, 6]]),
            "loss_weights": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        }
        loss = trainer.compute_loss(model, inputs)
        # NaN forward → weighted mean NaN → structural zero (L2).
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0)
        assert loss.requires_grad

    def test_citation_boost_shifts_loss(self):
        import torch

        from soup_cli.trainer.raft import make_raft_trainer_class

        cls = make_raft_trainer_class(_FakeBaseTrainer)
        trainer = cls()
        torch.manual_seed(0)
        model = _FakeModel(torch.randn(1, 4, 10))
        labels = torch.tensor([[-100, -100, 5, 6]])
        flat = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": labels,
            "loss_weights": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        }
        boosted = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": labels,
            "loss_weights": torch.tensor([[0.0, 0.0, 1.0, 5.0]]),
        }
        l_flat = trainer.compute_loss(model, flat).item()
        l_boost = trainer.compute_loss(model, boosted).item()
        # Boosting one answer token's weight shifts the weighted mean.
        assert l_flat != pytest.approx(l_boost)


class TestV07110ReviewFixRegressions:
    """Regression guards for the v0.71.10 review-fix code changes."""

    def test_validate_ra_dit_config_path_public_and_alias(self):
        from soup_cli.utils import ra_dit_run

        assert hasattr(ra_dit_run, "validate_ra_dit_config_path")
        assert "validate_ra_dit_config_path" in ra_dit_run.__all__
        # back-compat private alias still points at the public function (M5).
        assert (
            ra_dit_run._validate_config_path
            is ra_dit_run.validate_ra_dit_config_path
        )

    def test_render_raft_prompt_public_and_alias(self):
        from soup_cli.utils import raft

        assert hasattr(raft, "render_raft_prompt")
        # back-compat private alias (L3).
        assert raft._render_prompt is raft.render_raft_prompt

    def test_discover_skips_corrupt_registry_output(self, tmp_path, monkeypatch):
        import os

        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.ra_dit_run import discover_latest_retriever

        db = tmp_path / "reg.db"
        os.environ["SOUP_REGISTRY_DB_PATH"] = str(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        with RegistryStore() as store:
            # A corrupt retriever row (oversize output > 512 chars) — the
            # discovered output is run through validate_ra_dit_retriever_model
            # (M1), so this row is skipped rather than flowing into a config.
            store.push(
                name="ra-dit-retriever-bad", tag="v1",
                base_model="st/mini", task="embedding", run_id=None,
                config={
                    "task": "embedding", "output": "x" * 600,
                    "training": {"ra_dit_stage": "retriever"},
                },
            )
        # No clean retriever row → discovery returns None, not the corrupt one.
        assert discover_latest_retriever() is None

    def test_load_yaml_config_o_nofollow(self):
        import soup_cli.utils.ra_dit_run as rr

        with open(rr.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "O_NOFOLLOW" in src
        assert "os.fstat" in src

    def test_load_steering_artifact_outside_cwd_rejected(self, tmp_path):
        from soup_cli.utils.steering import load_steering_artifact

        # An absolute out-of-cwd dir is rejected by the containment helper.
        with pytest.raises(ValueError, match="cwd"):
            load_steering_artifact(str(tmp_path / "elsewhere"))

    def test_train_autolink_source_grep(self):
        import soup_cli.commands.train as train_mod

        with open(train_mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "autolink_generator_retriever" in src
        # advisory is markup-escaped (SEC MED-1).
        assert "escape" in src

    def test_serve_steer_strength_source_grep(self):
        import soup_cli.commands.serve as serve_mod

        with open(serve_mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "validate_steering_strength" in src

    def test_serve_bad_steer_name_message(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["serve", "-m", "model", "--steer", "bad/name"]
        )
        assert result.exit_code == 2
        assert "Invalid --steer" in result.output
