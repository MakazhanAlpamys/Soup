"""Tests for GitHub issue #304: cross-tokenizer drafts.

Covers:
1. same_tokenizer regression
2. Cross-tokenizer span-based acceptance kernel
3. Acceptance test that catches raw-ID comparison bugs
4. Cross-tokenizer distillation config (_build_distill_config_yaml with uld_strategy)
5. Same-tokenizer distill regression
6. Universal Assisted Decoding (UAD) capability detection and tokenizer plumbing
7. Unsupported transformers version / capability error path
8. Serve integration (_load_draft_tokenizer, _create_app, _generate_response)
"""

from typing import Sequence
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class _FakeTok:
    def __init__(
        self,
        vocab_size: int = 32000,
        encode_map: dict | None = None,
        decode_map: dict | None = None,
    ):
        self.vocab_size = vocab_size
        self._encode_map = encode_map or {}
        self._decode_map = decode_map or {}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self._encode_map:
            return list(self._encode_map[text])
        return [1, 2, 3]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        pieces = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int in self._decode_map:
                pieces.append(self._decode_map[tid_int])
            else:
                pieces.append(f"<tok_{tid_int}>")
        return "".join(pieces)

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs):
        import torch

        ids = self.encode(text)
        tensor_ids = torch.tensor([ids])
        return {"input_ids": tensor_ids, "attention_mask": torch.ones_like(tensor_ids)}


# ---------------------------------------------------------------------------
# PART 1: same_tokenizer regression
# ---------------------------------------------------------------------------
class TestSameTokenizerRegression:
    def test_identical_tokenizer_is_true(self):
        from soup_cli.utils.draft import same_tokenizer

        tok = _FakeTok(32000)
        assert same_tokenizer(tok, tok) is True
        assert same_tokenizer(_FakeTok(32000), _FakeTok(32000)) is True

    def test_different_vocab_size_is_false(self):
        from soup_cli.utils.draft import same_tokenizer

        assert same_tokenizer(_FakeTok(32000), _FakeTok(49152)) is False

    def test_same_vocab_size_differing_ids_is_false(self):
        from soup_cli.utils.draft import PROBE_CORPUS, same_tokenizer

        probe = PROBE_CORPUS[0]
        tok_a = _FakeTok(32000, {probe: [10, 20, 30]})
        tok_b = _FakeTok(32000, {probe: [99, 88, 77]})
        assert same_tokenizer(tok_a, tok_b) is False

    def test_broken_tokenizer_is_false(self):
        from soup_cli.utils.draft import same_tokenizer

        class _Broken:
            vocab_size = 32000

            def encode(self, text, add_special_tokens=False):
                raise RuntimeError("crash")

        assert same_tokenizer(_FakeTok(32000), _Broken()) is False


# ---------------------------------------------------------------------------
# PART 2: Cross-tokenizer acceptance kernel (pure functions)
# ---------------------------------------------------------------------------
class TestCrossTokenizerAcceptanceKernel:
    def test_empty_pieces_scores_zero(self):
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        assert count_accepted_spans([], ["hello"]) == 0
        assert count_accepted_spans(["hello"], []) == 0
        assert count_accepted_spans([], []) == 0
        assert compute_acceptance_spans([], ["hello"]) == 0.0
        assert compute_acceptance_spans(["hello"], []) == 0.0

    def test_identical_pieces_scores_full(self):
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        draft = ["The", " quick", " brown", " fox"]
        target = ["The", " quick", " brown", " fox"]
        assert count_accepted_spans(draft, target) == 4
        assert compute_acceptance_spans(draft, target) == 1.0

    def test_different_tokenization_boundaries_same_text_scores_full(self):
        """When draft and target split identical text at different boundaries,
        acceptance must be 100% (all target tokens are matched)."""
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        target = ["Hello", " world", "!"]
        draft = ["Hel", "lo", " ", "world", "!"]
        assert count_accepted_spans(draft, target) == 3
        assert compute_acceptance_spans(draft, target) == 1.0

    def test_partial_match_scores_correctly_not_trivially_one(self):
        """Demonstrates that non-matching spans are rejected."""
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        target = ["The", " quick", " brown", " fox"]
        draft = ["The", " fast", " brown", " fox"]
        # Target tokens: "The" (match), " quick" (mismatch), " brown" (match), " fox" (match)
        assert count_accepted_spans(draft, target) == 3
        assert compute_acceptance_spans(draft, target) == 0.75

    def test_completely_different_text_scores_zero(self):
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        target = ["apples", " and", " oranges"]
        draft = ["completely", " different", " words"]
        assert count_accepted_spans(draft, target) == 0
        assert compute_acceptance_spans(draft, target) == 0.0

    def test_zero_length_and_unicode_tokens(self):
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        target = ["", "你好", " ", "мир"]
        draft = ["你好", " ", "мир"]
        assert count_accepted_spans(draft, target) == 4
        assert compute_acceptance_spans(draft, target) == 1.0

    def test_boundary_merge_bias_is_pinned(self):
        """Regression test for #462: a merged prompt/generation boundary token
        costs exactly 1/n_gen, always downward."""
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        target = ["b", "c"]
        # control: clean boundary
        draft_clean = ["b", "c"]
        assert count_accepted_spans(draft_clean, target) == 2
        assert compute_acceptance_spans(draft_clean, target) == 1.0

        # merged boundary (b + c -> 'bc')
        draft_merged = ["bc"]
        assert count_accepted_spans(draft_merged, target) == 1
        assert compute_acceptance_spans(draft_merged, target) == 0.5

    def test_repeated_substrings_and_interrupted_spans(self):
        """Verify repeated words and interrupted spans do not produce false positives."""
        from soup_cli.utils.draft import compute_acceptance_spans, count_accepted_spans

        # Repeated words: "foo bar foo" vs "foo baz foo"
        target = ["foo", " bar", " foo"]
        draft = ["foo", " baz", " foo"]
        assert count_accepted_spans(draft, target) == 2
        assert compute_acceptance_spans(draft, target) == pytest.approx(2 / 3)

        # Interrupted span: draft inserts text in middle of target token
        target_interrupted = ["start", "unbroken", "end"]
        draft_interrupted = ["start", "un", "EXTRA", "broken", "end"]
        # "unbroken" is interrupted, so only "start" and "end" match
        assert count_accepted_spans(draft_interrupted, target_interrupted) == 2


# ---------------------------------------------------------------------------
# PART 3: CRITICAL REGRESSION TEST: Catches raw-ID comparison bugs
# ---------------------------------------------------------------------------
class TestRawTokenIdComparisonRegression:
    """This test MUST fail if the cross-tokenizer implementation simply compared
    raw token IDs.

    Setup:
    - Target vocabulary: Llama-like (token IDs: 101, 102, 103 for "def", " foo", "():")
    - Draft vocabulary: Qwen/GPT-like (token IDs: 501, 502, 503, 504 for "de", "f", " foo", "():")
    - Text: "def foo():"

    Target token IDs != Draft token IDs.
    Target token count (3) != Draft token count (4).

    Raw token ID comparison (count_accepted) FAILS:
    - If called directly with draft vs target IDs, raises ValueError for mismatched lengths.
    - If length padded/sliced, matches 0 tokens because 101 != 501, etc.

    Decoded span comparison (count_accepted_spans) SUCCEEDS:
    - Returns 3 accepted out of 3 target tokens (100% acceptance).
    """

    def test_cross_tokenizer_id_mismatch_succeeds_where_raw_ids_fail(self):
        from soup_cli.utils.draft import (
            compute_acceptance_spans,
            count_accepted,
            count_accepted_spans,
        )

        target_ids = [101, 102, 103]
        draft_ids = [501, 502, 503, 504]

        # Raw token ID comparison fails due to length difference
        with pytest.raises(ValueError, match="same length"):
            count_accepted(draft_ids, target_ids)

        # Raw token ID comparison with equal lengths also fails to match anything
        draft_ids_same_len = [501, 502, 503]
        assert count_accepted(draft_ids_same_len, target_ids) == 0

        # Decoded pieces correspond to the same text
        target_pieces = ["def", " foo", "():"]
        draft_pieces = ["de", "f", " foo", "():"]

        accepted = count_accepted_spans(draft_pieces, target_pieces)
        rate = compute_acceptance_spans(draft_pieces, target_pieces)

        assert accepted == 3
        assert rate == 1.0

    def test_cross_tokenizer_measure_acceptance_e2e(self):
        """Test measure_acceptance end-to-end with mismatched tokenizers."""
        import torch

        from soup_cli.utils.draft import measure_acceptance

        # Target tokenizer: produces 2 tokens for "hello world" -> [10, 20]
        # Decodes 10 -> "hello", 20 -> " world"
        target_tok = _FakeTok(
            vocab_size=32000,
            encode_map={"prompt": [1, 2], "prompthello world": [1, 2, 10, 20]},
            decode_map={1: "pr", 2: "ompt", 10: "hello", 20: " world"},
        )

        # Draft tokenizer: produces 3 tokens for "hello world" -> [100, 200, 300]
        # Decodes 100 -> "hel", 200 -> "lo", 300 -> " world"
        draft_tok = _FakeTok(
            vocab_size=49152,
            encode_map={"prompt": [50], "prompthello world": [50, 100, 200, 300]},
            decode_map={50: "prompt", 100: "hel", 200: "lo", 300: " world"},
        )

        class _MockTarget:
            def parameters(self):
                yield torch.zeros(1)

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 10, 20]])

        class _MockDraft:
            def parameters(self):
                yield torch.zeros(1)

            def __call__(self, input_ids=None, **kwargs):
                seq = int(input_ids.shape[1])
                logits = torch.zeros(1, seq, 49152)
                preds = [100, 200, 300]
                for i, tok in enumerate(preds):
                    if i < seq:
                        logits[0, i, tok] = 10.0

                class _Out:
                    pass

                out = _Out()
                out.logits = logits
                return out

        accepted, total = measure_acceptance(
            _MockTarget(),
            _MockDraft(),
            target_tok,
            ["prompt"],
            max_new_tokens=4,
            draft_tokenizer=draft_tok,
        )

        # Both target tokens ("hello", " world") were accurately reconstructed by draft!
        assert total == 2
        assert accepted == 2


# ---------------------------------------------------------------------------
# PART 4: Distill configuration tests (same-tokenizer vs cross-tokenizer)
# ---------------------------------------------------------------------------
class TestDistillConfigCrossTokenizer:
    def test_same_tokenizer_distill_config_no_uld_strategy(self):
        from soup_cli.commands.draft import _build_distill_config_yaml
        from soup_cli.config.loader import load_config_from_string

        yaml_text = _build_distill_config_yaml(
            draft_base="org/tiny",
            target="org/target",
            data="d.jsonl",
            out_dir="draftout",
            steps=100,
            data_rows=200,
            uld_strategy=None,
        )
        assert "uld_strategy" not in yaml_text
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.uld_strategy is None
        assert cfg.task == "distill"

    def test_cross_tokenizer_distill_config_has_wasserstein_aligned(self):
        from soup_cli.commands.draft import _build_distill_config_yaml
        from soup_cli.config.loader import load_config_from_string

        yaml_text = _build_distill_config_yaml(
            draft_base="org/tiny",
            target="org/target",
            data="d.jsonl",
            out_dir="draftout",
            steps=100,
            data_rows=200,
            uld_strategy="wasserstein_aligned",
        )
        assert "uld_strategy: wasserstein_aligned" in yaml_text
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.uld_strategy == "wasserstein_aligned"
        assert cfg.task == "distill"

    def test_distill_cli_auto_selects_wasserstein_aligned_on_mismatched_vocab(
        self, tmp_path, monkeypatch
    ):
        import json

        from typer.testing import CliRunner

        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        runner = CliRunner()
        d_file = tmp_path / "d.jsonl"
        d_file.write_text(
            "\n".join(json.dumps({"prompt": "hi", "response": "hello"}) for _ in range(200))
            + "\n"
        )

        # Monkeypatch vocab sizes to differ
        def _fake_vocab(model_id: str, trc: bool = False) -> int:
            return 32000 if "target" in model_id else 49152

        monkeypatch.setattr(draft_cmd, "_vocab_size_of", _fake_vocab)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "distill",
                "--target", "org/target",
                "--draft-base", "org/tiny",
                "--data", str(d_file),
                "-o", "draftout",
                "--plan-only",
            ],
        )

        assert result.exit_code == 0
        assert "uld_strategy: wasserstein_aligned" in result.output
        assert "cross-tokenizer" in result.output.lower()

    def test_distill_cli_same_vocab_differing_ids_selects_wasserstein_aligned(
        self, tmp_path, monkeypatch
    ):
        """When vocab size is equal but tokenizers differ, distill routes to ULD."""
        import json

        from typer.testing import CliRunner

        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        runner = CliRunner()
        d_file = tmp_path / "d.jsonl"
        d_file.write_text(
            "\n".join(json.dumps({"prompt": "hi", "response": "hello"}) for _ in range(200))
            + "\n"
        )

        # Equal vocab size
        monkeypatch.setattr(draft_cmd, "_vocab_size_of", lambda m, trc=False: 32000)

        # But differing tokenizers on probe corpus
        tok_a = _FakeTok(32000, {"The quick brown fox jumps over 13 lazy dogs.": [1, 2, 3]})
        tok_b = _FakeTok(32000, {"The quick brown fox jumps over 13 lazy dogs.": [9, 8, 7]})

        def _fake_tok_from_pretrained(m, **kwargs):
            return tok_a if "target" in m else tok_b

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", _fake_tok_from_pretrained
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "distill",
                "--target", "org/target",
                "--draft-base", "org/tiny",
                "--data", str(d_file),
                "-o", "draftout",
                "--plan-only",
            ],
        )

        assert result.exit_code == 0
        assert "uld_strategy: wasserstein_aligned" in result.output

    def test_distill_cli_tokenizer_load_failure_raises_error(
        self, tmp_path, monkeypatch
    ):
        """When tokenizers cannot be loaded to verify compatibility, distill fails."""
        import json

        from typer.testing import CliRunner

        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        runner = CliRunner()
        d_file = tmp_path / "d.jsonl"
        d_file.write_text(
            "\n".join(json.dumps({"prompt": "hi", "response": "hello"}) for _ in range(200))
            + "\n"
        )

        monkeypatch.setattr(draft_cmd, "_vocab_size_of", lambda m, trc=False: 32000)

        def _boom(*args, **kwargs):
            raise OSError("tokenizer files corrupted")

        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _boom)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "distill",
                "--target", "org/target",
                "--draft-base", "org/tiny",
                "--data", str(d_file),
                "-o", "draftout",
                "--plan-only",
            ],
        )

        assert result.exit_code == 1
        assert "could not verify tokenizer compatibility" in result.output.lower()


# ---------------------------------------------------------------------------
# PART 5: Universal Assisted Decoding (UAD) plumbing and error paths
# ---------------------------------------------------------------------------
class TestUniversalAssistedDecodingPlumbing:
    def test_supports_uad_returns_bool(self):
        from soup_cli.utils.draft import supports_universal_assisted_decoding

        supported = supports_universal_assisted_decoding()
        assert isinstance(supported, bool)

    def test_measure_throughput_passes_uad_tokenizers_when_mismatched(self):
        import torch

        from soup_cli.utils.draft import measure_throughput

        mock_target = MagicMock()
        mock_target.parameters.side_effect = lambda: iter([torch.zeros(1)])
        mock_target.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        target_tok = _FakeTok(32000)
        draft_tok = _FakeTok(49152)
        mock_draft = MagicMock()

        with mock_patch(
            "soup_cli.utils.draft.supports_universal_assisted_decoding", return_value=True
        ):
            tok_s = measure_throughput(
                mock_target,
                target_tok,
                ["prompt"],
                assistant_model=mock_draft,
                assistant_tokenizer=draft_tok,
                max_new_tokens=4,
            )
            assert tok_s >= 0.0
            call_kwargs = mock_target.generate.call_args[1]
            assert call_kwargs["assistant_model"] == mock_draft
            assert call_kwargs["tokenizer"] == target_tok
            assert call_kwargs["assistant_tokenizer"] == draft_tok

    def test_measure_throughput_same_tokenizer_passes_no_assistant_tokenizer(self):
        import torch

        from soup_cli.utils.draft import measure_throughput

        mock_target = MagicMock()
        mock_target.parameters.side_effect = lambda: iter([torch.zeros(1)])
        mock_target.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        target_tok = _FakeTok(32000)
        draft_tok = _FakeTok(32000)
        mock_draft = MagicMock()

        tok_s = measure_throughput(
            mock_target,
            target_tok,
            ["prompt"],
            assistant_model=mock_draft,
            assistant_tokenizer=draft_tok,
            max_new_tokens=4,
        )
        assert tok_s >= 0.0
        call_kwargs = mock_target.generate.call_args[1]
        assert call_kwargs["assistant_model"] == mock_draft
        assert "assistant_tokenizer" not in call_kwargs

    def test_measure_throughput_unsupported_transformers_raises_friendly_error(self):
        import torch

        from soup_cli.utils.draft import measure_throughput

        mock_target = MagicMock()
        mock_target.parameters.side_effect = lambda: iter([torch.zeros(1)])

        target_tok = _FakeTok(32000)
        draft_tok = _FakeTok(49152)
        mock_draft = MagicMock()

        with mock_patch(
            "soup_cli.utils.draft.supports_universal_assisted_decoding", return_value=False
        ):
            with pytest.raises(RuntimeError, match="Universal Assisted Decoding"):
                measure_throughput(
                    mock_target,
                    target_tok,
                    ["prompt"],
                    assistant_model=mock_draft,
                    assistant_tokenizer=draft_tok,
                    max_new_tokens=4,
                )


# ---------------------------------------------------------------------------
# PART 6: Serve integration tests
# ---------------------------------------------------------------------------
class TestServeIntegrationCrossTokenizer:
    def test_load_draft_tokenizer_success(self):
        from soup_cli.commands.serve import _load_draft_tokenizer

        mock_tok = MagicMock()
        with mock_patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
            res = _load_draft_tokenizer("some-model", trust_remote_code=False)
            assert res == mock_tok

    def test_load_draft_tokenizer_blocks_urls(self):
        from soup_cli.commands.serve import _load_draft_tokenizer

        res = _load_draft_tokenizer("http://evil.com/model")
        assert res is None

    def test_load_draft_tokenizer_oserror_returns_none(self):
        from soup_cli.commands.serve import _load_draft_tokenizer

        with mock_patch(
            "transformers.AutoTokenizer.from_pretrained", side_effect=OSError("not found")
        ):
            res = _load_draft_tokenizer("missing-tokenizer-model")
            assert res is None

    def test_generate_response_threads_uad_tokenizers(self):
        import torch

        from soup_cli.commands.serve import _generate_response

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        target_tok = _FakeTok(32000)
        draft_tok = _FakeTok(49152)
        mock_draft = MagicMock()

        with mock_patch(
            "soup_cli.utils.draft.supports_universal_assisted_decoding", return_value=True
        ):
            _generate_response(
                mock_model,
                target_tok,
                [{"role": "user", "content": "hello"}],
                max_tokens=10,
                assistant_model=mock_draft,
                assistant_tokenizer=draft_tok,
                num_assistant_tokens=3,
            )

            gen_call = mock_model.generate.call_args[1]
            assert gen_call["assistant_model"] == mock_draft
            assert gen_call["tokenizer"] == target_tok
            assert gen_call["assistant_tokenizer"] == draft_tok

    def test_generate_response_same_tokenizer_does_not_pass_assistant_tokenizer(self):
        import torch

        from soup_cli.commands.serve import _generate_response

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        tok = _FakeTok(32000)
        mock_draft = MagicMock()

        _generate_response(
            mock_model,
            tok,
            [{"role": "user", "content": "hello"}],
            max_tokens=10,
            assistant_model=mock_draft,
            assistant_tokenizer=tok,
            num_assistant_tokens=3,
        )

        gen_call = mock_model.generate.call_args[1]
        assert gen_call["assistant_model"] == mock_draft
        assert "assistant_tokenizer" not in gen_call
        assert "tokenizer" not in gen_call

    def test_generate_response_unsupported_uad_raises_error(self):
        from soup_cli.commands.serve import _generate_response

        mock_model = MagicMock()
        mock_model.device = "cpu"

        target_tok = _FakeTok(32000)
        draft_tok = _FakeTok(49152)
        mock_draft = MagicMock()

        with mock_patch(
            "soup_cli.utils.draft.supports_universal_assisted_decoding", return_value=False
        ):
            with pytest.raises(RuntimeError, match="Universal Assisted Decoding"):
                _generate_response(
                    mock_model,
                    target_tok,
                    [{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    assistant_model=mock_draft,
                    assistant_tokenizer=draft_tok,
                )
