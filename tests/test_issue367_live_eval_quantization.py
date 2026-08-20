"""#367 — live_eval.load_model_and_tokenizer took no quantization argument.

Every live evaluation path built on this helper (``soup ship`` leg 1/2,
``soup diagnose --base-model``, ``soup advise --probe-model``, ``soup eval
behavior``, ``tunability --live``) always loaded the base at full precision,
regardless of how the adapter was trained. An NF4-trained adapter was
therefore judged on a bf16 base it never saw during training.

Scoped to acceptance criteria 1 and 3 of the issue: threading the argument
through and a test that the reported numerics match the requested ones (here,
that ``from_pretrained`` actually receives the requested quantization_config).
Acceptance criteria 2 and 4 (``soup ship`` reporting the numerics used, and a
staleness gate on evidence recorded under mismatched numerics) are larger
integration work into ``soup ship``'s evidence machinery and are left open,
per the PR body.

Tests mock at the ``from_pretrained`` boundary, matching every other test in
this module's consumer chain (module docstring: "Tests mock at this
boundary... so the orchestration logic... is exercised without a GPU").
"""

from unittest.mock import MagicMock, patch

import pytest


def _fake_tokenizer():
    tok = MagicMock()
    tok.pad_token = "<pad>"
    return tok


class TestQuantizationReachesFromPretrained:
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_4bit_builds_an_nf4_config_and_pins_device_map(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        mock_model.return_value = MagicMock()

        load_model_and_tokenizer("some/model", device="cpu", quantization="4bit")

        _, kwargs = mock_model.call_args
        quant_config = kwargs["quantization_config"]
        assert quant_config.load_in_4bit is True
        assert quant_config.bnb_4bit_quant_type == "nf4"
        assert kwargs["device_map"] == "cpu"

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_8bit_builds_an_int8_config(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        mock_model.return_value = MagicMock()

        load_model_and_tokenizer("some/model", device="cpu", quantization="8bit")

        _, kwargs = mock_model.call_args
        assert kwargs["quantization_config"].load_in_8bit is True
        assert kwargs["device_map"] == "cpu"

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_unrecognised_quantization_is_rejected(self, mock_model, mock_tok):
        """Fail closed: the quant_menu formats that need a full TrainingConfig
        (gptq/awq/hqq/...) are explicitly out of scope here rather than
        silently loading at full precision."""
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        with pytest.raises(ValueError, match="not supported"):
            load_model_and_tokenizer("some/model", device="cpu", quantization="gptq")
        assert mock_model.call_count == 0


class TestBackwardsCompatibility:
    """The 11 existing callers pass no ``quantization`` argument at all; none
    of them may see a behavior change."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_unset_quantization_matches_the_pre_367_call_shape(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        fake_model = MagicMock()
        mock_model.return_value = fake_model

        load_model_and_tokenizer("some/model", device="cpu")

        _, kwargs = mock_model.call_args
        assert "quantization_config" not in kwargs
        assert "device_map" not in kwargs
        # Unquantized loads still move the model explicitly, as before #367.
        fake_model.to.assert_called_once_with("cpu")

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_none_and_the_string_none_are_both_unquantized(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        for value in (None, "none"):
            mock_model.reset_mock()
            fake_model = MagicMock()
            mock_model.return_value = fake_model
            load_model_and_tokenizer("some/model", device="cpu", quantization=value)
            assert "quantization_config" not in mock_model.call_args.kwargs


class TestQuantizedLoadIsNotMovedAfterward:
    """A quantized model is pinned to a device at ``from_pretrained`` time via
    ``device_map``; calling ``.to()`` on it afterward is what BNB rejects.
    This is the mechanism named in the fix's own comment, checked directly
    rather than trusted."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_to_is_not_called_on_a_4bit_load(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        fake_model = MagicMock()
        mock_model.return_value = fake_model

        load_model_and_tokenizer("some/model", device="cpu", quantization="4bit")

        fake_model.to.assert_not_called()


class TestBuildQuantizationConfigHelper:
    def test_none_and_literal_none_return_none(self):
        from soup_cli.utils.live_eval import _build_quantization_config

        assert _build_quantization_config(None) is None
        assert _build_quantization_config("none") is None

    def test_unsupported_value_raises_before_any_import(self):
        from soup_cli.utils.live_eval import _build_quantization_config

        with pytest.raises(ValueError, match="awq"):
            _build_quantization_config("awq")
