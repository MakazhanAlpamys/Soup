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
        assert quant_config.bnb_4bit_use_double_quant is True
        assert kwargs["device_map"] == "cpu"

    @patch("soup_cli.utils.gpu.get_compute_dtype")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_4bit_compute_dtype_comes_from_get_compute_dtype(
        self, mock_model, mock_tok, mock_get_dtype
    ):
        """The one wire connecting this fix to the #385/#387 T4 emulation-detect
        fix: bnb_4bit_compute_dtype must actually come from get_compute_dtype(),
        not a hardcoded value that happens to match on most hardware."""
        import torch

        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        mock_model.return_value = MagicMock()
        # BitsAndBytesConfig validates this field is a torch.dtype (or a string
        # naming one), so the sentinel has to be a real, distinctive dtype
        # rather than an opaque object -- float16 is never get_compute_dtype's
        # actual return value on any real hardware path, only bfloat16/float32.
        mock_get_dtype.return_value = torch.float16

        load_model_and_tokenizer("some/model", device="cpu", quantization="4bit")

        mock_get_dtype.assert_called_once_with()
        _, kwargs = mock_model.call_args
        assert kwargs["quantization_config"].bnb_4bit_compute_dtype is torch.float16

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_4bit_on_a_bare_cuda_device_gets_an_indexed_device_map(
        self, mock_model, mock_tok
    ):
        """A bare "cuda" has no index; accelerate's device_map resolution
        does torch.device(value).index next and raises a TypeError naming
        nothing the user could act on (the landmine layer_stream_runtime's
        _device_map_value already exists to avoid, reused here)."""
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        mock_model.return_value = MagicMock()

        with patch("torch.cuda.current_device", return_value=0):
            load_model_and_tokenizer("some/model", device="cuda", quantization="4bit")

        _, kwargs = mock_model.call_args
        assert kwargs["device_map"] == 0

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("peft.PeftModel.from_pretrained")
    def test_4bit_with_an_adapter_attaches_to_the_quantized_base(
        self, mock_peft, mock_model, mock_tok
    ):
        """Acceptance criterion 1: the base loads at the requested quantization
        AND the adapter attaches to it, in the same call."""
        from soup_cli.utils.live_eval import load_model_and_tokenizer

        mock_tok.return_value = _fake_tokenizer()
        base_model = MagicMock()
        mock_model.return_value = base_model
        adapted_model = MagicMock()
        mock_peft.return_value = adapted_model

        model, _, _ = load_model_and_tokenizer(
            "some/model", adapter="some/adapter", device="cpu", quantization="4bit"
        )

        _, kwargs = mock_model.call_args
        assert kwargs["quantization_config"].load_in_4bit is True
        mock_peft.assert_called_once_with(base_model, "some/adapter")
        assert model is adapted_model
        adapted_model.to.assert_not_called()
        base_model.to.assert_not_called()

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


# #367 checklist item 1, second half: the four internal callers must accept
# and forward ``quantization`` too. Criteria 2 and 4 stay out of scope, same
# as the original PR (see the module docstring above).


class TestTheFourCallersForwardQuantization:
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_make_generator_forwards_quantization(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import make_generator

        mock_tok.return_value = _fake_tokenizer()
        mock_model.return_value = MagicMock()

        make_generator("some/model", device="cpu", quantization="4bit")

        _, kwargs = mock_model.call_args
        assert kwargs["quantization_config"].load_in_4bit is True

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_make_multi_generator_forwards_quantization(self, mock_model, mock_tok):
        from soup_cli.utils.live_eval import make_multi_generator

        mock_tok.return_value = _fake_tokenizer()
        mock_model.return_value = MagicMock()

        make_multi_generator("some/model", device="cpu", quantization="8bit")

        _, kwargs = mock_model.call_args
        assert kwargs["quantization_config"].load_in_8bit is True

    def test_lora_probe_forwards_quantization(self):
        """Stops the probe at the load call (a RuntimeError side effect) since
        the LoRA train loop that follows is unrelated to this wiring fix."""
        from soup_cli.utils.live_eval import lora_probe

        with patch("soup_cli.utils.live_eval.load_model_and_tokenizer") as mock_load:
            mock_load.side_effect = RuntimeError("stop after load")
            with pytest.raises(RuntimeError, match="stop after load"):
                lora_probe(
                    "some/base",
                    [{"input": "a", "output": "b"}] * 3,
                    input_extractor=lambda r: r["input"],
                    output_extractor=lambda r: r["output"],
                    quantization="4bit",
                )
        assert mock_load.call_args.kwargs["quantization"] == "4bit"

    def test_measure_logit_agreement_forwards_quantization(self):
        from soup_cli.utils.live_eval import measure_logit_agreement

        with patch("soup_cli.utils.live_eval.load_model_and_tokenizer") as mock_load:
            mock_load.side_effect = RuntimeError("stop after load")
            with pytest.raises(RuntimeError, match="stop after load"):
                measure_logit_agreement(
                    "some/base",
                    [{"input": "a", "output": "b"}],
                    input_extractor=lambda r: r["input"],
                    output_extractor=lambda r: r["output"],
                    quantization="8bit",
                )
        assert mock_load.call_args.kwargs["quantization"] == "8bit"


class TestResolveGeneratorsThreadsQuantization:
    """Mirrors ``test_issue316_suites.py``'s ``TestTheBudgetReachesTheGenerator``:
    the wiring lives at the construction site, and mocking ``_verdict_live``
    (the CLI-level tests below) never actually reaches it."""

    def _capture(self, monkeypatch):
        from soup_cli.utils import live_eval

        calls = []

        def _fake(model_id, adapter=None, device=None, max_new_tokens=64, **kwargs):
            calls.append(kwargs.get("quantization"))
            return lambda prompt: ""

        monkeypatch.setattr(live_eval, "make_generator", _fake)
        return calls

    def test_adapter_branch_forwards_quantization(self, monkeypatch):
        from soup_cli.commands.ship import _resolve_generators

        calls = self._capture(monkeypatch)
        _resolve_generators(
            base="b", tuned=None, adapter="a", device="cpu", quantization="4bit"
        )
        assert calls == ["4bit", "4bit"]

    def test_tuned_branch_forwards_quantization(self, monkeypatch):
        from soup_cli.commands.ship import _resolve_generators

        calls = self._capture(monkeypatch)
        _resolve_generators(
            base="b", tuned="t", adapter=None, device="cpu", quantization="8bit"
        )
        assert calls == ["8bit", "8bit"]

    def test_unset_quantization_matches_the_pre_367_call_shape(self, monkeypatch):
        """No --config -> None reaches make_generator, same as every one of the
        11 pre-#367 callers that never passed the kwarg at all."""
        from soup_cli.commands.ship import _resolve_generators

        calls = self._capture(monkeypatch)
        _resolve_generators(base="b", tuned=None, adapter="a", device="cpu")
        assert calls == [None, None]

    def test_console_reports_the_numerics_used(self, monkeypatch, capsys):
        from soup_cli.commands.ship import _resolve_generators

        self._capture(monkeypatch)
        _resolve_generators(
            base="b", tuned=None, adapter="a", device="cpu", quantization="4bit"
        )
        assert "4bit" in capsys.readouterr().out

    def test_console_reports_the_bf16_fallback(self, monkeypatch, capsys):
        from soup_cli.commands.ship import _resolve_generators

        self._capture(monkeypatch)
        _resolve_generators(base="b", tuned=None, adapter="a", device="cpu")
        assert "bf16" in capsys.readouterr().out


class TestShipDerivesQuantizationFromConfig:
    """The only one of the four callers with an existing config surface to
    derive a default from: ``soup ship --config`` already loads the full
    ``SoupConfig`` used to train the adapter, including ``training.quantization``.

    ``advise --probe-model``, ``diagnose --base-model`` and ``tunability --live``
    take no ``--config``/``--quantization`` flag today, so they keep the
    unchanged bf16 default; wiring them needs a new config surface on each of
    those commands, which is follow-up work, not part of this fix.
    """

    def test_returns_none_without_a_config(self):
        from soup_cli.commands.ship import _live_eval_quantization_from_config

        assert _live_eval_quantization_from_config(None) is None

    def test_returns_the_supported_format(self):
        from soup_cli.commands.ship import _live_eval_quantization_from_config
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="m", data={"train": "t.jsonl"}, training={"quantization": "4bit"}
        )
        assert _live_eval_quantization_from_config(cfg) == "4bit"

    def test_unsupported_quant_menu_format_falls_back_to_none(self):
        """gptq/awq/hqq/... key off a full TrainingConfig live_eval callers
        don't have, so a run trained that way still loads bf16 here, same as
        the original PR's `_build_quantization_config` scope."""
        from soup_cli.commands.ship import _live_eval_quantization_from_config
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="m", data={"train": "t.jsonl"}, training={"quantization": "gptq"}
        )
        assert _live_eval_quantization_from_config(cfg) is None

    def test_ship_cli_threads_config_quantization_to_the_live_verdict(self, monkeypatch):
        """End-to-end through the CLI: --config's training.quantization reaches
        _verdict_live, exactly like the existing task_eval/noise_floor threading
        this file's sibling tests (test_v07139.py) already cover."""
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
        )

        captured = {}

        def _fake_live(**kwargs):
            captured.update(kwargs)
            win = build_task_win("metric", 0.5, 0.7)
            deltas = compute_benchmark_deltas({"b": 0.6}, {"b": 0.6})
            return decide_ship(win, deltas)

        monkeypatch.setattr(ship_cmd, "_verdict_live", _fake_live)
        cfg = "base: m\ndata:\n  train: t.jsonl\ntraining:\n  quantization: 4bit\n"
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(cfg, encoding="utf-8")
            res = runner.invoke(
                ship_cmd.app,
                ["--base", "m", "--adapter", "a", "--config", "soup.yaml"],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert captured["quantization"] == "4bit"

    def test_ship_cli_without_config_passes_none(self, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
        )

        captured = {}

        def _fake_live(**kwargs):
            captured.update(kwargs)
            win = build_task_win("metric", 0.5, 0.7)
            deltas = compute_benchmark_deltas({"b": 0.6}, {"b": 0.6})
            return decide_ship(win, deltas)

        monkeypatch.setattr(ship_cmd, "_verdict_live", _fake_live)
        runner = CliRunner()
        res = runner.invoke(ship_cmd.app, ["--base", "m", "--adapter", "a"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert captured["quantization"] is None

    def test_ship_cli_reaches_resolve_generators_through_the_real_verdict_live(
        self, monkeypatch
    ):
        """The two tests above mock `_verdict_live` itself, so they never run
        the line inside it that forwards `quantization` on to
        `_resolve_generators`. This one leaves `_verdict_live` real (mocking
        only `_resolve_generators`, same as `test_v07138.py`'s
        `TestShipLiveHeadline._run`) so that call site is actually exercised."""
        import json
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.commands import ship as ship_cmd

        captured = {}

        def _fake_resolve(base, tuned, adapter, device, quantization=None):
            captured["quantization"] = quantization
            return (lambda p: "hi", lambda p: "hi")

        monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
        cfg = "base: m\ndata:\n  train: t.jsonl\ntraining:\n  quantization: 4bit\n"
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(cfg, encoding="utf-8")
            Path("task.jsonl").write_text(
                json.dumps({"prompt": "say hi", "expected": "hi", "scoring": "contains"})
                + "\n",
                encoding="utf-8",
            )
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "m", "--adapter", "a", "--task-eval", "task.jsonl",
                    "--device", "cpu", "--config", "soup.yaml",
                ],
            )
            assert res.exit_code in (0, 2), (res.output, repr(res.exception))
            assert captured["quantization"] == "4bit"
