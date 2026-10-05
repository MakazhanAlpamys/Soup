"""#1448 - `soup train --profile` says it is writing a trace, and writes none.

The command enters the `profile_training` context around training, but
the profiler was never bound or stepped, and nothing passed it to
`trainer_wrapper.train()`. `torch.profiler`'s schedule advances only on
`step()`, so the trace was never written.

This file pins:
* `ProfileStepCallback` steps the profiler on `on_step_end` and swallows errors;
* `build_profile_callback` builds the callback or returns None when profiler is None;
* `soup train --profile` writes a valid Chrome trace JSON file in `<output>/profiles/`;
* `soup train` without `--profile` creates no `profiles/` directory;
* Mutation check: dropping the profiler callback wiring produces no trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.profiling import (
    ProfileStepCallback,
    build_profile_callback,
)


def _tiny_llama_dir(root: Path, vocab: int = 64, hidden: int = 64) -> str:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = root / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))

    raw = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(vocab)}, unk_token="t0"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="t1",
        eos_token="t2",
        pad_token="t0",
        unk_token="t0",
    )
    tok.save_pretrained(str(weights))
    return str(weights)


class TestProfileStepCallback:
    def test_callback_steps_profiler(self):
        profiler = MagicMock()
        cb = ProfileStepCallback(profiler)
        control = MagicMock()

        res = cb.on_step_end(args=None, state=None, control=control)

        assert profiler.step.call_count == 1
        assert res is control

    def test_callback_swallows_step_exception(self):
        profiler = MagicMock()
        profiler.step.side_effect = RuntimeError("step failure")
        cb = ProfileStepCallback(profiler)
        control = MagicMock()

        res = cb.on_step_end(args=None, state=None, control=control)

        assert profiler.step.call_count == 1
        assert res is control

    def test_callback_handles_none_or_missing_step(self):
        cb_none = ProfileStepCallback(None)
        res = cb_none.on_step_end(args=None, state=None, control="ctrl")
        assert res == "ctrl"

        cb_obj = ProfileStepCallback(object())
        res2 = cb_obj.on_step_end(args=None, state=None, control="ctrl2")
        assert res2 == "ctrl2"

    def test_build_profile_callback(self):
        assert build_profile_callback(None) is None

        profiler = MagicMock()
        cb = build_profile_callback(profiler)
        assert isinstance(cb, ProfileStepCallback)
        assert cb.profiler is profiler

    def test_attach_registers_on_the_hf_trainer(self):
        from types import SimpleNamespace

        from soup_cli.utils.profiling import attach_profile_callback

        hf_trainer, profiler = MagicMock(), MagicMock()
        assert attach_profile_callback(SimpleNamespace(trainer=hf_trainer), profiler) is True
        (callback,), _ = hf_trainer.add_callback.call_args
        assert isinstance(callback, ProfileStepCallback)
        assert callback.profiler is profiler

    def test_attach_reports_a_wrapper_without_a_trainer(self):
        from types import SimpleNamespace

        from soup_cli.utils.profiling import attach_profile_callback

        assert attach_profile_callback(SimpleNamespace(trainer=None), MagicMock()) is False

    def test_callback_logs_warning_on_step_failure(self, caplog: pytest.LogCaptureFixture):
        import logging

        profiler = MagicMock()
        profiler.step.side_effect = RuntimeError("step failure")
        cb = ProfileStepCallback(profiler)
        control = MagicMock()

        with caplog.at_level(logging.WARNING):
            cb.on_step_end(args=None, state=None, control=control)
            cb.on_step_end(args=None, state=None, control=control)

        assert "stepping profiler failed" in caplog.text
        assert caplog.text.count("stepping profiler failed") == 1


class TestTrainProfileCLI:
    def _setup_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_rows: int = 8
    ) -> Path:
        monkeypatch.chdir(tmp_path)
        weights = _tiny_llama_dir(tmp_path)
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"t3 t4 {i}"},
                    {"role": "assistant", "content": "t5 t6"},
                ]
            }
            for i in range(num_rows)
        ]
        data_path = tmp_path / "chat.jsonl"
        data_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        cfg_path = tmp_path / "soup.yaml"
        cfg_path.write_text(
            f"base: {json.dumps(weights)}\n"
            "task: sft\n"
            "data:\n"
            "  train: chat.jsonl\n"
            "  format: chatml\n"
            "  chat_template: chatml\n"
            "  max_length: 64\n"
            "training:\n"
            "  epochs: 1\n"
            "  batch_size: 2\n"
            "  gradient_accumulation_steps: 1\n"
            "  logging_steps: 1\n"
            "  save_steps: 1000\n"
            "  quantization: none\n"
            "  lora:\n"
            "    r: 4\n"
            "    alpha: 8\n"
            "    target_modules: [q_proj, v_proj]\n"
            "output: ./out\n",
            encoding="utf-8",
        )
        return cfg_path

    def test_train_with_profile_writes_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg_path = self._setup_run(tmp_path, monkeypatch, num_rows=8)
        result = CliRunner().invoke(
            app, ["train", "--config", str(cfg_path), "--profile", "--yes"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

        profiles_dir = tmp_path / "out" / "profiles"
        assert profiles_dir.is_dir(), f"profiles dir not found in {list(tmp_path.glob('*'))}"

        traces = list(profiles_dir.glob("*.trace.json"))
        assert len(traces) == 1, f"expected 1 trace, found {traces}"
        trace_data = json.loads(traces[0].read_text(encoding="utf-8"))
        assert "traceEvents" in trace_data

    def test_train_without_profile_writes_no_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg_path = self._setup_run(tmp_path, monkeypatch, num_rows=4)
        result = CliRunner().invoke(
            app, ["train", "--config", str(cfg_path), "--yes"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

        profiles_dir = tmp_path / "out" / "profiles"
        assert not profiles_dir.exists(), "profiles dir should not exist without --profile"

    def test_mutation_dropping_callback_writes_no_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg_path = self._setup_run(tmp_path, monkeypatch, num_rows=8)
        # Simulate dropping the callback wiring (the bug before this fix)
        import soup_cli.utils.profiling as profiling_mod

        monkeypatch.setattr(profiling_mod, "build_profile_callback", lambda prof: None)

        result = CliRunner().invoke(
            app, ["train", "--config", str(cfg_path), "--profile", "--yes"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

        profiles_dir = tmp_path / "out" / "profiles"
        trace_files = list(profiles_dir.glob("*.trace.json")) if profiles_dir.exists() else []
        assert len(trace_files) == 0, "trace should not be written when callback is dropped"

    def test_train_says_no_trace_when_the_trainer_has_no_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import re

        import soup_cli.utils.profiling as profiling_mod

        cfg_path = self._setup_run(tmp_path, monkeypatch, num_rows=4)
        monkeypatch.setattr(profiling_mod, "attach_profile_callback", lambda *a: False)
        result = CliRunner().invoke(app, ["train", "--config", str(cfg_path), "--profile", "--yes"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())
        assert "no trace will be written" in plain
