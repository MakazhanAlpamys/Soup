"""Tests for Issue #2048: soup eval export-check.

Verifies that an exported model preserves the behavioral delta of the
LoRA adapter it was produced from, and detects when quantization or merging
causes the adapter delta to be lost (reverting to base behavior).
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.export_check import (
    FunctionModelEvaluator,
    _char_overlap_ratio,
    _first_divergent_pos,
    _first_divergent_token,
    classify_artifact,
    load_prompts_file,
    render,
    run_export_check,
    run_export_check_from_factories,
)
from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


runner = CliRunner()


# ---------------------------------------------------------------------------
# Unit tests: Core comparison metrics & distance
# ---------------------------------------------------------------------------

class TestTextDistanceAndMetrics:
    def test_overlap_ratio_identical(self):
        assert _char_overlap_ratio("hello world", "hello world") == 1.0
        assert _char_overlap_ratio("", "") == 1.0

    def test_overlap_ratio_disjoint(self):
        assert _char_overlap_ratio("aaaa", "bbbb") == 0.0

    def test_overlap_ratio_empty(self):
        assert _char_overlap_ratio("text", "") == 0.0
        assert _char_overlap_ratio("", "text") == 0.0

    def test_first_divergent_pos(self):
        assert _first_divergent_pos("hello", "hello") is None
        assert _first_divergent_pos("hello", "help") == 3
        assert _first_divergent_pos("cat", "cats") == 3
        assert _first_divergent_pos("cats", "cat") == 3


# ---------------------------------------------------------------------------
# Unit tests: Artifact classification
# ---------------------------------------------------------------------------

class TestClassifyArtifact:
    def test_classify_hf_directory(self, tmp_path):
        hf_dir = tmp_path / "merged_model"
        hf_dir.mkdir()
        (hf_dir / "model.safetensors").write_bytes(b"dummy")
        assert classify_artifact(hf_dir) == "hf"

    def test_classify_refuses_gguf_file(self, tmp_path):
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="standalone GGUF file"):
            classify_artifact(gguf_file)

    def test_classify_refuses_onnx_file(self, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            classify_artifact(onnx_file)

    def test_classify_refuses_mlx_directory(self, tmp_path):
        mlx_dir = tmp_path / "mlx_model"
        mlx_dir.mkdir()
        (mlx_dir / "weights.npz").write_bytes(b"dummy")
        with pytest.raises(ValueError, match="MLX-fused"):
            classify_artifact(mlx_dir)

    def test_classify_refuses_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="no \\*\\.safetensors or \\*\\.bin"):
            classify_artifact(empty_dir)

    def test_classify_refuses_adapter_directory(self, tmp_path):
        adapter_dir = tmp_path / "adapter_dir"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"dummy")
        with pytest.raises(ValueError, match="is a LoRA adapter directory"):
            classify_artifact(adapter_dir)


# ---------------------------------------------------------------------------
# Unit tests: Prompts file loader
# ---------------------------------------------------------------------------

class TestLoadPromptsFile:
    def test_load_plain_text(self, tmp_path):
        p_file = tmp_path / "prompts.txt"
        p_file.write_text("Hello\n\nWorld\n", encoding="utf-8")
        assert load_prompts_file(str(p_file)) == ["Hello", "World"]

    def test_load_jsonl(self, tmp_path):
        p_file = tmp_path / "prompts.jsonl"
        p_file.write_text(
            json.dumps({"prompt": "Prompt 1"}) + "\n" +
            json.dumps({"prompt": "Prompt 2"}) + "\n",
            encoding="utf-8",
        )
        assert load_prompts_file(str(p_file)) == ["Prompt 1", "Prompt 2"]

    def test_load_alpaca_format(self, tmp_path):
        p_file = tmp_path / "alpaca.jsonl"
        p_file.write_text(
            json.dumps({"instruction": "Summarize", "input": "text here"}) + "\n" +
            json.dumps({"instruction": "Translate"}) + "\n",
            encoding="utf-8",
        )
        assert load_prompts_file(str(p_file)) == ["Summarize\n\ntext here", "Translate"]

    def test_load_chatml_format(self, tmp_path):
        p_file = tmp_path / "chatml.jsonl"
        msg_payload = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }
        p_file.write_text(json.dumps(msg_payload) + "\n", encoding="utf-8")
        assert load_prompts_file(str(p_file)) == ["You are helpful\n\nHello"]

    def test_load_bare_json_strings(self, tmp_path):
        p_file = tmp_path / "strings.jsonl"
        p_file.write_text('"Hello world"\n"Second prompt"\n', encoding="utf-8")
        assert load_prompts_file(str(p_file)) == ["Hello world", "Second prompt"]

    def test_load_empty_raises(self, tmp_path):
        p_file = tmp_path / "empty.txt"
        p_file.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="contains no prompts"):
            load_prompts_file(str(p_file))

    def test_load_malformed_json_raises(self, tmp_path):
        p_file = tmp_path / "bad.jsonl"
        p_file.write_text("{bad json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_prompts_file(str(p_file))


# ---------------------------------------------------------------------------
# Unit tests: run_export_check core logic
# ---------------------------------------------------------------------------

class TestRunExportCheck:
    def test_intact_adapter_passes(self):
        def base_gen(_prompt: str) -> str:
            return "base response"

        def ref_gen(_prompt: str) -> str:
            return "adapted response"

        def exported_gen(_prompt: str) -> str:
            return "adapted response"

        result = run_export_check(
            base_gen=base_gen,
            ref_gen=ref_gen,
            exported_gen=exported_gen,
            prompts=["test prompt 1", "test prompt 2"],
            min_agreement=0.80,
        )
        assert result.passed
        assert not result.delta_lost
        assert result.overall_agreement == 1.0
        assert result.exact_match_rate == 1.0
        assert all(r.verdict == "PASS" for r in result.rows)

    def test_destroyed_delta_fails(self):
        def base_gen(_prompt: str) -> str:
            return "base response"

        def ref_gen(_prompt: str) -> str:
            return "adapted response"

        def exported_gen(_prompt: str) -> str:
            return "base response"

        result = run_export_check(
            base_gen=base_gen,
            ref_gen=ref_gen,
            exported_gen=exported_gen,
            prompts=["test prompt 1"],
            min_agreement=0.80,
        )
        assert not result.passed
        assert result.delta_lost
        assert result.exact_match_rate == 0.0
        assert result.rows[0].verdict == "DELTA_LOST"

    def test_inactive_adapter_prompt(self):
        def base_gen(_prompt: str) -> str:
            return "identical response"

        def ref_gen(_prompt: str) -> str:
            return "identical response"

        def exported_gen(_prompt: str) -> str:
            return "identical response"

        result = run_export_check(
            base_gen=base_gen,
            ref_gen=ref_gen,
            exported_gen=exported_gen,
            prompts=["test prompt"],
            min_agreement=0.80,
        )
        assert not result.passed
        assert not result.delta_lost
        assert result.active_prompt_count == 0
        assert result.rows[0].verdict == "INACTIVE"

    def test_inactive_prompts_do_not_count_toward_agreement(self):
        def base_gen(prompt: str) -> str:
            return "base" if prompt == "active" else "same"

        def ref_gen(prompt: str) -> str:
            return "adapter" if prompt == "active" else "same"

        def exported_gen(prompt: str) -> str:
            return "adapter" if prompt == "active" else "unrelated"

        result = run_export_check(
            base_gen=base_gen,
            ref_gen=ref_gen,
            exported_gen=exported_gen,
            prompts=["active", "inactive"],
            min_agreement=1.0,
        )
        assert result.passed
        assert result.overall_agreement == 1.0
        assert result.exact_match_rate == 1.0
        assert result.active_prompt_count == 1

    def test_divergent_token_index(self):
        assert _first_divergent_token([1, 2, 3], [1, 2, 3]) is None
        assert _first_divergent_token([1, 2, 4], [1, 2, 5]) == 2
        assert _first_divergent_token([1, 2], [1, 2, 3]) == 2
        assert _first_divergent_token([1, 2, 3], [1, 2]) == 2

    def test_weak_delta_caught_despite_high_character_agreement(self):
        # Base: "The quick brown fox jumps over the lazy dog runs."
        # Ref (adapter): "The quick brown fox leaps over the lazy dog runs."
        # Exported: "The quick brown fox jumps over the lazy dog runs." (reverted to base)
        # Bigram overlap between exported and ref is 0.857 (above min_agreement 0.80),
        # but because exported is closer to base than to ref, the closer-to-base comparison
        # alone must flag this as DELTA_LOST!
        base_text = "The quick brown fox jumps over the lazy dog runs."
        ref_text = "The quick brown fox leaps over the lazy dog runs."
        exported_text = base_text

        result = run_export_check(
            base_gen=lambda _p: base_text,
            ref_gen=lambda _p: ref_text,
            exported_gen=lambda _p: exported_text,
            prompts=["Describe fox action:"],
            min_agreement=0.80,
        )
        row = result.rows[0]
        assert row.agree_export_ref > 0.80
        assert round(row.agree_export_ref, 3) == 0.857
        assert row.agree_export_base == 1.0
        assert result.delta_lost is True
        assert result.passed is False
        assert result.verdict == "DELTA_LOST"


class TestJSONTerminalSafety:
    def test_json_render_sanitizes_prompt_and_outputs_without_mutating_result(self):
        prompt = "[bold]probe\x1b]"
        base_output = "base [red]output\x1b]"
        ref_output = "adapter [green]output\x1b]"
        result = run_export_check(
            base_gen=lambda _prompt: base_output,
            ref_gen=lambda _prompt: ref_output,
            exported_gen=lambda _prompt: ref_output,
            prompts=[prompt],
        )

        rendered = render(result, fmt="json")
        data = json.loads(rendered)
        row = data["rows"][0]
        assert "\x1b" not in rendered
        assert row["prompt"] == r"\[bold]probe]"
        assert row["base_output"] == r"base \[red]output]"
        assert row["ref_output"] == r"adapter \[green]output]"
        assert result.rows[0].prompt == prompt
        assert result.rows[0].base_output == base_output
        assert result.rows[0].ref_output == ref_output


# ---------------------------------------------------------------------------
# CLI Integration Tests via CliRunner
# ---------------------------------------------------------------------------

class TestExportCheckCLI:
    def _setup_dirs(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "mock-base"}),
            encoding="utf-8",
        )

        exported_dir = tmp_path / "exported"
        exported_dir.mkdir()
        (exported_dir / "model.safetensors").write_bytes(b"dummy")

        return adapter_dir, exported_dir

    def test_cli_missing_options_exits_usage_error(self):
        result = runner.invoke(app, ["eval", "export-check"])
        assert result.exit_code == EXIT_USAGE_ERROR

    def test_cli_missing_prompts_exits_usage_error_with_guidance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(exported_dir),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR
        stripped = _strip_ansi(result.output)
        assert "give --prompts" in stripped

    def test_cli_path_outside_cwd_exits_usage_error(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)

        outside = tmp_path / "outside_adapter"
        outside.mkdir()

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(outside),
            "--exported", str(sub),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR

    def test_cli_prompts_outside_cwd_exits_usage_error(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        adapter_dir, exported_dir = self._setup_dirs(sub)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("adapter-specific prompt\n", encoding="utf-8")

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(exported_dir),
            "--prompts", str(prompts),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR

    def test_cli_unsupported_format_exits_usage_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "mock-base"}),
            encoding="utf-8",
        )
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_bytes(b"dummy")
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(gguf_file),
            "--prompts", str(prompts),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR
        assert "standalone GGUF" in _strip_ansi(result.output)

    def test_cli_unsupported_format_path_with_markup_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, _ = self._setup_dirs(tmp_path)
        gguf_file = tmp_path / "model[bold].gguf"
        gguf_file.write_bytes(b"dummy")
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(gguf_file),
            "--prompts", str(prompts),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR
        assert isinstance(result.exception, SystemExit)

    def test_cli_load_error_exits_runtime_error_without_stub(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")
        monkeypatch.setattr(
            "soup_cli.eval.export_check.make_hf_evaluator",
            lambda _path, **kwargs: (_ for _ in ()).throw(OSError("model unavailable")),
        )

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(exported_dir),
            "--prompts", str(prompts),
        ])
        assert result.exit_code == EXIT_RUNTIME_ERROR
        assert "Failed to load models" in _strip_ansi(result.output)

    def test_cli_allow_stub_fails_gate_with_code_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")
        monkeypatch.setattr(
            "soup_cli.eval.export_check.make_hf_evaluator",
            lambda _path, **kwargs: (_ for _ in ()).throw(OSError("model unavailable")),
        )

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(exported_dir),
            "--prompts", str(prompts),
            "--allow-stub",
        ])
        assert result.exit_code == EXIT_GATE_FAILED

    def test_cli_with_mocked_intact_generators_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")

        with patch("soup_cli.eval.export_check.make_hf_evaluator") as mock_hf_eval, \
             patch("soup_cli.eval.export_check.make_hf_adapter_evaluator") as mock_ref_eval:
            mock_hf_eval.side_effect = [
                FunctionModelEvaluator(lambda _p: "base"),      # base model
                FunctionModelEvaluator(lambda _p: "adapter"),   # exported model matches adapter
            ]
            mock_ref_eval.return_value = FunctionModelEvaluator(lambda _p: "adapter")

            result = runner.invoke(app, [
                "eval", "export-check",
                "--adapter", str(adapter_dir),
                "--exported", str(exported_dir),
                "--prompts", str(prompts),
                "--format", "json",
            ])
            assert result.exit_code == EXIT_OK
            data = json.loads(result.output)
            assert data["passed"] is True
            assert data["delta_lost"] is False

    def test_cli_with_mocked_destroyed_delta_fails_gate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")

        with patch("soup_cli.eval.export_check.make_hf_evaluator") as mock_hf_eval, \
             patch("soup_cli.eval.export_check.make_hf_adapter_evaluator") as mock_ref_eval:
            mock_hf_eval.side_effect = [
                FunctionModelEvaluator(lambda _p: "base response"),  # base model
                FunctionModelEvaluator(
                    lambda _p: "base response"
                ),  # exported model lost delta and reverted to base
            ]
            mock_ref_eval.return_value = FunctionModelEvaluator(lambda _p: "adapted response")

            result = runner.invoke(app, [
                "eval", "export-check",
                "--adapter", str(adapter_dir),
                "--exported", str(exported_dir),
                "--prompts", str(prompts),
                "--format", "json",
            ])
            assert result.exit_code == EXIT_GATE_FAILED
            data = json.loads(result.output)
            assert data["passed"] is False
            assert data["delta_lost"] is True

    def test_cli_no_evidence_fails_gate_with_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("prompt 1\n", encoding="utf-8")

        with patch("soup_cli.eval.export_check.make_hf_evaluator") as mock_hf_eval, \
             patch("soup_cli.eval.export_check.make_hf_adapter_evaluator") as mock_ref_eval:
            # Base, ref, and export all produce the exact same output -> adapter is inactive
            mock_hf_eval.side_effect = [
                FunctionModelEvaluator(lambda _p: "identical response"),
                FunctionModelEvaluator(lambda _p: "identical response"),
            ]
            mock_ref_eval.return_value = FunctionModelEvaluator(lambda _p: "identical response")

            result = runner.invoke(app, [
                "eval", "export-check",
                "--adapter", str(adapter_dir),
                "--exported", str(exported_dir),
                "--prompts", str(prompts),
                "--format", "json",
            ])
            assert result.exit_code == EXIT_GATE_FAILED
            data = json.loads(result.output)
            assert data["passed"] is False
            assert data["verdict"] == "NO_EVIDENCE"
            assert data["active_prompt_count"] == 0


# ---------------------------------------------------------------------------
# Offline Tiny-Model Integration Tests (tests._tiny_hf_models)
# ---------------------------------------------------------------------------

class TestTinyModelOfflineIntegration:
    """Uses real PyTorch/Transformers models and tokenizers locally without Hub access."""

    def test_tiny_model_evaluator_generate_and_nll(self):
        from soup_cli.eval.export_check import make_hf_evaluator
        from tests._tiny_hf_models import tiny_model_dir

        model_dir = tiny_model_dir("gpt2")
        evaluator = make_hf_evaluator(model_dir, max_new_tokens=5)

        # Greedy generation
        res = evaluator.generate("hello world", max_new_tokens=5)
        assert isinstance(res.text, str)
        assert isinstance(res.token_ids, list)
        assert len(res.token_ids) <= 5

        # Causal NLL computation
        nll = evaluator.compute_nll(["hello world", "the quick brown fox"])
        assert isinstance(nll, float)
        assert nll > 0.0

    def test_run_export_check_with_tiny_models(self):
        from soup_cli.eval.export_check import make_hf_evaluator, run_export_check
        from tests._tiny_hf_models import tiny_model_dir

        model_dir = tiny_model_dir("gpt2")
        evaluator = make_hf_evaluator(model_dir, max_new_tokens=5)

        # Base and reference identical -> INACTIVE, NO_EVIDENCE
        result = run_export_check(
            base_gen=evaluator,
            ref_gen=evaluator,
            exported_gen=evaluator,
            prompts=["hi there", "hello world"],
        )
        assert result.verdict == "NO_EVIDENCE"
        assert result.passed is False
        assert result.base_nll is not None
        assert result.ref_nll is not None
        assert result.exported_nll is not None


# ---------------------------------------------------------------------------
# Sequential loading regression tests
# ---------------------------------------------------------------------------

class TestSequentialLoading:
    """Verify that run_export_check_from_factories loads one model at a time."""

    def test_factories_called_exactly_once_each_in_order(self):
        """Each factory must be called exactly once and in base → ref → exported order."""
        call_log: list[str] = []

        def make_factory(label: str, output: str):
            def factory():
                call_log.append(f"load:{label}")
                return FunctionModelEvaluator(lambda _p, o=output: o)
            return factory

        result = run_export_check_from_factories(
            base_factory=make_factory("base", "base output"),
            ref_factory=make_factory("ref", "adapter output"),
            exported_factory=make_factory("exp", "adapter output"),
            prompts=["p1", "p2"],
        )
        assert call_log == ["load:base", "load:ref", "load:exp"]
        assert result.passed
        assert not result.delta_lost

    def test_sequential_intact_delta_passes(self):
        result = run_export_check_from_factories(
            base_factory=lambda: FunctionModelEvaluator(lambda _p: "base response"),
            ref_factory=lambda: FunctionModelEvaluator(lambda _p: "adapted response"),
            exported_factory=lambda: FunctionModelEvaluator(lambda _p: "adapted response"),
            prompts=["test prompt 1", "test prompt 2"],
            min_agreement=0.80,
        )
        assert result.passed
        assert not result.delta_lost
        assert result.overall_agreement == 1.0
        assert result.exact_match_rate == 1.0

    def test_sequential_destroyed_delta_fails(self):
        result = run_export_check_from_factories(
            base_factory=lambda: FunctionModelEvaluator(lambda _p: "base response"),
            ref_factory=lambda: FunctionModelEvaluator(lambda _p: "adapted response"),
            exported_factory=lambda: FunctionModelEvaluator(lambda _p: "base response"),
            prompts=["test prompt 1"],
            min_agreement=0.80,
        )
        assert not result.passed
        assert result.delta_lost
        assert result.verdict == "DELTA_LOST"

    def test_sequential_no_evidence_verdict(self):
        result = run_export_check_from_factories(
            base_factory=lambda: FunctionModelEvaluator(lambda _p: "same response"),
            ref_factory=lambda: FunctionModelEvaluator(lambda _p: "same response"),
            exported_factory=lambda: FunctionModelEvaluator(lambda _p: "same response"),
            prompts=["test prompt"],
        )
        assert not result.passed
        assert result.verdict == "NO_EVIDENCE"
        assert result.active_prompt_count == 0

    def test_sequential_nll_computed_per_model(self):
        """When factories provide NLL-capable evaluators, all three NLL values are populated."""
        nll_values = {"base": 1.0, "ref": 0.5, "exp": 0.5}

        def make_nll_factory(label: str, gen_out: str):
            def factory():
                return FunctionModelEvaluator(
                    lambda _p, o=gen_out: o,
                    nll_fn=lambda _texts: nll_values[label],
                )
            return factory

        result = run_export_check_from_factories(
            base_factory=make_nll_factory("base", "base out"),
            ref_factory=make_nll_factory("ref", "ref out"),
            exported_factory=make_nll_factory("exp", "ref out"),
            prompts=["p1"],
        )
        assert result.base_nll == 1.0
        assert result.ref_nll == 0.5
        assert result.exported_nll == 0.5
        assert result.passed

    def test_sequential_nll_closer_to_base_triggers_delta_lost(self):
        """When outputs agree but exported NLL reverts closer to base than ref,
        verdict is DELTA_LOST.

        Scenario: base NLL = 1.0, ref NLL = 0.5, exported NLL = 0.95.
        Generation outputs match reference (agreement = 1.0), but NLL delta was lost.
        """
        nll_map = {"base": 1.0, "ref": 0.5, "exp": 0.95}

        def make_factory(label: str):
            def factory():
                # Base generates "base response", ref and exp generate "adapted response"
                out = "base response" if label == "base" else "adapted response"
                return FunctionModelEvaluator(
                    lambda _p, o=out: o,
                    nll_fn=lambda _texts, tag=label: nll_map[tag],
                )
            return factory

        result = run_export_check_from_factories(
            base_factory=make_factory("base"),
            ref_factory=make_factory("ref"),
            exported_factory=make_factory("exp"),
            prompts=["prompt 1"],
            min_agreement=0.80,
        )
        assert result.base_nll == 1.0
        assert result.ref_nll == 0.5
        assert result.exported_nll == 0.95
        assert result.delta_lost is True
        assert result.passed is False
        assert result.verdict == "DELTA_LOST"

    def test_sequential_evaluator_released_before_next_factory_runs(self):
        """Verify that an evaluator is released before the next factory runs."""
        import weakref

        evaluator_refs: list[tuple[str, weakref.ReferenceType]] = []
        liveness_at_next_call: list[tuple[str, bool]] = []

        def make_tracked_factory(label: str):
            def factory():
                # Inspect whether any previously constructed evaluator is still alive
                for prev_label, ref in evaluator_refs:
                    is_alive = ref() is not None
                    liveness_at_next_call.append((f"{prev_label}_alive_at_{label}", is_alive))
                # Output matches ref output for exported to pass agreement
                out = "base output" if label == "base" else "adapter output"
                evaluator = FunctionModelEvaluator(lambda _p, o=out: o)
                evaluator_refs.append((label, weakref.ref(evaluator)))
                return evaluator
            return factory

        result = run_export_check_from_factories(
            base_factory=make_tracked_factory("base"),
            ref_factory=make_tracked_factory("ref"),
            exported_factory=make_tracked_factory("exp"),
            prompts=["prompt 1"],
        )
        assert result.passed

        # Check recorded liveness:
        # When ref_factory was invoked, base must already be released.
        # When exp_factory was invoked, base and ref must both already be released.
        assert ("base_alive_at_ref", False) in liveness_at_next_call
        assert ("base_alive_at_exp", False) in liveness_at_next_call
        assert ("ref_alive_at_exp", False) in liveness_at_next_call

        # At the end of execution, all three evaluators must be completely dereferenced
        for label, ref in evaluator_refs:
            assert ref() is None, f"{label} evaluator was not released at end"
