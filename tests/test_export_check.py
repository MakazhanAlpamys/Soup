"""Tests for Issue #2048: soup eval export-check.

Verifies that an exported model preserves the behavioral delta of the
LoRA adapter it was produced from, and detects when quantization or merging
causes the adapter delta to be lost (reverting to base behavior).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.export_check import (
    _char_overlap_ratio,
    _first_divergent_pos,
    classify_artifact,
    load_prompts_file,
    render,
    run_export_check,
)
from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
)

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

    def test_low_agreement_fails(self):
        def base_gen(_prompt: str) -> str:
            return "base response"

        def ref_gen(_prompt: str) -> str:
            return "adapted response"

        def exported_gen(_prompt: str) -> str:
            return "completely unrecognised gibberish"

        result = run_export_check(
            base_gen=base_gen,
            ref_gen=ref_gen,
            exported_gen=exported_gen,
            prompts=["test prompt"],
            min_agreement=0.80,
        )
        assert not result.passed
        assert not result.delta_lost
        assert result.overall_agreement < 0.80


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

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(gguf_file),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR
        assert "standalone GGUF" in result.output

    def test_cli_unsupported_format_path_with_markup_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, _ = self._setup_dirs(tmp_path)
        gguf_file = tmp_path / "model[bold].gguf"
        gguf_file.write_bytes(b"dummy")

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(gguf_file),
        ])
        assert result.exit_code == EXIT_USAGE_ERROR
        assert isinstance(result.exception, SystemExit)

    def test_cli_load_error_exits_runtime_error_without_stub(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        monkeypatch.setattr(
            "soup_cli.eval.export_check.make_hf_generator",
            lambda _path: (_ for _ in ()).throw(OSError("model unavailable")),
        )

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(exported_dir),
        ])
        assert result.exit_code == EXIT_RUNTIME_ERROR
        assert "Failed to load models" in result.output

    def test_cli_allow_stub_passes_gate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)
        monkeypatch.setattr(
            "soup_cli.eval.export_check.make_hf_generator",
            lambda _path: (_ for _ in ()).throw(OSError("model unavailable")),
        )

        result = runner.invoke(app, [
            "eval", "export-check",
            "--adapter", str(adapter_dir),
            "--exported", str(exported_dir),
            "--allow-stub",
        ])
        assert result.exit_code == EXIT_GATE_FAILED

    def test_cli_with_mocked_intact_generators_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)

        with patch("soup_cli.eval.export_check.make_hf_generator") as mock_hf_gen, \
             patch("soup_cli.eval.export_check.make_hf_adapter_generator") as mock_ref_gen:
            mock_hf_gen.side_effect = [
                lambda _p: "base",      # base model
                lambda _p: "adapter",   # exported model matches adapter
            ]
            mock_ref_gen.return_value = lambda _p: "adapter"

            result = runner.invoke(app, [
                "eval", "export-check",
                "--adapter", str(adapter_dir),
                "--exported", str(exported_dir),
                "--format", "json",
            ])
            assert result.exit_code == EXIT_OK
            data = json.loads(result.output)
            assert data["passed"] is True
            assert data["delta_lost"] is False

    def test_cli_with_mocked_destroyed_delta_fails_gate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter_dir, exported_dir = self._setup_dirs(tmp_path)

        with patch("soup_cli.eval.export_check.make_hf_generator") as mock_hf_gen, \
             patch("soup_cli.eval.export_check.make_hf_adapter_generator") as mock_ref_gen:
            mock_hf_gen.side_effect = [
                lambda _p: "base response",  # base model
                lambda _p: "base response",  # exported model lost delta and reverted to base
            ]
            mock_ref_gen.return_value = lambda _p: "adapted response"

            result = runner.invoke(app, [
                "eval", "export-check",
                "--adapter", str(adapter_dir),
                "--exported", str(exported_dir),
                "--format", "json",
            ])
            assert result.exit_code == EXIT_GATE_FAILED
            data = json.loads(result.output)
            assert data["passed"] is False
            assert data["delta_lost"] is True
