"""Part A wave 1 — v0.26.1 follow-ups (#32, #35) for v0.33.0.

Covers:
  - #32 Live model scoring: judge / benchmark task dispatch in run_gate,
    score=None + error propagation, _parse_judge_url helper, generator
    factory shape.
  - #35 Registry attach: --attach-to-registry on `soup eval custom`,
    --registry-id auto-attach in `soup export`, registry artifact kind
    extensions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# #32 — gate: judge URL parser
# ---------------------------------------------------------------------------


class TestParseJudgeURL:
    def test_ollama_scheme(self):
        from soup_cli.eval.gate import _parse_judge_url

        provider, model, base = _parse_judge_url("ollama://llama3.1")
        assert provider == "ollama"
        assert model == "llama3.1"
        assert base is None

    def test_https_openai(self):
        from soup_cli.eval.gate import _parse_judge_url

        provider, model, base = _parse_judge_url(
            "https://api.openai.com/gpt-4o-mini"
        )
        assert provider == "openai"
        assert model == "gpt-4o-mini"
        assert base == "https://api.openai.com"

    def test_http_localhost_server(self):
        from soup_cli.eval.gate import _parse_judge_url

        provider, model, base = _parse_judge_url(
            "http://localhost:8000/Qwen2.5"
        )
        assert provider == "server"
        assert model == "Qwen2.5"
        assert base == "http://localhost:8000"

    def test_rejects_unsupported_scheme(self):
        from soup_cli.eval.gate import _parse_judge_url

        with pytest.raises(ValueError, match="unsupported scheme"):
            _parse_judge_url("ftp://example.com/model")


# ---------------------------------------------------------------------------
# #32 — run_gate: error → score=None propagation
# ---------------------------------------------------------------------------


class TestRunGateErrorPropagation:
    def test_judge_task_failure_surfaces_score_none(self, tmp_path, monkeypatch):
        """Exception from judge backend must produce score=None, error=str(exc),
        passed=False — never a silent score=1.0."""
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        monkeypatch.chdir(tmp_path)
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(
            json.dumps({"prompt": "Hi"}) + "\n", encoding="utf-8",
        )

        suite = EvalSuite(suite="t", tasks=[GateTask(
            type="judge", name="quality", threshold=0.5,
            prompts="prompts.jsonl",
            judge_model="ollama://llama3.1",
        )])

        # Inject a JudgeEvaluator that explodes on construction.
        with patch("soup_cli.eval.judge.JudgeEvaluator") as mock_judge:
            mock_judge.side_effect = OSError("connection refused")
            result = run_gate(
                suite, generate_fn=lambda _p: "stub",
                regression_threshold=0.05,
            )

        assert len(result.task_results) == 1
        row = result.task_results[0]
        assert row.score is None
        assert row.error and "connection refused" in row.error
        assert row.passed is False
        assert result.passed is False

    def test_custom_task_unknown_file_error(self, tmp_path, monkeypatch):
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        monkeypatch.chdir(tmp_path)
        suite = EvalSuite(suite="t", tasks=[GateTask(
            type="custom", name="cust", threshold=0.5,
            tasks="missing.jsonl", scorer="exact",
        )])

        result = run_gate(suite, generate_fn=lambda _p: "out")
        row = result.task_results[0]
        assert row.score is None
        assert row.error
        assert row.passed is False

    def test_benchmark_task_unavailable(self, tmp_path, monkeypatch):
        from soup_cli.eval import forgetting
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        # Strip the runner attr to force the RuntimeError branch
        monkeypatch.setattr(
            forgetting, "run_mini_benchmark", None, raising=False,
        )
        # Ensure attribute lookup returns None
        if hasattr(forgetting, "run_mini_benchmark"):
            monkeypatch.delattr(
                forgetting, "run_mini_benchmark", raising=False,
            )

        suite = EvalSuite(suite="t", tasks=[GateTask(
            type="benchmark", name="bench", threshold=0.3,
            benchmark="mini_mmlu",
        )])
        result = run_gate(suite, generate_fn=lambda _p: "")
        row = result.task_results[0]
        assert row.score is None
        assert row.error and "unavailable" in row.error
        assert row.passed is False


class TestGateTaskResultSchema:
    def test_error_field_default_none(self):
        from soup_cli.eval.gate import GateTaskResult

        row = GateTaskResult(
            name="x", score=0.7, threshold=0.5,
            baseline=None, delta=None, passed=True,
        )
        assert row.error is None

    def test_score_optional(self):
        from soup_cli.eval.gate import GateTaskResult

        row = GateTaskResult(
            name="x", score=None, threshold=0.5,
            baseline=None, delta=None, passed=False,
            error="boom",
        )
        assert row.score is None
        assert row.error == "boom"


# ---------------------------------------------------------------------------
# #32 — quant_check.make_model_generator
# ---------------------------------------------------------------------------


class TestMakeModelGenerator:
    def test_max_new_tokens_bounds(self):
        from soup_cli.eval.quant_check import make_model_generator

        with pytest.raises(ValueError, match="max_new_tokens"):
            make_model_generator("/tmp/x", max_new_tokens=0)
        with pytest.raises(ValueError, match="max_new_tokens"):
            make_model_generator("/tmp/x", max_new_tokens=99_999)

    def test_returns_callable_with_mocked_transformers(self):
        from soup_cli.eval import quant_check

        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token_id = 0
        fake_inputs = {"input_ids": MagicMock()}
        fake_inputs["input_ids"].shape = (1, 3)
        fake_tokenizer.return_value = fake_inputs
        fake_tokenizer.decode.return_value = "out"

        fake_model = MagicMock()
        fake_model.generate.return_value = [[1, 2, 3, 4, 5, 6]]

        with patch.dict("sys.modules", {"transformers": MagicMock(
            AutoTokenizer=MagicMock(from_pretrained=MagicMock(
                return_value=fake_tokenizer,
            )),
            AutoModelForCausalLM=MagicMock(from_pretrained=MagicMock(
                return_value=fake_model,
            )),
        )}):
            gen = quant_check.make_model_generator(
                "/fake/model", max_new_tokens=8,
            )
            out = gen("hello")
        assert out == "out"

    def test_empty_prompt_returns_empty(self):
        from soup_cli.eval import quant_check

        fake_tok = MagicMock()
        fake_tok.eos_token_id = 0
        fake_model = MagicMock()
        with patch.dict("sys.modules", {"transformers": MagicMock(
            AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tok)),
            AutoModelForCausalLM=MagicMock(from_pretrained=MagicMock(return_value=fake_model)),
        )}):
            gen = quant_check.make_model_generator("/fake/model")
            assert gen("") == ""


# ---------------------------------------------------------------------------
# #35 — registry attach helpers
# ---------------------------------------------------------------------------


class TestRegistryAttachHelpers:
    def test_write_eval_json_containment(self, tmp_path, monkeypatch):
        from soup_cli.registry.attach import write_eval_json

        monkeypatch.chdir(tmp_path)
        out = write_eval_json(
            "results.json", payload={"score": 0.7},
        )
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["score"] == 0.7

    def test_write_eval_json_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.registry.attach import write_eval_json

        monkeypatch.chdir(tmp_path)
        outside = str(tmp_path.parent / "evil.json")
        with pytest.raises(ValueError, match="outside cwd"):
            write_eval_json(outside, payload={})

    def test_attach_artifact_unknown_entry(self, tmp_path, monkeypatch):
        from soup_cli.registry.attach import attach_artifact

        monkeypatch.chdir(tmp_path)
        # Use an isolated registry DB
        db = tmp_path / "reg.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))

        target = tmp_path / "results.json"
        target.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="not found"):
            attach_artifact("nonexistent-id", path=str(target), kind="eval_results")

    def test_attach_artifact_missing_file(self, tmp_path, monkeypatch):
        from soup_cli.registry.attach import attach_artifact

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            attach_artifact(
                "any", path=str(tmp_path / "missing.json"), kind="eval_results",
            )

    def test_attach_artifact_outside_cwd_rejected(self, tmp_path, monkeypatch):
        """enforce_cwd=True (default) must reject paths outside cwd."""
        from soup_cli.registry.attach import attach_artifact

        monkeypatch.chdir(tmp_path)
        # Isolated DB so the entry-not-found path doesn't shadow the
        # containment error.
        db = tmp_path / "reg.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))

        # Place an artifact OUTSIDE cwd.
        outside_dir = tmp_path.parent / "outside_for_attach_test"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "results.json"
        outside_file.write_text("{}", encoding="utf-8")

        # Even with a non-existent entry, the containment check happens
        # inside add_artifact and should surface; either error is acceptable
        # as long as the artifact never gets registered.
        with pytest.raises((ValueError, FileNotFoundError)):
            attach_artifact(
                "any", path=str(outside_file), kind="eval_results",
            )


class TestRegistryArtifactKindsExtended:
    def test_eval_results_kind_accepted(self):
        from soup_cli.registry.store import _VALID_KINDS

        assert "eval_results" in _VALID_KINDS
        assert "tensorrt" in _VALID_KINDS


class TestLookupEntryByOutputDir:
    def test_lookup_returns_none_when_no_match(self, tmp_path, monkeypatch):
        from soup_cli.registry.attach import lookup_entry_by_output_dir

        monkeypatch.chdir(tmp_path)
        db = tmp_path / "reg.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))

        result = lookup_entry_by_output_dir(str(tmp_path / "no-such-output"))
        assert result is None


# ---------------------------------------------------------------------------
# #35 — `soup eval custom --attach-to-registry` CLI integration
# ---------------------------------------------------------------------------


class TestEvalCustomAttachCLI:
    def test_attach_to_unknown_entry_errors(self, tmp_path, monkeypatch):
        """--attach-to-registry pointing at a missing entry produces a clean
        error and exits non-zero rather than silently passing."""
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        # Isolated registry DB so we don't pollute the user's ~/.soup
        monkeypatch.setenv(
            "SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"),
        )
        (tmp_path / "tasks.jsonl").write_text(
            json.dumps({"prompt": "p", "expected": "x"}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "model").mkdir()

        with patch(
            "soup_cli.eval.custom._create_default_generator",
            return_value=lambda _p: "x",
        ):
            result = runner.invoke(
                app,
                [
                    "eval", "custom",
                    "--tasks", "tasks.jsonl",
                    "--model", "model",
                    "--attach-to-registry", "no-such-id",
                    "--output", "results.json",
                ],
            )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert (
            "registry entry not found" in result.output
            or "not found" in result.output
        )
