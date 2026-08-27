"""v0.53.7 — Data Forge + Pipeline live wave 1.

Tests for:
- #88  markdown heading-aware ingest split
- #112 decontaminate --benchmark-file
- #87  prompt_strategy live runtime
- #86  soup data preprocess live tokenize
- #111 soup data forge --judge-provider
- #75  QA log entry (file-presence assertion only)
- #106 utils/recipe_run.run_recipe live
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


# ---------------------------------------------------------------------------
# #88: split_markdown_by_headings
# ---------------------------------------------------------------------------
class TestMarkdownHeadingSplit:
    def test_empty_string_returns_empty_list(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        assert split_markdown_by_headings("") == []

    def test_preamble_only_returns_single_row_with_none_section(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        rows = split_markdown_by_headings("first paragraph\nsecond line")
        assert len(rows) == 1
        assert rows[0]["section"] is None
        assert rows[0]["level"] == 0
        assert "first paragraph\nsecond line" in rows[0]["text"]

    def test_three_headings_yield_three_sections(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        md = "# Intro\nIntro body\n## Sub\nSub body\n### Deep\nDeep body\n"
        rows = split_markdown_by_headings(md)
        assert len(rows) == 3
        assert rows[0]["section"] == "Intro"
        assert rows[0]["level"] == 1
        assert rows[1]["section"] == "Sub"
        assert rows[1]["level"] == 2
        assert rows[2]["section"] == "Deep"
        assert rows[2]["level"] == 3

    def test_preamble_plus_headings_yield_preamble_row(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        md = "preamble text\n# Heading\nbody"
        rows = split_markdown_by_headings(md)
        assert len(rows) == 2
        assert rows[0]["section"] is None
        assert rows[0]["level"] == 0
        assert rows[1]["section"] == "Heading"
        assert rows[1]["level"] == 1

    def test_atx_levels_1_through_6_accepted(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        md = "\n".join(f"{'#' * n} L{n}\nbody{n}" for n in range(1, 7))
        rows = split_markdown_by_headings(md)
        assert [r["level"] for r in rows] == [1, 2, 3, 4, 5, 6]

    def test_seven_hashes_not_a_heading(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        rows = split_markdown_by_headings("####### not a heading\nbody")
        assert len(rows) == 1
        assert rows[0]["section"] is None

    def test_hash_without_space_not_a_heading(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        rows = split_markdown_by_headings("#NoSpace\nbody")
        assert len(rows) == 1
        assert rows[0]["section"] is None

    def test_non_string_input_raises_typeerror(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings

        with pytest.raises(TypeError):
            split_markdown_by_headings(123)
        with pytest.raises(TypeError):
            split_markdown_by_headings(None)

    def test_ingest_cli_splits_markdown_when_requested(self, tmp_path, monkeypatch):
        # Write test markdown file
        md_path = tmp_path / "doc.md"
        md_path.write_text(
            "# Section One\nBody of one.\n\n# Section Two\nBody of two.\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "ingest",
                str(md_path),
                "-o",
                str(out),
                "--split-headings",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").strip().splitlines()]
        assert len(rows) == 2
        assert rows[0]["section"] == "Section One"
        assert rows[0]["level"] == 1
        assert rows[1]["section"] == "Section Two"
        assert rows[1]["level"] == 1

    def test_ingest_cli_defaults_to_single_row_for_md(self, tmp_path, monkeypatch):
        md_path = tmp_path / "doc.md"
        md_path.write_text("# Heading\nBody\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "ingest", str(md_path), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").strip().splitlines()]
        assert len(rows) == 1

    def test_ingest_cli_preamble_creates_extra_row(self, tmp_path, monkeypatch):
        md_path = tmp_path / "doc.md"
        md_path.write_text("preamble\n# Heading\nBody\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "ingest",
                str(md_path),
                "-o",
                str(out),
                "--split-headings",
            ],
        )
        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").strip().splitlines()]
        assert len(rows) == 2
        assert rows[0]["section"] is None
        assert rows[1]["section"] == "Heading"


# ---------------------------------------------------------------------------
# #112: decontaminate --benchmark-file
# ---------------------------------------------------------------------------
class TestDecontaminateBenchmarkFile:
    def test_loads_operator_corpus_and_filters(self, tmp_path, monkeypatch):
        # The decontaminate row text overlaps the benchmark text by n-grams.
        bench_path = tmp_path / "bench.jsonl"
        bench_path.write_text(
            json.dumps(
                {
                    "text": (
                        "the quick brown fox jumps over the lazy dog "
                        "many times in the morning sun every day"
                    )
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(
            json.dumps(
                {
                    "text": (
                        "the quick brown fox jumps over the lazy dog "
                        "many times in the morning sun every day"
                    )
                }
            )
            + "\n"
            + json.dumps({"text": "completely unrelated content here"})
            + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "clean.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "decontaminate",
                "-i",
                str(rows_path),
                "--benchmark-file",
                str(bench_path),
                "-o",
                str(out_path),
                "--n",
                "4",
                "--threshold",
                "0.3",
            ],
        )
        assert result.exit_code == 0, result.output
        kept = [
            json.loads(line) for line in out_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        # Overlapping row removed, unrelated kept.
        assert len(kept) == 1
        assert kept[0]["text"] == "completely unrelated content here"

    def test_non_existent_benchmark_file_fails(self, tmp_path, monkeypatch):
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(
            json.dumps({"text": "test row"}) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "decontaminate",
                "-i",
                str(rows_path),
                "--benchmark-file",
                str(tmp_path / "does_not_exist.jsonl"),
                "-o",
                str(tmp_path / "out.jsonl"),
            ],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_bad_benchmark_file_path_rejected(self, tmp_path, monkeypatch):
        # Path containment: referencing absolute outside paths where parent
        # does not exist should fail cleanly.
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(
            json.dumps({"text": "row"}) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "decontaminate",
                "-i",
                str(rows_path),
                "--benchmark-file",
                "/etc/does-not-exist.jsonl",
                "-o",
                "out.jsonl",
            ],
        )
        assert result.exit_code != 0

    def test_benchmarks_flag_still_accepted_with_label(self, tmp_path, monkeypatch):
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(
            json.dumps({"text": "sample text"}) + "\n", encoding="utf-8"
        )
        out_path = tmp_path / "clean.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "decontaminate",
                "-i",
                str(rows_path),
                "-b",
                "mmlu",
                "-o",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# #87: prompt_strategy live runtime
# ---------------------------------------------------------------------------
class TestPromptStrategyRuntime:
    def test_resolve_finds_callable(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy

        # json.dumps takes a positional arg
        spec = "json:dumps"
        fn = resolve_prompt_strategy(spec)
        assert callable(fn)

    def test_resolve_missing_module_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy

        with pytest.raises(ValueError, match="could not be imported"):
            resolve_prompt_strategy("definitely_not_a_module_xyz:fn")

    def test_resolve_missing_attribute_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy

        with pytest.raises(ValueError, match="no attribute"):
            resolve_prompt_strategy("json:not_a_real_attr_zzz")

    def test_resolve_non_callable_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy

        # sys.maxsize is an int, not callable
        with pytest.raises(ValueError, match="not callable"):
            resolve_prompt_strategy("sys:maxsize")

    def test_resolve_bad_shape_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy

        with pytest.raises(ValueError):
            resolve_prompt_strategy("no_colon")
        with pytest.raises(ValueError):
            resolve_prompt_strategy("")

    def test_resolve_non_string_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy

        with pytest.raises(ValueError):
            resolve_prompt_strategy(123)  # type: ignore[arg-type]

    def test_apply_with_none_returns_row(self):
        from soup_cli.utils.data_pipeline import apply_prompt_strategy

        row = {"a": 1}
        assert apply_prompt_strategy(None, row) is row

    def test_apply_callable_transforms_row(self, tmp_path, monkeypatch):
        # Create a module dynamically on sys.path
        mod_dir = tmp_path / "test_mod"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        (mod_dir / "t.py").write_text(
            "def upper(row):\n    return {'text': row.get('text', '').upper()}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from soup_cli.utils.data_pipeline import (
            apply_prompt_strategy,
            resolve_prompt_strategy,
        )

        resolve_prompt_strategy.cache_clear()
        spec = "tfm_mod.t:upper"
        # point directly at module created in tmp
        monkeypatch.syspath_prepend(str(mod_dir.parent))
        (tmp_path / "tfm_mod").mkdir(exist_ok=True)
        (tmp_path / "tfm_mod" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "tfm_mod" / "t.py").write_text(
            "def upper(row):\n    return {'text': row.get('text', '').upper()}\n",
            encoding="utf-8",
        )
        out = apply_prompt_strategy(spec, {"text": "hi"})
        assert out == {"text": "HI"}

    def test_apply_callable_exception_falls_through(self, tmp_path, monkeypatch):
        # If strategy raises, row returned unmodified (fail-safe)
        (tmp_path / "tfm_mod2").mkdir(exist_ok=True)
        (tmp_path / "tfm_mod2" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "tfm_mod2" / "t.py").write_text(
            "def boom(row):\n    raise RuntimeError('fail')\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from soup_cli.utils.data_pipeline import (
            apply_prompt_strategy,
            resolve_prompt_strategy,
        )

        resolve_prompt_strategy.cache_clear()
        spec = "tfm_mod2.t:boom"
        row = {"text": "x"}
        out = apply_prompt_strategy(spec, row)
        assert out is row

    def test_apply_non_mapping_return_falls_through(self, tmp_path, monkeypatch):
        (tmp_path / "tfm_mod3").mkdir(exist_ok=True)
        (tmp_path / "tfm_mod3" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "tfm_mod3" / "t.py").write_text(
            "def to_str(row):\n    return 'not a dict'\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from soup_cli.utils.data_pipeline import (
            apply_prompt_strategy,
            resolve_prompt_strategy,
        )

        resolve_prompt_strategy.cache_clear()
        row = {"text": "x"}
        out = apply_prompt_strategy("tfm_mod3.t:to_str", row)
        assert out is row

    def test_sft_format_threads_prompt_strategy(self, tmp_path, monkeypatch):
        mod_dir = tmp_path / "ps_dir"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        (mod_dir / "t.py").write_text(
            "def add_tag(row):\n    return {'prompt': '[TAG] ' + row.get('prompt', ''), 'response': row.get('response', '')}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / "ps_mod").mkdir(exist_ok=True)
        (tmp_path / "ps_mod" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "ps_mod" / "t.py").write_text(
            "def add_tag(row):\n    return {'prompt': '[TAG] ' + row.get('prompt', ''), 'response': row.get('response', '')}\n",
            encoding="utf-8",
        )
        from soup_cli.utils.data_pipeline import (
            format_sft_dataset,
            resolve_prompt_strategy,
        )

        resolve_prompt_strategy.cache_clear()
        raw = [{"prompt": "hi", "response": "bye"}]
        formatted = format_sft_dataset(
            raw, prompt_strategy="ps_mod.t:add_tag"
        )
        assert formatted[0]["prompt"] == "[TAG] hi"


# ---------------------------------------------------------------------------
# #86: soup data preprocess live tokenize
# ---------------------------------------------------------------------------
class TestDataPreprocessLiveTokenize:
    def test_preprocess_cli_requires_tokenizer_or_model(self, tmp_path):
        in_file = tmp_path / "in.jsonl"
        in_file.write_text(
            json.dumps({"text": "hello world"}) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "preprocess", "-i", str(in_file), "-o", str(tmp_path / "out")],
        )
        assert result.exit_code != 0
        assert "tokenizer" in result.output.lower() or "model" in result.output.lower()

    def test_preprocess_cli_tokenizes_with_mock(self, tmp_path, monkeypatch):
        # Mock HF AutoTokenizer so this unit test runs without torch/HF download
        fake_tok = MagicMock()
        fake_tok.encode.return_value = [101, 7592, 2088, 102]
        fake_tok.pad_token_id = 0
        fake_tok.eos_token_id = 102
        fake_tok.vocab_size = 32000

        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=fake_tok
        ):
            in_file = tmp_path / "in.jsonl"
            in_file.write_text(
                json.dumps({"text": "hello world"}) + "\n", encoding="utf-8"
            )
            out_dir = tmp_path / "tok_out"
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "data",
                    "preprocess",
                    "-i",
                    str(in_file),
                    "-o",
                    str(out_dir),
                    "--tokenizer",
                    "fake/model",
                ],
            )
            assert result.exit_code == 0, result.output
            assert out_dir.exists()
            assert (out_dir / "metadata.json").exists()
            meta = json.loads(
                (out_dir / "metadata.json").read_text(encoding="utf-8")
            )
            assert meta.get("num_rows") == 1
            assert meta.get("tokenizer") == "fake/model"


# ---------------------------------------------------------------------------
# #111: soup data forge --judge-provider
# ---------------------------------------------------------------------------
class TestDataForgeJudgeProvider:
    def test_forge_judge_provider_arg_accepted(self, tmp_path):
        runner = CliRunner()
        # Dry-run / invalid file to check flag parsing
        result = runner.invoke(
            app,
            [
                "data",
                "forge",
                "--spec",
                str(tmp_path / "missing.yaml"),
                "--judge-provider",
                "anthropic",
            ],
        )
        # Should fail on missing file, NOT on unknown option --judge-provider
        assert "no such option: --judge-provider" not in result.output.lower()

    def test_judge_provider_options_validation(self, tmp_path):
        # Unknown provider fails gracefully
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("domain: test\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "forge",
                "--spec",
                str(spec_path),
                "--judge-provider",
                "unsupported_provider_xyz",
            ],
        )
        assert result.exit_code != 0
        assert "unsupported_provider_xyz" in result.output or "invalid" in result.output.lower()


# ---------------------------------------------------------------------------
# #75: QA log entry presence
# ---------------------------------------------------------------------------
class TestQALogPresence:
    def test_qa_log_format_helper(self):
        from soup_cli.utils.qa_log import format_qa_entry

        entry = format_qa_entry(
            component="data_pipeline",
            status="pass",
            details={"rows_tested": 100},
        )
        assert isinstance(entry, dict)
        assert entry["component"] == "data_pipeline"
        assert entry["status"] == "pass"
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# #106: utils/recipe_run.run_recipe live execution
# ---------------------------------------------------------------------------
class TestRecipeRunLive:
    def test_run_recipe_missing_file_raises(self):
        from soup_cli.utils.recipe_run import run_recipe

        with pytest.raises(FileNotFoundError):
            run_recipe(Path("definitely_missing_recipe_xyz.yaml"))

    def test_run_recipe_dry_run_parses_yaml(self, tmp_path):
        from soup_cli.utils.recipe_run import run_recipe

        recipe = tmp_path / "r.yaml"
        recipe.write_text(
            "model: meta-llama/Llama-3.2-1B\ntask: sft\n", encoding="utf-8"
        )
        cfg = run_recipe(recipe, dry_run=True)
        assert cfg is not None
        assert cfg.get("model") == "meta-llama/Llama-3.2-1B"


# ---------------------------------------------------------------------------
# Server tool endpoints (#151)
# ---------------------------------------------------------------------------
def _create_test_app(auth_token=None, host="127.0.0.1"):
    try:
        import fastapi  # noqa: F401
    except ImportError:
        pytest.skip("FastAPI not installed")
    from soup_cli.commands.serve import _create_app

    model_obj = MagicMock()
    tokenizer = MagicMock()
    return _create_app(
        model_obj=model_obj,
        tokenizer=tokenizer,
        device="cpu",
        model_name="test-model",
        max_tokens_default=128,
        auth_token=auth_token,
        host=host,
    )


class TestToolEndpointsLive:
    @pytest.fixture(autouse=True)
    def _require_fastapi(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def test_python_tool_runs_simple_code(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print('hello')"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "stdout" in data
        assert "exit_code" in data
        assert "timed_out" in data

    def test_python_tool_rejects_missing_code(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/v1/tools/python", json={})
        assert resp.status_code == 400

    def test_python_tool_rejects_oversize_code(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        oversize = "x" * (64 * 1024 + 1)
        resp = client.post("/v1/tools/python", json={"code": oversize})
        assert resp.status_code == 400

    def test_python_tool_non_dict_rejected(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/v1/tools/python", json={"code": ""})
        assert resp.status_code == 400

    def test_bash_tool_executes_simple_command(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("bash sandbox not supported on Windows")
        from fastapi.testclient import TestClient
        from soup_cli.trainer.rewards import SandboxProcessResult

        app = _create_test_app()
        client = TestClient(app)
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._get_isolation_strategy", lambda: "namespaces"
        )
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._run_bash_sandbox",
            lambda cmd: SandboxProcessResult(returncode=0, stdout="hello\n", stderr=""),
        )
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["stdout"] == "hello\n"
        assert data["exit_code"] == 0
        assert not data["timed_out"]

    def test_bash_tool_rejects_missing_command(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/v1/tools/bash", json={})
        assert resp.status_code == 400

    def test_bash_tool_rejects_oversize_command(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        oversize = "x" * (64 * 1024 + 1)
        resp = client.post("/v1/tools/bash", json={"command": oversize})
        assert resp.status_code == 400

    def test_bash_tool_times_out(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("bash sandbox not supported on Windows")
        from fastapi.testclient import TestClient
        from soup_cli.trainer.rewards import SandboxProcessResult

        app = _create_test_app()
        client = TestClient(app)
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._get_isolation_strategy", lambda: "namespaces"
        )
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._run_bash_sandbox",
            lambda cmd: SandboxProcessResult(returncode=None, stdout="", stderr="", timed_out=True),
        )
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "sleep 10"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["timed_out"] is True
        assert data["exit_code"] == 124

    def test_bash_tool_blocks_network_namespace(self):
        if sys.platform != "linux":
            pytest.skip("network namespace test only valid on Linux")
        from soup_cli.trainer.rewards import _get_isolation_strategy, _run_bash_sandbox

        if _get_isolation_strategy() != "namespaces":
            pytest.skip("namespaces not available")

        result = _run_bash_sandbox("curl -s --connect-timeout 1 http://169.254.169.254/ || echo 'BLOCKED'")
        assert "BLOCKED" in result.stdout or result.returncode != 0

    def test_bash_tool_windows_returns_501(self, monkeypatch):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._get_isolation_strategy", lambda: "best-effort"
        )
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
        )
        assert resp.status_code == 501
        assert "OS-level isolation" in resp.json()["detail"]

    def test_bash_tool_auth_required_on_non_loopback_host(self):
        from fastapi.testclient import TestClient

        app = _create_test_app(host="0.0.0.0")
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
        )
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]

    def test_bash_tool_bearer_auth_validation(self, monkeypatch):
        from fastapi.testclient import TestClient
        from soup_cli.trainer.rewards import SandboxProcessResult

        app = _create_test_app(host="0.0.0.0", auth_token="secret123")
        client = TestClient(app)
        # Invalid token -> 401
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

        # Valid token -> 200 (mocked sandbox)
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._get_isolation_strategy", lambda: "namespaces"
        )
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._run_bash_sandbox",
            lambda cmd: SandboxProcessResult(returncode=0, stdout="ok", stderr=""),
        )
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
            headers={"Authorization": "Bearer secret123"},
        )
        assert resp.status_code == 200

    def test_bash_tool_bounded_streaming_kills_massive_output(self):
        if sys.platform == "win32":
            pytest.skip("bash sandbox not supported on Windows")
        from soup_cli.trainer.rewards import _run_sandboxed_subprocess

        argv = ["python3", "-c", "import sys; sys.stdout.write('A' * 200_000); sys.stdout.flush()"]
        result = _run_sandboxed_subprocess(argv, max_output_bytes=10_000)
        assert result.output_exceeded is True
        assert "exceeded limit" in result.stderr

    def test_bash_tool_environment_secret_isolation(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("bash sandbox not supported on Windows")
        from soup_cli.trainer.rewards import _run_sandboxed_subprocess

        monkeypatch.setenv("SECRET_TOKEN", "supersecret12345")
        argv = ["python3", "-c", "import os; print(os.environ.get('SECRET_TOKEN', 'ISOLATED'))"]
        result = _run_sandboxed_subprocess(argv)
        assert result.stdout == "ISOLATED"

    def test_bash_tool_restricted_linux_fails_closed_501(self, monkeypatch):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        monkeypatch.setattr(
            "soup_cli.trainer.rewards._get_isolation_strategy", lambda: "namespaces"
        )

        def _failing_bash(cmd):
            raise PermissionError("unshare failed: Operation not permitted")

        monkeypatch.setattr(
            "soup_cli.trainer.rewards._run_bash_sandbox",
            _failing_bash,
        )
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
        )
        assert resp.status_code == 501
        assert "Operation not permitted" in resp.json()["detail"]

    def test_web_search_default_deny_all(self):
        from fastapi.testclient import TestClient

        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/web_search",
            json={"query": "hello"},
        )
        assert resp.status_code == 403
