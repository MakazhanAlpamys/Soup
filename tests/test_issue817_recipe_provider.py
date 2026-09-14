"""Regression tests for #817/#930 recipe provider wiring and failure accounting."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _write_recipe(tmp_path: Path) -> Path:
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"text": "keep"}) + "\n" + json.dumps({"text": "drop"}) + "\n",
        encoding="utf-8",
    )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "nodes:\n"
        "  - {name: seed1, kind: seed, config: {path: prompts.jsonl}}\n"
        "  - {name: llm1, kind: llm_text, config: {prompt: 'GENERATE {text}'}}\n"
        "  - {name: judge1, kind: judge, config: {prompt: 'JUDGE {llm1}'}}\n"
        "  - {name: samp1, kind: sampler, config: {}}\n"
        "edges: [[seed1, llm1], [llm1, judge1], [judge1, samp1]]\n",
        encoding="utf-8",
    )
    return recipe_path


def _write_single_provider_recipe(tmp_path: Path, *, kind: str) -> Path:
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"text": "keep"}) + "\n" + json.dumps({"text": "drop"}) + "\n",
        encoding="utf-8",
    )
    prompt = "GENERATE {text}" if kind == "llm_text" else "JUDGE {text}"
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "nodes:\n"
        "  - {name: seed1, kind: seed, config: {path: prompts.jsonl}}\n"
        f"  - {{name: provider1, kind: {kind}, config: {{prompt: '{prompt}'}}}}\n"
        "  - {name: samp1, kind: sampler, config: {}}\n"
        "edges: [[seed1, provider1], [provider1, samp1]]\n",
        encoding="utf-8",
    )
    return recipe_path


def test_recipe_cli_refuses_llm_nodes_without_provider(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)

    result = CliRunner().invoke(
        app,
        ["data", "recipe", str(recipe_path), "--execute", "--output", "out"],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 2, (output, repr(result.exception))
    assert "--provider" in output
    assert not (tmp_path / "out").exists()


def test_recipe_cli_provider_generates_and_rejects_rows(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.cli import app
    from soup_cli.utils import data_forge

    calls: list[tuple[str, str, str | None]] = []

    def fake_make(provider: str, *, model: str, base_url: str | None = None, **_kwargs):
        calls.append((provider, model, base_url))

        def generate(prompt: str) -> dict[str, str]:
            if prompt.startswith("GENERATE "):
                return {"text": f"answer:{prompt.removeprefix('GENERATE ')}"}
            return {"text": "REJECT" if "answer:drop" in prompt else "OK"}

        return generate

    monkeypatch.setattr(data_forge, "make_judge_provider_fn", fake_make)
    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "OLLAMA",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost:11434",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "samp1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["text"] for row in rows] == ["keep"]
    assert rows[0]["llm1"] == "answer:keep"
    assert rows[0]["judge1"] is True
    assert calls == [
        ("ollama", "test-model", "http://localhost:11434"),
        ("ollama", "test-model", "http://localhost:11434"),
    ]


def test_recipe_cli_offline_mode_is_explicit_and_loud(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--offline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Offline recipe mode" in strip_ansi(result.output)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "samp1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert all(row["llm1"].startswith("llm_text(offline):") for row in rows)


def test_run_recipe_requires_explicit_offline_opt_in(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.utils.recipe_dag import load_recipe_yaml
    from soup_cli.utils.recipe_run import run_recipe

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)
    dag = load_recipe_yaml(str(recipe_path))

    with pytest.raises(ValueError, match="judge_provider="):
        run_recipe(dag, output_dir="out")
    assert not (tmp_path / "out").exists()


def test_recipe_cli_rejects_unknown_provider_before_creating_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "openai",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 2, (output, repr(result.exception))
    assert "Unknown --provider 'openai'" in output
    assert not (tmp_path / "out").exists()


def test_recipe_cli_rejects_provider_with_offline_before_creating_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "ollama",
            "--offline",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 2, (output, repr(result.exception))
    assert "--provider and --offline cannot be used together" in output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [("--model", "test-model"), ("--base-url", "http://localhost:11434")],
)
def test_recipe_cli_requires_provider_for_provider_options(
    tmp_path: Path,
    monkeypatch,
    option: str,
    value: str,
) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)
    result = CliRunner().invoke(
        app,
        ["data", "recipe", str(recipe_path), "--execute", "--output", "out", option, value],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 2, (output, repr(result.exception))
    assert "--model and --base-url require --provider" in output
    assert not (tmp_path / "out").exists()


def test_run_recipe_rejects_provider_with_offline(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.utils.recipe_dag import load_recipe_yaml
    from soup_cli.utils.recipe_run import run_recipe

    monkeypatch.chdir(tmp_path)
    dag = load_recipe_yaml(str(_write_recipe(tmp_path)))

    with pytest.raises(ValueError, match="judge_provider and offline"):
        run_recipe(dag, output_dir="out", judge_provider="ollama", offline=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"judge_model": "test-model"}, {"judge_base_url": "http://localhost:11434"}],
)
def test_run_recipe_requires_provider_for_provider_parameters(
    tmp_path: Path,
    monkeypatch,
    kwargs: dict[str, str],
) -> None:
    from soup_cli.utils.recipe_dag import load_recipe_yaml
    from soup_cli.utils.recipe_run import run_recipe

    monkeypatch.chdir(tmp_path)
    dag = load_recipe_yaml(str(_write_recipe(tmp_path)))

    with pytest.raises(ValueError, match="require judge_provider="):
        run_recipe(dag, output_dir="out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_recipe_cli_closed_provider_fails_with_count_and_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_single_provider_recipe(tmp_path, kind="llm_text")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "ollama",
            "--base-url",
            endpoint,
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1, (output, repr(result.exception))
    assert "all 2 provider calls failed" in output
    assert endpoint in output
    checkpoint = json.loads((tmp_path / "out" / ".checkpoint.json").read_text())
    assert checkpoint["status"] == "failed"
    assert checkpoint["failed_node"] == "provider1"
    assert checkpoint["provider_call_counts"]["provider1"] == 2
    assert checkpoint["provider_failure_counts"]["provider1"] == 2
    assert not (tmp_path / "out" / "samp1.jsonl").exists()


def test_recipe_cli_preserves_legitimate_empty_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.cli import app
    from soup_cli.utils import data_forge

    def fake_make(*_args, raise_on_error: bool = False, **_kwargs):
        assert raise_on_error is True
        return lambda _prompt: {"text": ""}

    monkeypatch.setattr(data_forge, "make_judge_provider_fn", fake_make)
    monkeypatch.chdir(tmp_path)
    recipe_path = _write_single_provider_recipe(tmp_path, kind="llm_text")

    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "ollama",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, (output, repr(result.exception))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "samp1.jsonl").read_text().splitlines()
    ]
    assert [row["provider1"] for row in rows] == ["", ""]
    assert "2 provider calls, 0 failures" in output


def test_recipe_cli_reports_partial_failure_and_keeps_successful_empty_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.cli import app
    from soup_cli.utils import data_forge

    def fake_make(*_args, raise_on_error: bool = False, **_kwargs):
        assert raise_on_error is True

        def generate(prompt: str) -> dict[str, str]:
            if prompt.endswith("keep"):
                raise OSError("provider unavailable")
            return {"text": ""}

        return generate

    monkeypatch.setattr(data_forge, "make_judge_provider_fn", fake_make)
    monkeypatch.chdir(tmp_path)
    recipe_path = _write_single_provider_recipe(tmp_path, kind="llm_text")

    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "ollama",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, (output, repr(result.exception))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "samp1.jsonl").read_text().splitlines()
    ]
    assert rows == [{"text": "drop", "provider1": ""}]
    assert "2 provider calls, 1 failure" in output


def test_recipe_cli_judge_node_fails_when_every_provider_call_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.cli import app
    from soup_cli.utils import data_forge

    def fake_make(*_args, raise_on_error: bool = False, **_kwargs):
        assert raise_on_error is True

        def generate(_prompt: str) -> dict[str, str]:
            raise TimeoutError("provider timed out")

        return generate

    monkeypatch.setattr(data_forge, "make_judge_provider_fn", fake_make)
    monkeypatch.chdir(tmp_path)
    recipe_path = _write_single_provider_recipe(tmp_path, kind="judge")

    result = CliRunner().invoke(
        app,
        [
            "data",
            "recipe",
            str(recipe_path),
            "--execute",
            "--output",
            "out",
            "--provider",
            "ollama",
            "--base-url",
            "http://localhost:11434",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1, (output, repr(result.exception))
    assert "judge node 'provider1': all 2 provider calls failed" in output
    assert "http://localhost:11434" in output


@pytest.mark.parametrize("failure", ["timeout", "http", "malformed"])
def test_provider_factory_strict_mode_surfaces_request_failures(
    monkeypatch,
    failure: str,
) -> None:
    import httpx

    from soup_cli.utils.data_forge import ProviderCallError, make_judge_provider_fn

    class Response:
        status_code = 503 if failure == "http" else 200

        def json(self):
            if failure == "malformed":
                return {"choices": []}
            return {"choices": [{"message": {"content": "unused"}}]}

    def post(*_args, **_kwargs):
        if failure == "timeout":
            raise httpx.TimeoutException("timed out")
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    provider = make_judge_provider_fn("ollama", raise_on_error=True)

    with pytest.raises(ProviderCallError):
        provider("prompt")


def test_provider_factory_default_keeps_legacy_empty_failure_result(monkeypatch) -> None:
    import httpx

    from soup_cli.utils.data_forge import make_judge_provider_fn

    class Response:
        status_code = 503

    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: Response())

    assert make_judge_provider_fn("ollama")("prompt") == {"text": ""}


@pytest.mark.parametrize(
    ("provider_name", "base_url"),
    [("anthropic", None), ("vllm", "http://localhost:8000")],
)
def test_provider_factory_strict_mode_covers_every_backend(
    monkeypatch,
    provider_name: str,
    base_url: str | None,
) -> None:
    import httpx

    from soup_cli.utils.data_forge import ProviderCallError, make_judge_provider_fn

    class Response:
        status_code = 503

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-value")
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: Response())
    provider = make_judge_provider_fn(
        provider_name,
        base_url=base_url,
        raise_on_error=True,
    )

    with pytest.raises(ProviderCallError):
        provider("prompt")


def test_provider_factory_rejects_non_boolean_strict_mode() -> None:
    from soup_cli.utils.data_forge import make_judge_provider_fn

    with pytest.raises(TypeError, match="raise_on_error"):
        make_judge_provider_fn("ollama", raise_on_error="yes")  # type: ignore[arg-type]


def test_provider_failure_endpoint_label_omits_credentials_path_and_query() -> None:
    from soup_cli.utils.recipe_run import _provider_endpoint_label

    assert (
        _provider_endpoint_label(
            "ollama",
            "http://user:synthetic-password@localhost:11434/private?token=synthetic",
        )
        == "http://localhost:11434"
    )
