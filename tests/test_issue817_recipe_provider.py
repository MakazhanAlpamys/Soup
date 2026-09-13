"""Regression tests for #817: recipe CLI provider wiring and explicit offline mode."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner


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


def test_recipe_cli_refuses_llm_nodes_without_provider(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)

    result = CliRunner().invoke(
        app,
        ["data", "recipe", str(recipe_path), "--execute", "--output", "out"],
    )

    assert result.exit_code == 2
    assert "--provider" in result.output
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
            "ollama",
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
    assert "Offline recipe mode" in result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "samp1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert all(row["llm1"].startswith("llm_text(offline):") for row in rows)


def test_run_recipe_requires_explicit_offline_opt_in(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from soup_cli.utils.recipe_dag import load_recipe_yaml
    from soup_cli.utils.recipe_run import run_recipe

    monkeypatch.chdir(tmp_path)
    recipe_path = _write_recipe(tmp_path)
    dag = load_recipe_yaml(str(recipe_path))

    with pytest.raises(ValueError, match="--provider"):
        run_recipe(dag, output_dir="out")
    assert not (tmp_path / "out").exists()
