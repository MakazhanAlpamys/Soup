"""v0.69.0 Part D — Persona-Hub diversity sampler."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import persona_hub


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Bundled personas
# -----------------------------------------------------------------------------


class TestBundledPersonas:
    def test_at_least_one(self) -> None:
        personas = persona_hub.list_bundled_personas()
        assert len(personas) >= 5
        assert all(isinstance(p, str) for p in personas)

    def test_bundled_returns_tuple(self) -> None:
        personas = persona_hub.list_bundled_personas()
        assert isinstance(personas, tuple)

    def test_bundled_immutable(self) -> None:
        before = persona_hub.list_bundled_personas()
        # tuples are immutable; verify identity stability + no mutator
        after = persona_hub.list_bundled_personas()
        assert before == after


# -----------------------------------------------------------------------------
# Persona / style validators
# -----------------------------------------------------------------------------


class TestValidatePersona:
    def test_happy(self) -> None:
        assert persona_hub.validate_persona("a curious student") == "a curious student"

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.validate_persona("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            persona_hub.validate_persona("a\x00b")

    def test_oversize(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.validate_persona("p" * 2048)

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            persona_hub.validate_persona(42)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            persona_hub.validate_persona(True)


class TestValidateStyle:
    def test_happy(self) -> None:
        assert persona_hub.validate_style("formal") == "formal"

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.validate_style("")

    def test_oversize(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.validate_style("x" * 200)

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.validate_style("formal\x00")


# -----------------------------------------------------------------------------
# Sample matrix
# -----------------------------------------------------------------------------


class TestSamplePersonaMatrix:
    def test_happy(self) -> None:
        rows = persona_hub.sample_persona_matrix(
            prompts=["Explain X"],
            personas=["student", "engineer"],
            styles=["formal"],
            n=4,
            seed=0,
        )
        assert len(rows) == 4
        assert all("prompt" in r and "persona" in r and "style" in r for r in rows)

    def test_deterministic(self) -> None:
        kwargs = dict(
            prompts=["P1", "P2"],
            personas=["A", "B"],
            styles=["s1", "s2"],
            n=8,
            seed=42,
        )
        a = persona_hub.sample_persona_matrix(**kwargs)
        b = persona_hub.sample_persona_matrix(**kwargs)
        assert a == b

    def test_different_seeds_diverge(self) -> None:
        a = persona_hub.sample_persona_matrix(
            prompts=["P1", "P2"],
            personas=["A", "B"],
            styles=["s1", "s2"],
            n=10,
            seed=1,
        )
        b = persona_hub.sample_persona_matrix(
            prompts=["P1", "P2"],
            personas=["A", "B"],
            styles=["s1", "s2"],
            n=10,
            seed=2,
        )
        # With 2x2x2=8 combinations across n=10 samples, the orderings
        # should differ for at least one cell.
        assert a != b

    def test_empty_prompts_rejected(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.sample_persona_matrix(
                prompts=[],
                personas=["A"],
                styles=["s"],
                n=4,
                seed=0,
            )

    def test_empty_personas_rejected(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.sample_persona_matrix(
                prompts=["P"],
                personas=[],
                styles=["s"],
                n=4,
                seed=0,
            )

    def test_empty_styles_rejected(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.sample_persona_matrix(
                prompts=["P"],
                personas=["A"],
                styles=[],
                n=4,
                seed=0,
            )

    def test_n_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.sample_persona_matrix(
                prompts=["P"],
                personas=["A"],
                styles=["s"],
                n=0,
                seed=0,
            )

    def test_n_overcap(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.sample_persona_matrix(
                prompts=["P"],
                personas=["A"],
                styles=["s"],
                n=persona_hub._MAX_SAMPLES + 1,
                seed=0,
            )

    def test_n_bool(self) -> None:
        with pytest.raises(TypeError):
            persona_hub.sample_persona_matrix(
                prompts=["P"],
                personas=["A"],
                styles=["s"],
                n=True,
                seed=0,
            )

    def test_seed_bool(self) -> None:
        with pytest.raises(TypeError):
            persona_hub.sample_persona_matrix(
                prompts=["P"],
                personas=["A"],
                styles=["s"],
                n=4,
                seed=False,
            )

    def test_per_list_caps(self) -> None:
        big = ["x"] * (persona_hub._MAX_LIST_LEN + 1)
        with pytest.raises(ValueError):
            persona_hub.sample_persona_matrix(
                prompts=big,
                personas=["A"],
                styles=["s"],
                n=4,
                seed=0,
            )


# -----------------------------------------------------------------------------
# Topic diversity
# -----------------------------------------------------------------------------


class TestComputeTopicDiversity:
    def test_high_diversity(self) -> None:
        rows = [
            {"text": "cat dog mouse"},
            {"text": "engine wheel car"},
            {"text": "ocean wave fish"},
        ]
        score = persona_hub.compute_topic_diversity(rows)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_low_diversity_repeated_text(self) -> None:
        rows = [{"text": "same"} for _ in range(5)]
        score = persona_hub.compute_topic_diversity(rows)
        # entropy should be near 0 (single token, no variation)
        assert score < 0.5

    def test_empty_returns_zero(self) -> None:
        assert persona_hub.compute_topic_diversity([]) == 0.0

    def test_non_list_rejected(self) -> None:
        with pytest.raises(TypeError):
            persona_hub.compute_topic_diversity("rows")  # type: ignore[arg-type]

    def test_rows_without_text_skipped(self) -> None:
        # No text fields anywhere → returns 0.0 (no signal).
        rows = [{"unrelated": "x"}, {"y": 1}]
        score = persona_hub.compute_topic_diversity(rows)
        assert score == 0.0

    def test_score_finite(self) -> None:
        rows = [{"text": "a b c"}, {"text": "c d e"}]
        score = persona_hub.compute_topic_diversity(rows)
        assert math.isfinite(score)


# -----------------------------------------------------------------------------
# PersonaPlan + factory
# -----------------------------------------------------------------------------


class TestPersonaPlan:
    def test_frozen(self) -> None:
        plan = persona_hub.PersonaPlan(
            prompts=("P",),
            personas=("A",),
            styles=("s",),
            n=4,
            seed=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.n = 999  # type: ignore[misc]

    def test_tuples_required(self) -> None:
        with pytest.raises(TypeError):
            persona_hub.PersonaPlan(
                prompts=["P"],  # type: ignore[arg-type]
                personas=("A",),
                styles=("s",),
                n=4,
                seed=0,
            )


class TestBuildPersonaPlan:
    def test_happy(self) -> None:
        plan = persona_hub.build_persona_plan(
            prompts=["P"],
            personas=["A"],
            styles=["s"],
            n=4,
            seed=0,
        )
        assert plan.n == 4
        assert plan.prompts == ("P",)

    def test_validators_propagate(self) -> None:
        with pytest.raises(ValueError):
            persona_hub.build_persona_plan(
                prompts=[],
                personas=["A"],
                styles=["s"],
                n=4,
                seed=0,
            )


# -----------------------------------------------------------------------------
# CLI smoke
# -----------------------------------------------------------------------------


class TestPersonaMixCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["data", "persona-mix", "--help"])
        assert result.exit_code == 0, result.output

    def test_writes_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        prompts = _write(
            tmp_path / "prompts.jsonl",
            '{"prompt": "Explain X"}\n{"prompt": "Explain Y"}\n',
        )
        out = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "persona-mix",
                "--prompts",
                str(prompts),
                "--n",
                "4",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 4

    def test_missing_prompts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "persona-mix",
                "--prompts",
                "nope.jsonl",
                "--n",
                "4",
                "--output",
                str(tmp_path / "o.jsonl"),
            ],
        )
        assert result.exit_code != 0

    def test_outside_cwd_prompts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "p.jsonl", '{"prompt": "X"}\n')
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "persona-mix",
                "--prompts",
                str(outside / "p.jsonl"),
                "--n",
                "4",
                "--output",
                str(sub / "out.jsonl"),
            ],
        )
        assert result.exit_code != 0


# -----------------------------------------------------------------------------
# Source wiring
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_heavy_imports(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = (root / "soup_cli" / "utils" / "persona_hub.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport sentence_transformers",
        ):
            assert forbidden not in src

    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        assert __version__ == "0.69.0"
