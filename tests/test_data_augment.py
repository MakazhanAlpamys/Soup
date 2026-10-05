"""Tests for data augmentation (Part F of v0.25.0)."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Strategy unit tests (with a mocked provider)
# ---------------------------------------------------------------------------


class FakeProvider:
    """Deterministic fake LLM provider for testing."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.calls += 1
        return f"rewritten-{self.calls}"


class TestAugmentStrategies:
    def test_rephrase_basic(self):
        from soup_cli.data.augment import augment_rephrase

        examples = [
            {"instruction": "What is Python?", "output": "Python is a language."},
        ]
        provider = FakeProvider()
        augmented = augment_rephrase(examples, provider=provider, count=2)
        # 1 original × 2 count × 2 string fields (instruction + output) = 4 provider calls
        assert provider.calls == 4
        # 1 original → 2 new augmentations
        assert len(augmented) == 2
        for row in augmented:
            assert "instruction" in row or "output" in row

    def test_translate_produces_langs(self):
        from soup_cli.data.augment import augment_translate

        examples = [
            {"instruction": "Hello", "output": "Hi"},
        ]
        provider = FakeProvider()
        augmented = augment_translate(examples, provider=provider, languages=["ru", "zh"])
        # 1 example × 2 languages × 2 string fields = 4 provider calls
        assert provider.calls == 4
        assert len(augmented) == 2

    def test_style_produces_styles(self):
        from soup_cli.data.augment import augment_style

        examples = [
            {
                "instruction": "Explain recursion.",
                "output": "It's when a function calls itself.",
            },
        ]
        provider = FakeProvider()
        augmented = augment_style(examples, provider=provider, styles=["formal", "casual"])
        # 1 example × 2 styles × 2 string fields = 4 provider calls
        assert provider.calls == 4
        assert len(augmented) == 2

    def test_count_cap_enforced(self):
        """count > 10 is rejected at strategy level (not just CLI)."""
        from soup_cli.data.augment import augment_rephrase

        examples = [{"instruction": "x", "output": "y"}]
        provider = FakeProvider()
        with pytest.raises(ValueError):
            augment_rephrase(examples, provider=provider, count=11)

    def test_empty_examples_returns_empty(self):
        from soup_cli.data.augment import (
            augment_rephrase,
            augment_style,
            augment_translate,
        )

        provider = FakeProvider()
        assert augment_rephrase([], provider=provider, count=2) == []
        assert augment_translate([], provider=provider, languages=["ru"]) == []
        assert augment_style([], provider=provider, styles=["formal"]) == []
        assert provider.calls == 0

    def test_translate_empty_languages_rejected(self):
        from soup_cli.data.augment import augment_translate

        with pytest.raises(ValueError):
            augment_translate([{"x": "y"}], provider=FakeProvider(), languages=[])

    def test_style_empty_styles_rejected(self):
        from soup_cli.data.augment import augment_style

        with pytest.raises(ValueError):
            augment_style([{"x": "y"}], provider=FakeProvider(), styles=[])

    def test_provider_exception_drops_row_and_counts_failure(self):
        """A failing provider call drops the row instead of writing an
        unrewritten/empty field, and is counted in ``stats`` (#1274)."""
        from soup_cli.data.augment import augment_rephrase
        from soup_cli.utils.data_forge import ForgeJudgeStats

        class FailingProvider:
            def generate(self, prompt: str, max_tokens: int = 512) -> str:
                raise RuntimeError("provider down")

        stats = ForgeJudgeStats()
        augmented = augment_rephrase(
            [{"instruction": "x"}],
            provider=FailingProvider(),
            count=1,
            stats=stats,
        )
        assert augmented == []
        assert (stats.calls, stats.failures) == (1, 1)
        assert stats.first_error == "provider down"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestAugmentCLI:
    def _write_data(self, tmp_path):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(3)]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_augment_help(self):
        result = runner.invoke(app, ["data", "augment", "--help"])
        assert result.exit_code == 0
        assert "augment" in result.output.lower()

    def test_augment_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "augment",
                "--input",
                "nonexistent.jsonl",
                "--output",
                "out.jsonl",
                "--strategy",
                "rephrase",
                "--provider",
                "ollama",
            ],
        )
        assert result.exit_code != 0

    def test_augment_count_cap(self, tmp_path, monkeypatch):
        """CLI rejects count > 10."""
        monkeypatch.chdir(tmp_path)
        path = self._write_data(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "augment",
                "--input",
                str(path.name),
                "--output",
                "out.jsonl",
                "--strategy",
                "rephrase",
                "--count",
                "15",
                "--provider",
                "ollama",
            ],
        )
        assert result.exit_code != 0

    def test_input_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "augment",
                "--input",
                "../../../etc/passwd",
                "--output",
                "out.jsonl",
                "--strategy",
                "rephrase",
                "--provider",
                "ollama",
            ],
        )
        assert result.exit_code != 0

    def test_output_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = self._write_data(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "augment",
                "--input",
                str(path.name),
                "--output",
                "../../evil.jsonl",
                "--strategy",
                "rephrase",
                "--provider",
                "ollama",
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Strategy validation
# ---------------------------------------------------------------------------


class TestStrategyValidation:
    def test_unknown_strategy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "data.jsonl"
        path.write_text('{"instruction": "x", "output": "y"}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "data",
                "augment",
                "--input",
                str(path.name),
                "--output",
                "out.jsonl",
                "--strategy",
                "bogus",
                "--provider",
                "ollama",
            ],
        )
        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
