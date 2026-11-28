"""Tests for data quality filters — perplexity and coherence scoring."""

from unittest.mock import patch as mock_patch

# ─── Coherence Scoring Tests ─────────────────────────────────────────────


class TestCoherenceScoring:
    """Test the coherence scoring function."""

    def test_empty_text_returns_zero(self):
        """Empty text should have 0 coherence."""
        from soup_cli.utils.quality import compute_coherence_score

        scores = compute_coherence_score([""])
        assert scores[0] == 0.0

    def test_whitespace_only_returns_zero(self):
        """Whitespace-only text should have 0 coherence."""
        from soup_cli.utils.quality import compute_coherence_score

        scores = compute_coherence_score(["   \n\t  "])
        assert scores[0] == 0.0

    def test_short_text_returns_low_score(self):
        """Very short text (< 3 words) should have low coherence."""
        from soup_cli.utils.quality import compute_coherence_score

        scores = compute_coherence_score(["Hi there"])
        assert scores[0] <= 0.3

    def test_coherent_text_returns_high_score(self):
        """Well-formed English text should have high coherence."""
        from soup_cli.utils.quality import compute_coherence_score

        text = (
            "Python is a versatile programming language. "
            "It is widely used for web development, data analysis, "
            "and machine learning applications."
        )
        scores = compute_coherence_score([text])
        assert scores[0] > 0.5

    def test_repetitive_text_returns_lower_score(self):
        """Highly repetitive text should score lower."""
        from soup_cli.utils.quality import compute_coherence_score

        normal = "Python is a programming language used for web development."
        repetitive = "the the the the the the the the the the the"

        scores = compute_coherence_score([normal, repetitive])
        assert scores[0] > scores[1]

    def test_scores_in_valid_range(self):
        """All coherence scores should be in [0, 1]."""
        from soup_cli.utils.quality import compute_coherence_score

        texts = [
            "Hello world!",
            "This is a proper sentence with good structure.",
            "asdfghjkl qwerty",
            "a b c d e f g h i j k l m n o p q r s t",
        ]
        scores = compute_coherence_score(texts)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_multiple_texts_scored_independently(self):
        """Each text should get its own score."""
        from soup_cli.utils.quality import compute_coherence_score

        scores = compute_coherence_score(["good text here", "another good text"])
        assert len(scores) == 2

    def test_empty_list_returns_empty(self):
        """Empty input should return empty output."""
        from soup_cli.utils.quality import compute_coherence_score

        scores = compute_coherence_score([])
        assert scores == []


# ─── _score_coherence Internal Tests ─────────────────────────────────────


class TestScoreCoherenceInternal:
    """Test the internal _score_coherence function."""

    def test_none_text_returns_zero(self):
        """None-like input should return 0."""
        from soup_cli.utils.quality import _score_coherence

        assert _score_coherence("") == 0.0

    def test_text_with_punctuation_scores_higher(self):
        """Text with sentence-ending punctuation should score higher."""
        from soup_cli.utils.quality import _score_coherence

        no_punct = "This is some text without any punctuation marks"
        with_punct = "This is some text. It has proper punctuation!"

        assert _score_coherence(with_punct) >= _score_coherence(no_punct)

    def test_diverse_vocabulary_scores_higher(self):
        """Text with diverse vocabulary should score higher."""
        from soup_cli.utils.quality import _score_coherence

        diverse = "The quick brown fox jumps over the lazy sleeping dog."
        monotone = "go go go go go go go go go go go go go go go."

        assert _score_coherence(diverse) > _score_coherence(monotone)


# ─── filter_by_quality Tests ─────────────────────────────────────────────


class TestFilterByQuality:
    """Test the filter_by_quality function."""

    def test_empty_data_returns_empty(self):
        """Empty data should return empty kept and removed."""
        from soup_cli.utils.quality import filter_by_quality

        kept, removed = filter_by_quality([], coherence_threshold=0.5)
        assert kept == []
        assert removed == []

    def test_coherence_filter_removes_low_quality(self):
        """Rows below coherence threshold should be removed."""
        from soup_cli.utils.quality import filter_by_quality

        data = [
            {"text": "This is a well-written sentence. It has proper structure."},
            {"text": "x"},
        ]
        kept, removed = filter_by_quality(data, coherence_threshold=0.3)
        assert len(kept) == 1
        assert len(removed) == 1
        assert kept[0]["text"].startswith("This is")

    def test_no_threshold_keeps_all(self):
        """Without thresholds, all data should be kept (but need at least one)."""
        # filter_by_quality requires at least one threshold, but passing
        # both as None effectively keeps all
        from soup_cli.utils.quality import filter_by_quality

        data = [{"text": "hello"}, {"text": "world"}]
        kept, removed = filter_by_quality(data)
        assert len(kept) == 2
        assert len(removed) == 0

    def test_text_field_option(self):
        """text_field parameter should select which field to score."""
        from soup_cli.utils.quality import filter_by_quality

        data = [
            {"instruction": "x", "output": "This is a good detailed response."},
            {"instruction": "y", "output": "z"},
        ]
        kept, removed = filter_by_quality(
            data, coherence_threshold=0.3, text_field="output",
        )
        assert len(kept) >= 1

    def test_perplexity_filter_requires_model(self):
        """Perplexity filtering should try to load a model."""
        from soup_cli.utils.quality import filter_by_quality

        data = [{"text": "Hello world."}]

        # Mock torch/transformers to avoid actual model loading
        with mock_patch(
            "soup_cli.utils.quality.compute_perplexity_scores",
            return_value=[50.0],
        ):
            kept, removed = filter_by_quality(
                data, perplexity_threshold=100.0,
            )
            assert len(kept) == 1
            assert len(removed) == 0

    def test_perplexity_filter_removes_high_perplexity(self):
        """Rows with perplexity above threshold should be removed."""
        from soup_cli.utils.quality import filter_by_quality

        data = [
            {"text": "Normal text."},
            {"text": "Gibberish asdlkfj asdlkfj."},
        ]

        with mock_patch(
            "soup_cli.utils.quality.compute_perplexity_scores",
            return_value=[50.0, 5000.0],
        ):
            kept, removed = filter_by_quality(
                data, perplexity_threshold=100.0,
            )
            assert len(kept) == 1
            assert len(removed) == 1


# ─── CLI Command Tests ───────────────────────────────────────────────────


class TestFilterCommand:
    """Test the soup data filter CLI command."""

    def test_filter_command_exists(self):
        """soup data filter should be a registered command."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "filter", "--help"])
        assert result.exit_code == 0
        assert "perplexity" in result.output.lower()
        assert "coherence" in result.output.lower()

    def test_filter_requires_threshold_or_score_only(self):
        """soup data filter should require at least one filter option."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "filter", "nonexistent.jsonl"])
        assert result.exit_code != 0

    def test_filter_file_not_found(self):
        """soup data filter should fail gracefully for missing file."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["data", "filter", "nonexistent.jsonl", "--coherence", "0.5"]
        )
        assert result.exit_code != 0

    def test_filter_with_coherence(self, tmp_path, monkeypatch):
        """soup data filter --coherence should filter low-quality rows."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        # Create test data
        data_file = tmp_path / "test.jsonl"
        rows = [
            {"text": "This is a well-written sentence. It has proper structure and meaning."},
            {"text": "x"},
        ]
        with open(data_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        output_file = tmp_path / "filtered.jsonl"
        runner = CliRunner()
        result = runner.invoke(app, [
            "data", "filter",
            str(data_file),
            "--coherence", "0.3",
            "--output", str(output_file),
        ])
        assert result.exit_code == 0
        assert output_file.exists()

        # Read output
        with open(output_file) as f:
            filtered = [json.loads(line) for line in f if line.strip()]
        assert len(filtered) == 1
        assert filtered[0]["text"].startswith("This is")

    def test_filter_score_only(self, tmp_path, monkeypatch):
        """soup data filter --score-only should add scores without removing rows."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        data_file = tmp_path / "test.jsonl"
        rows = [
            {"text": "Hello world. This is a test sentence."},
            {"text": "Another test."},
        ]
        with open(data_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        output_file = tmp_path / "scored.jsonl"
        runner = CliRunner()

        # Score-only requests both scores; keep this CLI test offline.
        monkeypatch.setattr(
            "soup_cli.utils.quality.compute_perplexity_scores",
            lambda *_args, **_kwargs: [10.0, 20.0],
        )
        result = runner.invoke(app, [
            "data", "filter",
            str(data_file),
            "--score-only",
            "--output", str(output_file),
        ])
        assert result.exit_code == 0, (result.output, result.exception)
        with open(output_file) as f:
            scored = [json.loads(line) for line in f if line.strip()]
        assert [row["text"] for row in scored] == [row["text"] for row in rows]
        assert [row["_perplexity_score"] for row in scored] == [10.0, 20.0]
        assert all("_coherence_score" in row for row in scored)


# ─── --field missing from rows (#3437) ───────────────────────────────────

FIELD_ROWS = [
    {
        "instruction": "Describe the weather.",
        "output": "The weather is warm and bright today. We walked to the park after lunch.",
    },
    # No `output`: scored on the whole row it would pass any threshold here.
    {"instruction": "Explain clearly how green plants turn sunlight into food for themselves."},
    {"instruction": "Say a word.", "output": "x"},
]


class TestFilterMissingField:
    """soup data filter --field does not score a row on fields it was not asked for."""

    @staticmethod
    def _run(tmp_path, monkeypatch, rows, *args):
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        monkeypatch.chdir(tmp_path)
        with open(tmp_path / "d.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        result = CliRunner().invoke(app, ["data", "filter", "d.jsonl", *args])
        return result, " ".join(strip_ansi(result.output).split())

    @staticmethod
    def _read(path):
        import json

        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @staticmethod
    def _patch_perplexity(monkeypatch, seen=None):
        def fake(texts, **_kwargs):
            if seen is not None:
                seen.append(list(texts))
            return [10.0 + index for index in range(len(texts))]

        monkeypatch.setattr("soup_cli.utils.quality.compute_perplexity_scores", fake)

    @staticmethod
    def _no_perplexity(monkeypatch):
        def fail(*_args, **_kwargs):
            raise AssertionError("perplexity scoring ran")

        monkeypatch.setattr("soup_cli.utils.quality.compute_perplexity_scores", fail)

    def test_field_absent_from_every_row_is_refused(self, tmp_path, monkeypatch):
        self._no_perplexity(monkeypatch)
        result, output = self._run(
            tmp_path, monkeypatch, FIELD_ROWS,
            "--coherence", "0.5", "--perplexity", "100", "--field", "outptu",
            "-o", "f.jsonl",
        )
        assert result.exit_code == 1, output
        assert (
            "--field 'outptu' is absent from every row; choose a field present in the dataset."
            in output
        )
        assert "Scoring" not in output
        assert not (tmp_path / "f.jsonl").exists()

    def test_field_absent_from_every_row_is_refused_under_score_only(
        self, tmp_path, monkeypatch,
    ):
        self._no_perplexity(monkeypatch)
        result, output = self._run(
            tmp_path, monkeypatch, FIELD_ROWS, "--score-only", "--field", "outptu",
        )
        assert result.exit_code == 1, output
        assert "is absent from every row" in output
        assert not (tmp_path / "d_scored.jsonl").exists()

    def test_row_without_the_field_is_removed_and_counted(self, tmp_path, monkeypatch):
        from soup_cli.utils.quality import compute_coherence_score

        result, output = self._run(
            tmp_path, monkeypatch, FIELD_ROWS,
            "--coherence", "0.5", "--field", "output", "-o", "f.jsonl",
        )
        assert result.exit_code == 0, output
        assert (
            "1 of 3 rows have no 'output' and were not scored; they are not in the output."
            in output
        )
        assert "3 -> 1 rows (-2 removed; 1 has no 'output')" in output
        kept = self._read(tmp_path / "f.jsonl")
        assert kept == [FIELD_ROWS[0]]
        # The average is over the two scored rows, not over three.
        scored = compute_coherence_score([FIELD_ROWS[0]["output"], FIELD_ROWS[2]["output"]])
        assert f"Avg coherence: {sum(scored) / 2:.3f} (threshold: 0.5)" in output

    def test_null_value_counts_as_missing(self, tmp_path, monkeypatch):
        rows = [FIELD_ROWS[0], {**FIELD_ROWS[1], "output": None}, FIELD_ROWS[2]]
        result, output = self._run(
            tmp_path, monkeypatch, rows,
            "--coherence", "0.5", "--field", "output", "-o", "f.jsonl",
        )
        assert result.exit_code == 0, output
        assert "1 of 3 rows have no 'output'" in output
        assert "3 -> 1 rows (-2 removed; 1 has no 'output')" in output
        assert self._read(tmp_path / "f.jsonl") == [FIELD_ROWS[0]]

    def test_perplexity_skips_the_row_without_the_field(self, tmp_path, monkeypatch):
        seen = []
        self._patch_perplexity(monkeypatch, seen)
        result, output = self._run(
            tmp_path, monkeypatch, FIELD_ROWS,
            "--perplexity", "100", "--field", "output", "-o", "f.jsonl",
        )
        assert result.exit_code == 0, output
        assert seen == [[FIELD_ROWS[0]["output"], FIELD_ROWS[2]["output"]]]
        assert "3 -> 2 rows (-1 removed; 1 has no 'output')" in output
        assert "Avg perplexity: 10.5 (threshold: 100.0)" in output
        assert self._read(tmp_path / "f.jsonl") == [FIELD_ROWS[0], FIELD_ROWS[2]]

    def test_score_only_writes_null_scores_for_the_row(self, tmp_path, monkeypatch):
        self._patch_perplexity(monkeypatch)
        result, output = self._run(
            tmp_path, monkeypatch, FIELD_ROWS,
            "--score-only", "--field", "output", "-o", "s.jsonl",
        )
        assert result.exit_code == 0, output
        assert (
            "1 of 3 rows have no 'output' and were not scored; their scores are null."
            in output
        )
        scored = self._read(tmp_path / "s.jsonl")
        assert [row["instruction"] for row in scored] == [r["instruction"] for r in FIELD_ROWS]
        assert scored[1]["_coherence_score"] is None
        assert scored[1]["_perplexity_score"] is None
        assert [scored[0]["_perplexity_score"], scored[2]["_perplexity_score"]] == [10.0, 11.0]
        assert isinstance(scored[0]["_coherence_score"], float)

    def test_field_every_row_has_is_unchanged(self, tmp_path, monkeypatch):
        rows = [FIELD_ROWS[0], FIELD_ROWS[2]]
        result, output = self._run(
            tmp_path, monkeypatch, rows,
            "--coherence", "0.5", "--field", "output", "-o", "f.jsonl",
        )
        assert result.exit_code == 0, output
        assert "2 -> 1 rows (-1 removed)" in output
        assert "were not scored" not in output and "has no" not in output
        assert self._read(tmp_path / "f.jsonl") == [FIELD_ROWS[0]]

    def test_no_field_scores_whole_rows_as_before(self, tmp_path, monkeypatch):
        result, output = self._run(
            tmp_path, monkeypatch, FIELD_ROWS, "--coherence", "0.5", "-o", "f.jsonl",
        )
        assert result.exit_code == 0, output
        assert "3 -> 3 rows (-0 removed)" in output
        assert "were not scored" not in output
        assert self._read(tmp_path / "f.jsonl") == FIELD_ROWS
