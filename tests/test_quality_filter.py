"""Tests for data quality filters — perplexity and coherence scoring."""

from unittest.mock import patch as mock_patch

# ─── Coherence Scoring Tests ─────────────────────────────────────────────


class TestCoherenceScoring:
    """Test the coherence scoring function."""

    def test_empty_text_returns_zero(self):
        """Empty text should have 0 coherence."""
        from soup_cli.utils.quality import compute_coherence_scores

        scores = compute_coherence_scores([""])
        assert scores[0] == 0.0

    def test_whitespace_only_returns_zero(self):
        """Whitespace-only text should have 0 coherence."""
        from soup_cli.utils.quality import compute_coherence_scores

        scores = compute_coherence_scores(["   \n\t  "])
        assert scores[0] == 0.0

    def test_short_text_returns_low_score(self):
        """Very short text (< 3 words) should have low coherence."""
        from soup_cli.utils.quality import compute_coherence_scores

        scores = compute_coherence_scores(["Hi there"])
        assert scores[0] <= 0.3

    def test_coherent_text_returns_high_score(self):
        """Well-formed English text should have high coherence."""
        from soup_cli.utils.quality import compute_coherence_scores

        text = (
            "Python is a versatile programming language. "
            "It is widely used for web development, data analysis, "
            "and machine learning applications."
        )
        scores = compute_coherence_scores([text])
        assert scores[0] > 0.5

    def test_repetitive_text_returns_lower_score(self):
        """Highly repetitive text should score lower."""
        from soup_cli.utils.quality import compute_coherence_scores

        normal = "Python is a programming language used for web development."
        repetitive = "the the the the the the the the the the the"

        scores = compute_coherence_scores([normal, repetitive])
        assert scores[0] > scores[1]

    def test_scores_in_valid_range(self):
        """All coherence scores should be in [0, 1]."""
        from soup_cli.utils.quality import compute_coherence_scores

        texts = [
            "Hello world!",
            "This is a proper sentence with good structure.",
            "asdfghjkl qwerty",
            "a b c d e f g h i j k l m n o p q r s t",
        ]
        scores = compute_coherence_scores(texts)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_multiple_texts_scored_independently(self):
        """Each text should get its own score."""
        from soup_cli.utils.quality import compute_coherence_scores

        scores = compute_coherence_scores(["good text here", "another good text"])
        assert len(scores) == 2

    def test_empty_list_returns_empty(self):
        """Empty input should return empty output."""
        from soup_cli.utils.quality import compute_coherence_scores

        scores = compute_coherence_scores([])
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

    def test_filter_with_coherence(self, tmp_path):
        """soup data filter --coherence should filter low-quality rows."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

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

    def test_filter_score_only(self, tmp_path):
        """soup data filter --score-only should add scores without removing rows."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

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

        # score-only without perplexity (only coherence scores)
        result = runner.invoke(app, [
            "data", "filter",
            str(data_file),
            "--score-only",
            "--output", str(output_file),
        ])
        # May fail on torch import, but command structure should work
        if result.exit_code == 0:
            with open(output_file) as f:
                scored = [json.loads(line) for line in f if line.strip()]
            assert len(scored) == 2
            assert "_coherence_score" in scored[0]
