"""Tests for Semantic Stratified Splitting in `soup data split`."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


@pytest.fixture
def dummy_dataset(tmp_path):
    ds_path = tmp_path / "dataset.jsonl"
    rows = [
        {"text": "Python programming language and coding", "category": "code"},
        {"text": "Writing code in python", "category": "code"},
        {"text": "Python software development", "category": "code"},
        {"text": "Grade school basic math and arithmetic", "category": "math"},
        {"text": "Solving equations and numbers", "category": "math"},
        {"text": "Basic algebra mathematics tutor", "category": "math"},
        {"text": "Creative storytelling writing poetry", "category": "writing"},
        {"text": "Novel author fantasy book prompt", "category": "writing"},
        {"text": "Short story narrative creative writer", "category": "writing"},
        {"text": "How to write screenplays and drama", "category": "writing"},
    ]
    ds_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return ds_path


class TestSemanticSplit:
    def test_stratify_exclusivity(self, dummy_dataset):
        """Cannot use --stratify and --stratify-semantic together."""
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "20",
                "--stratify", "category",
                "--stratify-semantic",
            ],
        )
        assert result.exit_code == 1
        assert "Cannot use --stratify and --stratify-semantic together" in result.output

    def test_invalid_num_clusters(self, dummy_dataset):
        """--num-clusters must be a positive integer."""
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "20",
                "--stratify-semantic",
                "--num-clusters", "0",
            ],
        )
        assert result.exit_code == 1
        assert "--num-clusters must be a positive integer" in result.output

    def test_semantic_split_happy_path(self, dummy_dataset, tmp_path, monkeypatch):
        """Happy path for semantic stratified splitting."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "30",  # absolute count 3
                "--test", "20",  # absolute count 2
                "--stratify-semantic",
                "--num-clusters", "3",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0

        train_file = tmp_path / "dataset_train.jsonl"
        val_file = tmp_path / "dataset_val.jsonl"
        test_file = tmp_path / "dataset_test.jsonl"

        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()

        train_rows = [
            json.loads(line)
            for line in train_file.read_text(encoding="utf-8").splitlines()
        ]
        val_rows = [
            json.loads(line) for line in val_file.read_text(encoding="utf-8").splitlines()
        ]
        test_rows = [
            json.loads(line)
            for line in test_file.read_text(encoding="utf-8").splitlines()
        ]

        # Verify that all rows are accounted for and no split is empty
        assert len(train_rows) + len(val_rows) + len(test_rows) == 10
        assert len(train_rows) > 0
        assert len(val_rows) > 0
        assert len(test_rows) > 0

    def test_semantic_split_missing_dependency(self, dummy_dataset, monkeypatch):
        """Test that missing scikit-learn prints an install hint and exits 1."""
        import sys
        monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
        monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)

        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "20",
                "--stratify-semantic",
            ],
        )
        assert result.exit_code == 1
        assert "requires scikit-learn" in result.output
        assert "pip install" in result.output

    def test_semantic_split_empty_vocabulary(self, tmp_path, monkeypatch):
        """Test ValueError handling when documents only contain stop words."""
        monkeypatch.chdir(tmp_path)
        ds_path = tmp_path / "stop_words.jsonl"
        # Only stop words or single characters (TfidfVectorizer will ignore)
        rows = [{"text": "the a and of"}, {"text": "a of"}, {"text": "and the"}]
        ds_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        result = runner.invoke(
            app,
            [
                "data", "split", str(ds_path),
                "--val", "30",
                "--stratify-semantic",
            ],
        )
        assert result.exit_code == 1
        assert "only contain stop words" in result.output

    def test_semantic_split_row_limit(self, tmp_path, monkeypatch):
        """Test row cap limits (capped at 50k rows)."""
        monkeypatch.chdir(tmp_path)
        ds_path = tmp_path / "large.jsonl"
        # Write dummy file so the file exists validation passes
        ds_path.write_text("{}", encoding="utf-8")
        # Mock load_raw_data to return a list of 50001 items without writing a huge file
        import soup_cli.commands.data as data_mod
        monkeypatch.setattr(data_mod, "load_raw_data", lambda path: [{"text": "a"}] * 50001)

        result = runner.invoke(
            app,
            [
                "data", "split", str(ds_path),
                "--val", "20",
                "--stratify-semantic",
            ],
        )
        assert result.exit_code == 1
        assert "capped at 50,000 rows" in result.output


    def test_num_clusters_warning(self, dummy_dataset):
        """Warning is printed when --num-clusters is passed without --stratify-semantic."""
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "20",
                "--num-clusters", "3",
            ],
        )
        assert result.exit_code == 0
        assert (
            "Warning: --num-clusters was passed but --stratify-semantic is not enabled"
            in result.output
        )

    def test_semantic_stratification_distribution(self, tmp_path, monkeypatch):
        """Verify that semantic stratification balances topic distributions."""

        monkeypatch.chdir(tmp_path)
        coding_texts = [
            "python programming language coding development",
            "write python function sort list dictionary",
            "import json parse file python script",
            "debug python code trace stack exception error",
            "object oriented programming python class init self",
            "list comprehension python map filter reduce lambda",
            "pip install package python dependencies environment virtualenv",
            "read csv file pandas dataframe python data science",
            "python decorator wrapper function arguments return",
            "asyncio async await coroutine concurrent python multitasking",
            "flask fastapi web server routing endpoints requests response python",
            "python unit testing unittest pytest assertions mocks fixtures",
            "string manipulation regex matching split join search python",
            "web scraping beautifulsoup scrapy parse html page python",
            "python sql connection cursor execute queries database select",
            "multiprocessing multi threading parallel execution python interpreter lock",
            "django web application framework mvc patterns model view template python",
            "python script automation shell command execute system path",
        ]
        baking_texts = [
            "bake sourdough bread flour yeast water salt starter recipe",
            "knead dough fermentation rising proofing baking oven temperature",
            "pastry dough butter rolling pin baking croissants puff pastry",
            "chocolate chip cookies baking tray sweet recipe ingredients sugar",
            "cake decorator frosting piping bag vanilla sponge cake baking tier",
            "bread maker machine automatic knead rise bake loaf recipe",
        ]

        rows = []
        for text in coding_texts:
            rows.append({"text": text, "category": "code"})
        for text in baking_texts:
            rows.append({"text": text, "category": "baking"})

        ds_path = tmp_path / "dataset.jsonl"
        ds_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        train_file = tmp_path / "dataset_train.jsonl"
        val_file = tmp_path / "dataset_val.jsonl"
        test_file = tmp_path / "dataset_test.jsonl"

        # 1. Run Semantic Stratified Split (under the same seed)
        result = runner.invoke(
            app,
            [
                "data", "split", str(ds_path),
                "--val", "25",
                "--test", "25",
                "--stratify-semantic",
                "--num-clusters", "2",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0

        val_rows = [
            json.loads(line) for line in val_file.read_text(encoding="utf-8").splitlines()
        ]
        test_rows = [
            json.loads(line) for line in test_file.read_text(encoding="utf-8").splitlines()
        ]

        # Verify that both splits contain at least 1 baking prompt (successfully stratified)
        val_baking = [r["category"] for r in val_rows].count("baking")
        test_baking = [r["category"] for r in test_rows].count("baking")
        assert val_baking >= 1
        assert test_baking >= 1

        # 2. Run Random Split as a Control (under the same seed)
        train_file.unlink(missing_ok=True)
        val_file.unlink(missing_ok=True)
        test_file.unlink(missing_ok=True)

        result_random = runner.invoke(
            app,
            [
                "data", "split", str(ds_path),
                "--val", "25",
                "--test", "25",
                "--seed", "42",
            ],
        )
        assert result_random.exit_code == 0

        val_rows_rand = [
            json.loads(line) for line in val_file.read_text(encoding="utf-8").splitlines()
        ]
        val_baking_rand = [r["category"] for r in val_rows_rand].count("baking")

        # Control assertion: plain random split fails to capture any baking prompts in validation
        assert val_baking_rand == 0


