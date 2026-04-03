"""Tests for HuggingFace Dataset Hub browser: search, preview, download."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from soup_cli.commands.data import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# soup data search
# ---------------------------------------------------------------------------


class TestDataSearch:
    """Tests for `soup data search` subcommand."""

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_basic(self, mock_list):
        """Basic search returns results table."""
        mock_ds = MagicMock()
        mock_ds.id = "teknium/OpenHermes-2.5"
        mock_ds.downloads = 50000
        mock_ds.likes = 200
        mock_ds.tags = ["en", "sft"]
        mock_list.return_value = [mock_ds]

        result = runner.invoke(app, ["search", "openhermes"])
        assert result.exit_code == 0
        assert "teknium/OpenHermes-2.5" in result.output

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_no_results(self, mock_list):
        """Search with no results shows message."""
        mock_list.return_value = []

        result = runner.invoke(app, ["search", "nonexistent_dataset_xyz_123"])
        assert result.exit_code == 0
        assert "No datasets found" in result.output

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_limit(self, mock_list):
        """--limit controls number of results."""
        datasets = []
        for idx in range(5):
            mock_ds = MagicMock()
            mock_ds.id = f"user/dataset-{idx}"
            mock_ds.downloads = 100
            mock_ds.likes = 10
            mock_ds.tags = []
            datasets.append(mock_ds)
        mock_list.return_value = datasets

        result = runner.invoke(app, ["search", "dataset", "--limit", "3"])
        assert result.exit_code == 0

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_sort_downloads(self, mock_list):
        """--sort downloads sorts by download count."""
        mock_list.return_value = []

        result = runner.invoke(
            app, ["search", "code", "--sort", "downloads"]
        )
        assert result.exit_code == 0
        mock_list.assert_called_once()
        call_kwargs = mock_list.call_args
        assert call_kwargs[1].get("sort") == "downloads"

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_sort_likes(self, mock_list):
        """--sort likes sorts by likes."""
        mock_list.return_value = []

        result = runner.invoke(app, ["search", "code", "--sort", "likes"])
        assert result.exit_code == 0
        call_kwargs = mock_list.call_args
        assert call_kwargs[1].get("sort") == "likes"

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_sort_invalid(self, mock_list):
        """Invalid --sort value is rejected."""
        result = runner.invoke(app, ["search", "code", "--sort", "invalid"])
        assert result.exit_code != 0

    @patch("soup_cli.commands.data.list_datasets")
    def test_search_huggingface_hub_not_installed(self, mock_list):
        """Graceful error when huggingface_hub is missing."""
        mock_list.side_effect = ImportError("No module named 'huggingface_hub'")

        result = runner.invoke(app, ["search", "code"])
        assert result.exit_code == 1
        assert "huggingface" in result.output.lower()


# ---------------------------------------------------------------------------
# soup data preview
# ---------------------------------------------------------------------------


class TestDataPreview:
    """Tests for `soup data preview` subcommand."""

    @patch("soup_cli.commands.data._hf_dataset_info")
    def test_preview_basic(self, mock_info):
        """Preview shows dataset info table."""
        mock_info.return_value = {
            "id": "teknium/OpenHermes-2.5",
            "description": "A large collection of instruction pairs.",
            "downloads": 50000,
            "likes": 200,
            "size_bytes": 2_000_000_000,
            "splits": {"train": 1_000_000, "test": 10_000},
            "features": ["conversations"],
            "tags": ["en", "sft"],
        }

        result = runner.invoke(app, ["preview", "teknium/OpenHermes-2.5"])
        assert result.exit_code == 0
        assert "teknium/OpenHermes-2.5" in result.output

    @patch("soup_cli.commands.data._hf_dataset_info")
    def test_preview_not_found(self, mock_info):
        """Preview of nonexistent dataset shows error."""
        mock_info.side_effect = ValueError("Dataset not found")

        result = runner.invoke(app, ["preview", "nonexistent/dataset"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    @patch("soup_cli.commands.data._hf_dataset_info")
    def test_preview_shows_splits(self, mock_info):
        """Preview shows split information."""
        mock_info.return_value = {
            "id": "test/ds",
            "description": "Test",
            "downloads": 100,
            "likes": 5,
            "size_bytes": 1000,
            "splits": {"train": 500, "validation": 100},
            "features": ["text"],
            "tags": [],
        }

        result = runner.invoke(app, ["preview", "test/ds"])
        assert result.exit_code == 0
        assert "train" in result.output

    @patch("soup_cli.commands.data._hf_dataset_info")
    def test_preview_shows_features(self, mock_info):
        """Preview shows feature columns."""
        mock_info.return_value = {
            "id": "test/ds",
            "description": "Test",
            "downloads": 100,
            "likes": 5,
            "size_bytes": 1000,
            "splits": {"train": 500},
            "features": ["instruction", "output", "input"],
            "tags": [],
        }

        result = runner.invoke(app, ["preview", "test/ds"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# soup data download
# ---------------------------------------------------------------------------


class TestDataDownload:
    """Tests for `soup data download` subcommand."""

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_basic(self, mock_download, tmp_path, monkeypatch):
        """Basic download writes JSONL output."""
        monkeypatch.chdir(tmp_path)
        output_file = tmp_path / "data.jsonl"
        mock_download.return_value = [
            {"instruction": "What is 2+2?", "output": "4"},
            {"instruction": "Hello", "output": "Hi there"},
        ]

        result = runner.invoke(
            app, ["download", "test/dataset", "-o", str(output_file)]
        )
        assert result.exit_code == 0
        assert output_file.exists()
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_with_split(self, mock_download, tmp_path, monkeypatch):
        """--split flag is passed to download function."""
        monkeypatch.chdir(tmp_path)
        output_file = tmp_path / "data.jsonl"
        mock_download.return_value = [{"text": "hello"}]

        result = runner.invoke(
            app,
            ["download", "test/dataset", "--split", "train[:100]", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert call_kwargs[1].get("split") == "train[:100]"

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_with_format_conversion(self, mock_download, tmp_path, monkeypatch):
        """--format converts downloaded data."""
        monkeypatch.chdir(tmp_path)
        output_file = tmp_path / "data.jsonl"
        # Alpaca format input
        mock_download.return_value = [
            {"instruction": "Q1", "input": "", "output": "A1"},
        ]

        result = runner.invoke(
            app,
            [
                "download", "test/dataset",
                "--format", "sharegpt",
                "-o", str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_empty_dataset(self, mock_download, tmp_path, monkeypatch):
        """Empty download result shows error."""
        monkeypatch.chdir(tmp_path)
        mock_download.return_value = []
        output_file = tmp_path / "data.jsonl"

        result = runner.invoke(
            app, ["download", "test/empty", "-o", str(output_file)]
        )
        assert result.exit_code == 1
        assert "empty" in result.output.lower() or "no data" in result.output.lower()

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_error_handling(self, mock_download, tmp_path):
        """Download failure shows friendly error."""
        mock_download.side_effect = ValueError("Dataset not found on HuggingFace Hub")
        output_file = tmp_path / "data.jsonl"

        result = runner.invoke(
            app, ["download", "nonexistent/dataset", "-o", str(output_file)]
        )
        assert result.exit_code == 1

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_output_path_traversal(self, mock_download, tmp_path):
        """Output path must stay under cwd."""
        mock_download.return_value = [{"text": "hello"}]

        result = runner.invoke(
            app,
            ["download", "test/dataset", "-o", "/etc/passwd"],
        )
        assert result.exit_code == 1
        assert "current working directory" in result.output.lower()

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_default_output(self, mock_download, tmp_path, monkeypatch):
        """Default output file uses dataset name."""
        monkeypatch.chdir(tmp_path)
        mock_download.return_value = [{"text": "hello"}]

        result = runner.invoke(
            app, ["download", "user/my-dataset"]
        )
        assert result.exit_code == 0
        # Default should be my-dataset.jsonl in cwd
        expected = tmp_path / "my-dataset.jsonl"
        assert expected.exists()

    @patch("soup_cli.commands.data._hf_download_dataset")
    def test_download_samples_limit(self, mock_download, tmp_path, monkeypatch):
        """--samples limits number of downloaded rows."""
        monkeypatch.chdir(tmp_path)
        output_file = tmp_path / "data.jsonl"
        mock_download.return_value = [
            {"text": f"row {idx}"} for idx in range(10)
        ]

        result = runner.invoke(
            app, ["download", "test/dataset", "--samples", "5", "-o", str(output_file)]
        )
        assert result.exit_code == 0
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_format_size_bytes(self):
        """_format_size_bytes formats various sizes correctly."""
        from soup_cli.commands.data import _format_size_bytes

        assert _format_size_bytes(0) == "0 B"
        assert _format_size_bytes(500) == "500 B"
        assert "KB" in _format_size_bytes(1024)
        assert "MB" in _format_size_bytes(1024 * 1024)
        assert "GB" in _format_size_bytes(1024 * 1024 * 1024)

    def test_format_size_bytes_none(self):
        """_format_size_bytes handles None."""
        from soup_cli.commands.data import _format_size_bytes

        assert _format_size_bytes(None) == "unknown"

    def test_format_count(self):
        """_format_count formats large numbers."""
        from soup_cli.commands.data import _format_count

        assert _format_count(500) == "500"
        assert "K" in _format_count(1500)
        assert "M" in _format_count(1_500_000)


# ---------------------------------------------------------------------------
# Security edge cases
# ---------------------------------------------------------------------------


class TestSecurityEdgeCases:
    """Security-relevant edge case tests."""

    def test_download_samples_over_limit(self):
        """--samples above 1M is rejected."""
        result = runner.invoke(
            app,
            ["download", "test/dataset", "--samples", "2000000", "-o", "out.jsonl"],
        )
        assert result.exit_code == 1
        assert "1,000,000" in result.output

    @patch("soup_cli.commands.data._hf_dataset_info")
    def test_preview_huggingface_hub_not_installed(self, mock_info):
        """Preview gracefully handles missing huggingface_hub."""
        mock_info.side_effect = ImportError("No module named 'huggingface_hub'")
        result = runner.invoke(app, ["preview", "test/ds"])
        assert result.exit_code == 1
        assert "huggingface" in result.output.lower()
