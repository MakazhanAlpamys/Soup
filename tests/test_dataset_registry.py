"""Tests for dataset info registry: register, unregister, list, resolve."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from soup_cli.commands.data import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


class TestDatasetRegistry:
    """Tests for dataset registry utility functions."""

    def test_register_dataset(self, tmp_path):
        """register_dataset adds entry to registry."""
        from soup_cli.utils.registry import (
            load_registry,
            register_dataset,
        )

        registry_path = tmp_path / "datasets.json"
        register_dataset(
            "my-instruct",
            str(tmp_path / "data.jsonl"),
            "alpaca",
            registry_path=registry_path,
        )
        reg = load_registry(registry_path)
        assert "my-instruct" in reg
        assert reg["my-instruct"]["format"] == "alpaca"

    def test_register_overwrites(self, tmp_path):
        """Registering same name overwrites previous entry."""
        from soup_cli.utils.registry import (
            load_registry,
            register_dataset,
        )

        registry_path = tmp_path / "datasets.json"
        register_dataset("ds", "/a.jsonl", "alpaca", registry_path=registry_path)
        register_dataset("ds", "/b.jsonl", "sharegpt", registry_path=registry_path)
        reg = load_registry(registry_path)
        assert reg["ds"]["path"].endswith("b.jsonl")
        assert reg["ds"]["format"] == "sharegpt"

    def test_unregister_dataset(self, tmp_path):
        """unregister_dataset removes entry."""
        from soup_cli.utils.registry import (
            load_registry,
            register_dataset,
            unregister_dataset,
        )

        registry_path = tmp_path / "datasets.json"
        register_dataset("ds", "/a.jsonl", "alpaca", registry_path=registry_path)
        result = unregister_dataset("ds", registry_path=registry_path)
        assert result is True
        reg = load_registry(registry_path)
        assert "ds" not in reg

    def test_unregister_nonexistent(self, tmp_path):
        """unregister_dataset returns False for missing name."""
        from soup_cli.utils.registry import unregister_dataset

        registry_path = tmp_path / "datasets.json"
        result = unregister_dataset("nope", registry_path=registry_path)
        assert result is False

    def test_load_empty_registry(self, tmp_path):
        """load_registry returns empty dict for missing file."""
        from soup_cli.utils.registry import load_registry

        registry_path = tmp_path / "datasets.json"
        assert load_registry(registry_path) == {}

    def test_resolve_registered_name(self, tmp_path):
        """resolve_dataset returns path for registered name."""
        from soup_cli.utils.registry import register_dataset, resolve_dataset

        registry_path = tmp_path / "datasets.json"
        data_path = str(tmp_path / "data.jsonl")
        register_dataset("my-ds", data_path, "alpaca", registry_path=registry_path)
        resolved = resolve_dataset("my-ds", registry_path=registry_path)
        assert resolved is not None
        assert resolved["path"] == data_path

    def test_resolve_unregistered_name(self, tmp_path):
        """resolve_dataset returns None for unknown name."""
        from soup_cli.utils.registry import resolve_dataset

        registry_path = tmp_path / "datasets.json"
        assert resolve_dataset("unknown", registry_path=registry_path) is None

    def test_list_registry(self, tmp_path):
        """load_registry returns all entries."""
        from soup_cli.utils.registry import (
            load_registry,
            register_dataset,
        )

        registry_path = tmp_path / "datasets.json"
        register_dataset("a", "/a.jsonl", "alpaca", registry_path=registry_path)
        register_dataset("b", "/b.jsonl", "dpo", registry_path=registry_path)
        reg = load_registry(registry_path)
        assert len(reg) == 2
        assert "a" in reg
        assert "b" in reg


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class TestRegistryCLI:
    """Tests for `soup data register/unregister/list` CLI commands."""

    @patch("soup_cli.commands.data._get_registry_path")
    def test_register_cli(self, mock_path, tmp_path, monkeypatch):
        """CLI register creates registry entry."""
        monkeypatch.chdir(tmp_path)
        registry_path = tmp_path / "datasets.json"
        mock_path.return_value = registry_path

        # Create a dummy data file
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"instruction": "hi", "output": "hello"}\n')

        result = runner.invoke(
            app,
            ["register", "--name", "my-ds", "--path", str(data_file), "--format", "alpaca"],
        )
        assert result.exit_code == 0
        assert "registered" in result.output.lower()

    @patch("soup_cli.commands.data._get_registry_path")
    def test_unregister_cli(self, mock_path, tmp_path):
        """CLI unregister removes entry."""
        from soup_cli.utils.registry import register_dataset

        registry_path = tmp_path / "datasets.json"
        mock_path.return_value = registry_path
        register_dataset("my-ds", "/data.jsonl", "alpaca", registry_path=registry_path)

        result = runner.invoke(app, ["unregister", "--name", "my-ds"])
        assert result.exit_code == 0
        assert "removed" in result.output.lower()

    @patch("soup_cli.commands.data._get_registry_path")
    def test_unregister_missing_cli(self, mock_path, tmp_path):
        """CLI unregister for missing name shows error."""
        registry_path = tmp_path / "datasets.json"
        mock_path.return_value = registry_path

        result = runner.invoke(app, ["unregister", "--name", "nonexistent"])
        assert result.exit_code == 1

    @patch("soup_cli.commands.data._get_registry_path")
    def test_list_registry_cli(self, mock_path, tmp_path):
        """CLI list shows registered datasets."""
        from soup_cli.utils.registry import register_dataset

        registry_path = tmp_path / "datasets.json"
        mock_path.return_value = registry_path
        register_dataset("ds1", "/a.jsonl", "alpaca", registry_path=registry_path)
        register_dataset("ds2", "/b.jsonl", "dpo", registry_path=registry_path)

        # list is a Python builtin, command is "registry" or "list"
        result = runner.invoke(app, ["registry"])
        assert result.exit_code == 0
        assert "ds1" in result.output
        assert "ds2" in result.output

    @patch("soup_cli.commands.data._get_registry_path")
    def test_list_empty_registry_cli(self, mock_path, tmp_path):
        """CLI list on empty registry shows message."""
        registry_path = tmp_path / "datasets.json"
        mock_path.return_value = registry_path

        result = runner.invoke(app, ["registry"])
        assert result.exit_code == 0
        assert "no datasets" in result.output.lower() or "empty" in result.output.lower()


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestRegistryValidation:
    """Tests for dataset name validation."""

    def test_name_with_path_separator_rejected(self, tmp_path):
        """Names with / or \\ are rejected."""
        from soup_cli.utils.registry import register_dataset

        registry_path = tmp_path / "datasets.json"
        with __import__("pytest").raises(ValueError):
            register_dataset(
                "bad/name", "/data.jsonl", "alpaca", registry_path=registry_path
            )

    def test_name_with_null_byte_rejected(self, tmp_path):
        """Names with null bytes are rejected."""
        from soup_cli.utils.registry import register_dataset

        registry_path = tmp_path / "datasets.json"
        with __import__("pytest").raises(ValueError):
            register_dataset(
                "bad\x00name", "/data.jsonl", "alpaca", registry_path=registry_path
            )

    def test_empty_name_rejected(self, tmp_path):
        """Empty names are rejected."""
        from soup_cli.utils.registry import register_dataset

        registry_path = tmp_path / "datasets.json"
        with __import__("pytest").raises(ValueError):
            register_dataset("", "/data.jsonl", "alpaca", registry_path=registry_path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRegistryErrorHandling:
    """Tests for registry error handling."""

    def test_load_corrupted_registry(self, tmp_path):
        """Corrupted JSON file raises ValueError."""
        from soup_cli.utils.registry import load_registry

        path = tmp_path / "datasets.json"
        path.write_text("{not valid json", encoding="utf-8")
        with __import__("pytest").raises(ValueError, match="corrupted"):
            load_registry(path)

    def test_load_non_dict_registry(self, tmp_path):
        """Non-dict JSON content raises ValueError."""
        from soup_cli.utils.registry import load_registry

        path = tmp_path / "datasets.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with __import__("pytest").raises(ValueError, match="unexpected format"):
            load_registry(path)
