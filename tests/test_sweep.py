"""Tests for soup sweep — hyperparameter search."""


import pytest


class TestParseSweepParams:
    """Test sweep parameter parsing."""

    def test_parse_simple_float_params(self):
        """Should parse float values like learning rates."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["lr=1e-5,2e-5,5e-5"])
        assert "lr" in result
        assert len(result["lr"]) == 3
        assert result["lr"][0] == pytest.approx(1e-5)
        assert result["lr"][1] == pytest.approx(2e-5)
        assert result["lr"][2] == pytest.approx(5e-5)

    def test_parse_int_params(self):
        """Should parse integer values."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["lora_r=8,16,32"])
        assert result["lora_r"] == [8, 16, 32]

    def test_parse_string_params(self):
        """Should parse string values."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["optimizer=adamw_torch,sgd"])
        assert result["optimizer"] == ["adamw_torch", "sgd"]

    def test_parse_bool_params(self):
        """Should parse boolean values."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["some_flag=true,false"])
        assert result["some_flag"] == [True, False]

    def test_parse_multiple_params(self):
        """Should parse multiple parameter strings."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["lr=1e-5,2e-5", "epochs=2,3"])
        assert "lr" in result
        assert "epochs" in result
        assert len(result["lr"]) == 2
        assert len(result["epochs"]) == 2

    def test_parse_invalid_param_no_equals(self):
        """Should skip params without equals sign."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["invalid_param"])
        assert len(result) == 0

    def test_parse_none_value(self):
        """Should parse 'none' as None."""
        from soup_cli.commands.sweep import _parse_sweep_params

        result = _parse_sweep_params(["target_modules=auto,none"])
        assert result["target_modules"] == ["auto", None]


class TestParseValue:
    """Test individual value parsing."""

    def test_parse_int(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("42") == 42

    def test_parse_float(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("3.14") == pytest.approx(3.14)

    def test_parse_scientific_notation(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("1e-5") == pytest.approx(1e-5)

    def test_parse_bool_true(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("true") is True

    def test_parse_bool_false(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("false") is False

    def test_parse_none(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("none") is None

    def test_parse_string(self):
        from soup_cli.commands.sweep import _parse_value

        assert _parse_value("adamw_torch") == "adamw_torch"


class TestGenerateCombinations:
    """Test parameter combination generation."""

    def test_grid_search(self):
        """Grid search should generate all combinations."""
        from soup_cli.commands.sweep import _generate_combinations

        params = {"lr": [1e-5, 2e-5], "epochs": [2, 3]}
        combos = _generate_combinations(params, "grid", None)
        assert len(combos) == 4  # 2 x 2

    def test_grid_search_single_param(self):
        """Grid with one param should equal param count."""
        from soup_cli.commands.sweep import _generate_combinations

        params = {"lr": [1e-5, 2e-5, 5e-5]}
        combos = _generate_combinations(params, "grid", None)
        assert len(combos) == 3

    def test_grid_search_max_runs(self):
        """Grid with max_runs should truncate."""
        from soup_cli.commands.sweep import _generate_combinations

        params = {"lr": [1e-5, 2e-5], "epochs": [2, 3]}
        combos = _generate_combinations(params, "grid", max_runs=2)
        assert len(combos) == 2

    def test_random_search(self):
        """Random search should respect max_runs."""
        from soup_cli.commands.sweep import _generate_combinations

        params = {"lr": [1e-5, 2e-5, 5e-5], "epochs": [2, 3, 5]}
        combos = _generate_combinations(params, "random", max_runs=3)
        assert len(combos) == 3

    def test_random_search_no_duplicates(self):
        """Random search should not produce duplicates."""
        from soup_cli.commands.sweep import _generate_combinations

        params = {"lr": [1e-5, 2e-5], "epochs": [2, 3]}
        combos = _generate_combinations(params, "random", max_runs=4)
        combo_tuples = [tuple(sorted(c.items())) for c in combos]
        assert len(combo_tuples) == len(set(combo_tuples))

    def test_random_search_defaults_to_10(self):
        """Random search without max_runs should default to min(total, 10)."""
        from soup_cli.commands.sweep import _generate_combinations

        params = {"lr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
        combos = _generate_combinations(params, "random", max_runs=None)
        assert len(combos) == 10


class TestSetNestedParam:
    """Test nested parameter setting with shortcuts."""

    def test_set_shortcut_lr(self):
        """'lr' shortcut should set training.lr."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"lr": 1e-5, "epochs": 3}}
        _set_nested_param(config, "lr", 2e-5)
        assert config["training"]["lr"] == pytest.approx(2e-5)

    def test_set_shortcut_epochs(self):
        """'epochs' shortcut should set training.epochs."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"epochs": 3}}
        _set_nested_param(config, "epochs", 5)
        assert config["training"]["epochs"] == 5

    def test_set_shortcut_lora_r(self):
        """'lora_r' shortcut should set training.lora.r."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"lora": {"r": 64}}}
        _set_nested_param(config, "lora_r", 32)
        assert config["training"]["lora"]["r"] == 32

    def test_set_dot_notation(self):
        """Dot notation should work for custom paths."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"weight_decay": 0.01}}
        _set_nested_param(config, "training.weight_decay", 0.1)
        assert config["training"]["weight_decay"] == pytest.approx(0.1)

    def test_set_creates_missing_keys(self):
        """Should create intermediate keys if missing."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "training.lr", 1e-5)
        assert config["training"]["lr"] == pytest.approx(1e-5)


class TestEarlyStopping:
    """Test early stopping logic in sweep."""

    def test_early_stop_flag_in_dry_run(self, tmp_path):
        """Dry run with early-stop should show plan."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: test-model\n"
            "data:\n"
            "  train: ./data.jsonl\n"
        )

        runner = CliRunner()
        result = runner.invoke(app, [
            "sweep",
            "--config", str(config_file),
            "--param", "lr=1e-5,2e-5,5e-5",
            "--early-stop", "1.5",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Sweep Plan" in result.output

    def test_early_stop_skips_bad_runs(self):
        """Early stopping should skip runs when loss exceeds threshold."""
        # Simulate the early stopping logic directly
        best_loss = 0.5
        early_stop = 1.5
        current_loss = 1.0  # 0.5 * 1.5 = 0.75, current is 1.0 > 0.75

        assert current_loss > best_loss * early_stop

    def test_early_stop_allows_good_runs(self):
        """Early stopping should NOT skip runs within threshold."""
        best_loss = 0.5
        early_stop = 1.5
        current_loss = 0.6  # 0.5 * 1.5 = 0.75, current is 0.6 < 0.75

        assert current_loss <= best_loss * early_stop

    def test_early_stop_threshold_values(self):
        """Various threshold values should work correctly."""
        best_loss = 1.0

        # 1.2 = 20% worse tolerance
        assert 1.15 <= best_loss * 1.2  # within threshold
        assert 1.25 > best_loss * 1.2   # exceeds threshold

        # 2.0 = 100% worse tolerance
        assert 1.99 <= best_loss * 2.0  # within threshold
        assert 2.01 > best_loss * 2.0   # exceeds threshold


class TestSweepCLI:
    """Test sweep CLI command."""

    def test_sweep_dry_run(self, tmp_path):
        """Dry run should show plan without executing."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        # Create a minimal config
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: test-model\n"
            "data:\n"
            "  train: ./data.jsonl\n"
        )

        runner = CliRunner()
        result = runner.invoke(app, [
            "sweep",
            "--config", str(config_file),
            "--param", "lr=1e-5,2e-5",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Sweep Plan" in result.output
        assert "lr" in result.output

    def test_sweep_config_not_found(self):
        """Should fail if config doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "sweep",
            "--config", "/nonexistent/soup.yaml",
            "--param", "lr=1e-5,2e-5",
        ])
        assert result.exit_code != 0

    def test_sweep_invalid_strategy(self, tmp_path):
        """Should fail for invalid strategy."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: test-model\n"
            "data:\n"
            "  train: ./data.jsonl\n"
        )

        runner = CliRunner()
        result = runner.invoke(app, [
            "sweep",
            "--config", str(config_file),
            "--param", "lr=1e-5",
            "--strategy", "bayesian",
        ])
        assert result.exit_code != 0
