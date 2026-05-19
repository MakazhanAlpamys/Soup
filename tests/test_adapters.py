"""Tests for soup adapters — adapter management command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def _create_adapter(path: Path, base_model: str = "meta-llama/Llama-3.1-8B",
                    lora_r: int = 16, lora_alpha: int = 32,
                    task_type: str = "CAUSAL_LM") -> Path:
    """Create a fake adapter directory with adapter_config.json."""
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": base_model,
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.05,
        "task_type": task_type,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "peft_type": "LORA",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    # Create a small dummy safetensors file (just for size detection)
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    return path


class TestAdaptersList:
    """Test soup adapters list command."""

    def test_list_finds_adapters(self, tmp_path):
        """List should find adapters in directory."""
        _create_adapter(tmp_path / "output" / "checkpoint-100")
        _create_adapter(tmp_path / "output" / "checkpoint-200")
        result = runner.invoke(app, ["adapters", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "checkpoint-100" in result.output
        assert "checkpoint-200" in result.output

    def test_list_no_adapters(self, tmp_path):
        """List should handle directory with no adapters."""
        result = runner.invoke(app, ["adapters", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "No adapters found" in result.output

    def test_list_nonexistent_dir(self):
        """List should fail for nonexistent directory."""
        result = runner.invoke(app, ["adapters", "list", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_list_shows_base_model(self, tmp_path):
        """List output shows base model name."""
        _create_adapter(
            tmp_path / "my-adapter",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
        )
        result = runner.invoke(app, ["adapters", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "Llama-3.1-8B" in result.output


class TestAdaptersInfo:
    """Test soup adapters info command."""

    def test_info_shows_metadata(self, tmp_path):
        """Info should show adapter metadata."""
        adapter_path = _create_adapter(
            tmp_path / "my-adapter",
            base_model="meta-llama/Llama-3.1-8B",
            lora_r=16,
            lora_alpha=32,
        )
        result = runner.invoke(app, ["adapters", "info", str(adapter_path)])
        assert result.exit_code == 0
        assert "Llama-3.1-8B" in result.output
        assert "16" in result.output  # lora_r
        assert "32" in result.output  # lora_alpha

    def test_info_shows_target_modules(self, tmp_path):
        """Info should show target modules."""
        adapter_path = _create_adapter(tmp_path / "adapter")
        result = runner.invoke(app, ["adapters", "info", str(adapter_path)])
        assert result.exit_code == 0
        assert "q_proj" in result.output

    def test_info_nonexistent_path(self):
        """Info should fail for nonexistent adapter."""
        result = runner.invoke(app, ["adapters", "info", "/nonexistent/adapter"])
        assert result.exit_code != 0

    def test_info_no_adapter_config(self, tmp_path):
        """Info should fail if directory has no adapter_config.json."""
        tmp_path.mkdir(exist_ok=True)
        result = runner.invoke(app, ["adapters", "info", str(tmp_path)])
        assert result.exit_code != 0

    def test_info_shows_disk_size(self, tmp_path):
        """Info should show approximate size on disk."""
        adapter_path = _create_adapter(tmp_path / "adapter")
        result = runner.invoke(app, ["adapters", "info", str(adapter_path)])
        assert result.exit_code == 0
        # Should show some size info
        assert "Size" in result.output or "KB" in result.output or "MB" in result.output


class TestAdaptersCompare:
    """Test soup adapters compare command."""

    def test_compare_two_adapters(self, tmp_path):
        """Compare two adapters side-by-side."""
        adapter1 = _create_adapter(
            tmp_path / "adapter1",
            base_model="meta-llama/Llama-3.1-8B",
            lora_r=16, lora_alpha=32,
        )
        adapter2 = _create_adapter(
            tmp_path / "adapter2",
            base_model="meta-llama/Llama-3.1-8B",
            lora_r=64, lora_alpha=128,
        )
        result = runner.invoke(app, [
            "adapters", "compare",
            str(adapter1), str(adapter2),
        ])
        assert result.exit_code == 0
        assert "16" in result.output  # adapter1 r
        assert "64" in result.output  # adapter2 r

    def test_compare_different_base_models(self, tmp_path):
        """Compare highlights different base models."""
        adapter1 = _create_adapter(
            tmp_path / "a1", base_model="meta-llama/Llama-3.1-8B",
        )
        adapter2 = _create_adapter(
            tmp_path / "a2", base_model="Qwen/Qwen2.5-7B",
        )
        result = runner.invoke(app, [
            "adapters", "compare", str(adapter1), str(adapter2),
        ])
        assert result.exit_code == 0
        assert "Llama-3.1-8B" in result.output
        assert "Qwen2.5-7B" in result.output

    def test_compare_nonexistent_adapter(self, tmp_path):
        """Compare should fail if one adapter doesn't exist."""
        adapter1 = _create_adapter(tmp_path / "adapter1")
        result = runner.invoke(app, [
            "adapters", "compare",
            str(adapter1), "/nonexistent",
        ])
        assert result.exit_code != 0


class TestAdaptersMarkupEscape:
    """Regression tests for Rich-markup escape across list/info/compare.

    Closes issue #174 (v0.57.0 Known Limitation 9) — a crafted
    `adapter_config.json` value like `base_model_name_or_path: "[link=evil]
    click[/]"` previously rendered as live markup in the terminal.
    Every adapter-config-sourced value must pass through `rich.markup.escape`
    before embedding in a Rich f-string.
    """

    CRAFTED_MARKUP = "[link=http://evil.example/]click-me[/]"
    CRAFTED_BOLD = "[red bold]injected[/]"

    def test_list_escapes_crafted_base_model(self, tmp_path):
        """list: crafted base_model renders as literal `[link=...]` text."""
        _create_adapter(tmp_path / "adapter_a", base_model=self.CRAFTED_MARKUP)
        result = runner.invoke(app, ["adapters", "list", str(tmp_path)])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Literal `[link=` substring must survive in output (escaped form
        # uses \[link= in Rich-rendered ANSI; absence of an actual ANSI
        # link sequence is the security signal).
        assert "click-me" in result.output
        assert "\x1b]8;" not in result.output, (
            "Rich rendered crafted [link=...] as a real ANSI hyperlink "
            "— escape() is not being applied to base_model_name_or_path."
        )

    def test_info_escapes_crafted_base_model(self, tmp_path):
        """info: crafted base_model in Panel renders as literal text."""
        adapter = tmp_path / "adapter_info"
        _create_adapter(adapter, base_model=self.CRAFTED_MARKUP)
        result = runner.invoke(app, ["adapters", "info", str(adapter)])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "click-me" in result.output
        assert "\x1b]8;" not in result.output, (
            "Rich rendered crafted [link=...] as a real ANSI hyperlink "
            "in `adapters info` — escape() missing on base_model."
        )

    def test_compare_escapes_equal_crafted_values(self, tmp_path):
        """compare: shared crafted value MUST escape even when highlight branch skips.

        Regression for the security gap surfaced in PR #175 review:
        the original patch only escaped inside the `val1_str != val2_str`
        branch, leaving equal-value rows un-escaped.
        """
        a = tmp_path / "adapter_left"
        b = tmp_path / "adapter_right"
        _create_adapter(a, base_model=self.CRAFTED_BOLD)
        _create_adapter(b, base_model=self.CRAFTED_BOLD)  # IDENTICAL crafted value
        result = runner.invoke(app, ["adapters", "compare", str(a), str(b)])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Literal "injected" text must appear (the [red bold] tag must
        # NOT be interpreted as markup styling).
        assert "injected" in result.output
        # A live `[red bold]` rendered tag would emit ANSI color codes
        # for the inner text alone; assert the raw bracketed token
        # literal is present (escape() preserves it).
        assert "[red bold]" in result.output or r"\[red bold]" in result.output

    def test_compare_escapes_differing_crafted_values(self, tmp_path):
        """compare: crafted values that DIFFER (highlight branch fires) also escape."""
        a = tmp_path / "adapter_left"
        b = tmp_path / "adapter_right"
        _create_adapter(a, base_model=self.CRAFTED_BOLD)
        _create_adapter(b, base_model="meta-llama/Llama-3.1-8B")  # benign
        result = runner.invoke(app, ["adapters", "compare", str(a), str(b)])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "injected" in result.output


class TestAdapterDiscovery:
    """Test adapter discovery helper function."""

    def test_find_adapters_recursive(self, tmp_path):
        from soup_cli.commands.adapters import _find_adapters

        _create_adapter(tmp_path / "output" / "checkpoint-100")
        _create_adapter(tmp_path / "output" / "checkpoint-200")
        _create_adapter(tmp_path / "other" / "model")

        adapters = _find_adapters(tmp_path)
        assert len(adapters) == 3

    def test_find_adapters_empty(self, tmp_path):
        from soup_cli.commands.adapters import _find_adapters

        adapters = _find_adapters(tmp_path)
        assert adapters == []

    def test_find_adapters_respects_max_depth(self, tmp_path):
        """Adapters beyond max_depth should not be found."""
        from soup_cli.commands.adapters import _find_adapters

        deep = tmp_path
        for _ in range(8):
            deep = deep / "sub"
        _create_adapter(deep)
        adapters = _find_adapters(tmp_path, max_depth=6)
        assert len(adapters) == 0

    def test_read_adapter_config(self, tmp_path):
        from soup_cli.commands.adapters import _read_adapter_config

        adapter_path = _create_adapter(
            tmp_path / "adapter",
            base_model="test-model",
            lora_r=32,
        )
        config = _read_adapter_config(adapter_path)
        assert config["base_model_name_or_path"] == "test-model"
        assert config["r"] == 32
