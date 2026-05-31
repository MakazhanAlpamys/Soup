"""v0.64.0 Part E — Shell completions with config introspection."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_module_imports():
    from soup_cli.utils import completions

    assert hasattr(completions, "SUPPORTED_SHELLS")
    assert hasattr(completions, "validate_shell")
    assert hasattr(completions, "render_bash_script")
    assert hasattr(completions, "render_zsh_script")
    assert hasattr(completions, "render_fish_script")
    assert hasattr(completions, "render_completion_script")
    assert hasattr(completions, "complete_recipe_name")
    assert hasattr(completions, "complete_target_modules")


# ---------------------------------------------------------------------------
# SUPPORTED_SHELLS
# ---------------------------------------------------------------------------


def test_supported_shells():
    from soup_cli.utils.completions import SUPPORTED_SHELLS

    assert "bash" in SUPPORTED_SHELLS
    assert "zsh" in SUPPORTED_SHELLS
    assert "fish" in SUPPORTED_SHELLS


def test_supported_shells_is_frozenset():
    from soup_cli.utils.completions import SUPPORTED_SHELLS

    assert isinstance(SUPPORTED_SHELLS, frozenset)


# ---------------------------------------------------------------------------
# validate_shell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", ["bash", "zsh", "fish"])
def test_validate_shell_happy(v):
    from soup_cli.utils.completions import validate_shell

    assert validate_shell(v) == v


def test_validate_shell_case_insensitive():
    from soup_cli.utils.completions import validate_shell

    assert validate_shell("BASH") == "bash"
    assert validate_shell("Fish") == "fish"


@pytest.mark.parametrize(
    "bad",
    [True, False, None, "", "tcsh", "powershell", "ksh", "csh", "x" * 33],
)
def test_validate_shell_rejects(bad):
    from soup_cli.utils.completions import validate_shell

    with pytest.raises((TypeError, ValueError)):
        validate_shell(bad)


def test_validate_shell_rejects_null_byte():
    from soup_cli.utils.completions import validate_shell

    with pytest.raises(ValueError, match="null"):
        validate_shell("ba\x00sh")


# ---------------------------------------------------------------------------
# render_*_script
# ---------------------------------------------------------------------------


def test_render_bash_script_basic():
    from soup_cli.utils.completions import render_bash_script

    text = render_bash_script()
    assert "_soup_complete" in text or "complete -F" in text
    assert "soup" in text


def test_render_zsh_script_basic():
    from soup_cli.utils.completions import render_zsh_script

    text = render_zsh_script()
    assert "#compdef soup" in text or "_soup" in text


def test_render_fish_script_basic():
    from soup_cli.utils.completions import render_fish_script

    text = render_fish_script()
    assert "complete -c soup" in text or "complete --command soup" in text


def test_render_completion_script_dispatch():
    from soup_cli.utils.completions import render_completion_script

    bash = render_completion_script("bash")
    assert "soup" in bash
    zsh = render_completion_script("zsh")
    assert "soup" in zsh
    fish = render_completion_script("fish")
    assert "soup" in fish


def test_render_completion_script_rejects_unknown():
    from soup_cli.utils.completions import render_completion_script

    with pytest.raises(ValueError):
        render_completion_script("tcsh")


# ---------------------------------------------------------------------------
# complete_recipe_name
# ---------------------------------------------------------------------------


def test_complete_recipe_name_returns_list():
    from soup_cli.utils.completions import complete_recipe_name

    suggestions = complete_recipe_name("")
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0


def test_complete_recipe_name_filters_prefix():
    from soup_cli.utils.completions import complete_recipe_name

    suggestions = complete_recipe_name("llama")
    # Every result must start with the prefix (case-insensitive)
    for s in suggestions:
        assert s.lower().startswith("llama")


def test_complete_recipe_name_empty_for_nonsense():
    from soup_cli.utils.completions import complete_recipe_name

    suggestions = complete_recipe_name("definitely-not-a-recipe-zzzzzz")
    assert suggestions == []


def test_complete_recipe_name_rejects_bool():
    from soup_cli.utils.completions import complete_recipe_name

    with pytest.raises(TypeError):
        complete_recipe_name(True)  # type: ignore[arg-type]


def test_complete_recipe_name_null_byte_returns_empty():
    from soup_cli.utils.completions import complete_recipe_name

    # Defensive: null byte returns empty rather than raising.
    assert complete_recipe_name("foo\x00bar") == []


# ---------------------------------------------------------------------------
# complete_target_modules
# ---------------------------------------------------------------------------


def test_complete_target_modules_default():
    from soup_cli.utils.completions import complete_target_modules

    # With no base, returns canonical Llama-shape modules
    suggestions = complete_target_modules("", base=None)
    assert "q_proj" in suggestions
    assert "k_proj" in suggestions


def test_complete_target_modules_filters_prefix():
    from soup_cli.utils.completions import complete_target_modules

    suggestions = complete_target_modules("q_", base=None)
    for s in suggestions:
        assert s.startswith("q_")


def test_complete_target_modules_rejects_bool():
    from soup_cli.utils.completions import complete_target_modules

    with pytest.raises(TypeError):
        complete_target_modules(True, base=None)  # type: ignore[arg-type]


def test_complete_target_modules_handles_unknown_base():
    """When base is set but we can't probe it, fall back to default modules."""
    from soup_cli.utils.completions import complete_target_modules

    suggestions = complete_target_modules("", base="some/nonexistent-model")
    # Should still return non-empty default
    assert len(suggestions) > 0


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_completions_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["completions", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_completions_bash():
    from soup_cli.cli import app

    result = runner.invoke(app, ["completions", "bash"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "soup" in result.stdout


def test_cli_completions_zsh():
    from soup_cli.cli import app

    result = runner.invoke(app, ["completions", "zsh"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_completions_fish():
    from soup_cli.cli import app

    result = runner.invoke(app, ["completions", "fish"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_completions_unknown_shell():
    from soup_cli.cli import app

    result = runner.invoke(app, ["completions", "tcsh"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Source-wiring regression
# ---------------------------------------------------------------------------


def test_no_heavy_top_level_imports():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "completions.py"
    text = src.read_text(encoding="utf-8")
    import re
    for bad in ["^import torch", "^from torch", "^import transformers", "^from transformers"]:
        assert not re.search(bad, text, re.MULTILINE)
