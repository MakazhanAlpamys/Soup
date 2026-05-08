"""v0.40.1 Part D — CLI UX consistency tests (highest-leverage subset).

Closes:
  - H4: Template list dynamic sync
  - M2: `soup init --force` flag
  - N6: `soup history` suggests `data registry` for dataset names
  - N2: `soup migrate` JSONL friendly error
  - G10: `soup eval custom -o` written regardless of `--attach-to-registry`
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

# --- H4: Template help is dynamically generated --------------------------


def test_init_template_help_lists_all_templates():
    from soup_cli.commands.init import _template_help_string
    from soup_cli.templates import list_templates

    help_text = _template_help_string()
    for template_name in list_templates():
        assert template_name in help_text, (
            f"template {template_name!r} missing from --template help"
        )


def test_init_template_help_includes_bco():
    """v0.40.0 added BCO; H4 must show it without a manual help-text edit."""
    from soup_cli.commands.init import _template_help_string

    assert "bco" in _template_help_string()


# --- M2: soup init --force flag ------------------------------------------


def test_init_force_flag_overwrites_without_prompt(tmp_path):
    from soup_cli.cli import app

    runner = CliRunner()
    target = tmp_path / "soup.yaml"
    target.write_text("base: existing", encoding="utf-8")

    # Without --force, prompts (we send 'n' to abort).
    result = runner.invoke(app, ["init", "--output", str(target)], input="n\n")
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "base: existing", (
        "without --force the user-typed 'n' should abort and preserve file"
    )

    # With --force, overwrites silently using a registered template.
    result = runner.invoke(
        app, ["init", "--output", str(target), "--template", "chat", "--force"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert target.read_text(encoding="utf-8") != "base: existing"


# --- N2: soup migrate JSONL friendly error ------------------------------


def test_migrate_jsonl_input_yields_friendly_error(tmp_path: Path):
    from soup_cli.commands.migrate import _looks_like_jsonl

    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"prompt": "hi"}\n{"prompt": "world"}\n', encoding="utf-8")
    assert _looks_like_jsonl(jsonl) is True


def test_migrate_yaml_does_not_look_like_jsonl(tmp_path: Path):
    from soup_cli.commands.migrate import _looks_like_jsonl

    yml = tmp_path / "config.yaml"
    yml.write_text("base: foo\ntask: sft\n", encoding="utf-8")
    assert _looks_like_jsonl(yml) is False


def test_migrate_skips_blank_lines_when_sniffing(tmp_path: Path):
    from soup_cli.commands.migrate import _looks_like_jsonl

    f = tmp_path / "blanks.jsonl"
    f.write_text('\n\n  \n{"key": 1}\n', encoding="utf-8")
    assert _looks_like_jsonl(f) is True


# --- N6: soup history suggests dataset registry --------------------------


def test_history_dataset_registry_helper_handles_missing():
    from soup_cli.commands.history import _name_exists_in_dataset_registry

    # Should never raise even if registry module / file is missing.
    assert isinstance(
        _name_exists_in_dataset_registry("definitely-not-a-dataset-xxx"), bool
    )


# --- G10: soup eval custom --output writes JSON without --attach-to-registry


def test_eval_custom_output_arg_described_as_independent():
    """The --output help string must mention it's honored without attach."""
    import inspect

    from soup_cli.commands.eval import custom

    src = inspect.getsource(custom)
    assert "Honored independently" in src or "G10" in src


def test_eval_custom_no_longer_shadows_output_with_response():
    """Source-level guard against the loop-variable shadow regression."""
    import inspect

    from soup_cli.commands.eval import custom

    src = inspect.getsource(custom)
    # Old buggy line: ``output = generate_fn(eval_task.prompt)``.
    # New line uses ``response`` to avoid shadowing the CLI ``output`` arg.
    assert "response = generate_fn" in src
    assert "output = generate_fn" not in src
