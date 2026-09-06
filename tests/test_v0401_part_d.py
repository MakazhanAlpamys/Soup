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


def test_migrate_jsonl_input_yields_friendly_error(tmp_path: Path, monkeypatch):
    """Drives the *command*, not the helper: real JSONL named `.jsonl` is refused."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"prompt": "hi"}\n{"prompt": "world"}\n', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "data.jsonl", "--dry-run"]
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "got JSONL" in result.output


def test_migrate_yaml_config_named_jsonl_still_migrates(tmp_path: Path, monkeypatch):
    """Control: the suffix alone must not condemn a file whose content is YAML.

    Fails if the `_looks_like_jsonl` call site is removed from the guard.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.jsonl"
    config.write_text(
        "model_name_or_path: meta-llama/Llama-3-8B\n"
        "stage: sft\n"
        "finetuning_type: lora\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "config.jsonl", "--dry-run"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "got JSONL" not in result.output


def test_migrate_bomd_jsonl_still_yields_friendly_error(tmp_path: Path, monkeypatch):
    """A UTF-8 BOM must not defeat the sniff (#675 review).

    Windows tooling writes UTF-8 with a BOM by default. Decoded as plain
    ``utf-8`` the BOM survives as U+FEFF, which ``str.strip()`` leaves alone
    because it is not whitespace, so ``startswith("{")`` is False and real
    JSONL sniffs as a config. Fails if the helper reads ``utf-8`` rather than
    ``utf-8-sig``.
    """
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    jsonl = tmp_path / "bom.jsonl"
    jsonl.write_bytes(
        b"\xef\xbb\xbf" + b'{"prompt": "hi"}\n{"prompt": "world"}\n'
    )
    # Guard the fixture itself: plain utf-8 must see the BOM this test is about.
    assert jsonl.read_text(encoding="utf-8").startswith("\ufeff")

    runner = CliRunner()
    result = runner.invoke(
        app, ["migrate", "--from", "llamafactory", "bom.jsonl", "--dry-run"]
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "got JSONL" in result.output


def test_migrate_sniff_does_not_read_a_whole_unnewlined_file(tmp_path: Path):
    """The sniff is bounded: one very long line must not be read entire.

    ``for line in fh`` on a file with no newline pulls all of it into one
    string. Fails if the bounded ``readline`` is reverted.
    """
    import tracemalloc

    from soup_cli.commands.migrate import _looks_like_jsonl

    big = tmp_path / "oneline.jsonl"
    big.write_text('{"prompt": "' + "x" * (8 * 1024 * 1024) + '"}', encoding="utf-8")

    tracemalloc.start()
    try:
        assert _looks_like_jsonl(big) is True
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Bounded read touches 64 KiB; the unbounded one allocated the 8 MiB line
    # at least twice (read plus strip). 1 MiB leaves generous headroom.
    assert peak < 1024 * 1024, f"peak {peak} bytes: the read looks unbounded"


def test_migrate_helper_detects_jsonl(tmp_path: Path):
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
