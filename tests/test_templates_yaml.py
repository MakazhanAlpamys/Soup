"""Tests for v0.39.0 Part E — template registry YAML migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTemplateRegistry:
    def test_list_templates_includes_core_set(self):
        from soup_cli.templates import list_templates
        names = list_templates()
        for required in (
            "chat", "code", "reasoning", "vision", "kto", "orpo",
            "simpo", "ipo", "pretrain", "moe", "longcontext", "embedding",
            "audio", "tool-calling", "rlhf", "medical",
        ):
            assert required in names, f"{required} missing from {names}"

    def test_list_templates_returns_sorted(self):
        from soup_cli.templates import list_templates
        names = list_templates()
        assert names == sorted(names)

    def test_load_template_chat_from_yaml(self):
        from soup_cli.templates import load_template
        body = load_template("chat")
        assert body is not None
        assert "Soup template: Chat Assistant" in body
        assert "task: sft" in body

    def test_load_template_unknown_returns_none(self):
        from soup_cli.templates import load_template
        assert load_template("does-not-exist") is None

    def test_load_template_rejects_path_traversal(self):
        from soup_cli.templates import load_template
        with pytest.raises(ValueError):
            load_template("../../etc/passwd")
        with pytest.raises(ValueError):
            load_template("foo/bar")
        with pytest.raises(ValueError):
            load_template("foo\\bar")

    def test_load_template_rejects_null_byte(self):
        from soup_cli.templates import load_template
        with pytest.raises(ValueError):
            load_template("chat\x00malicious")

    def test_load_template_rejects_empty_name(self):
        from soup_cli.templates import load_template
        with pytest.raises(ValueError):
            load_template("")

    def test_yaml_files_exist_for_all_inline(self):
        from soup_cli.config.schema import TEMPLATES
        templates_dir = Path(__file__).resolve().parent.parent / "soup_cli" / "templates"
        for name in TEMPLATES:
            yaml_path = templates_dir / f"{name}.yaml"
            assert yaml_path.is_file(), f"Missing YAML for inline template {name}"

    def test_manifest_well_formed(self):
        templates_dir = Path(__file__).resolve().parent.parent / "soup_cli" / "templates"
        manifest_path = templates_dir / "manifest.json"
        assert manifest_path.is_file()
        with manifest_path.open() as f:
            data = json.load(f)
        assert "templates" in data
        assert "version" in data
        assert isinstance(data["templates"], dict)

    def test_yaml_content_matches_inline_for_all_templates(self):
        """All 16 inline templates must match their YAML siblings exactly.

        v0.39.0 Part E ships both sources for back-compat; drift between
        them is a contributor footgun. This guards against silent edits
        of one source without the other.
        """
        from soup_cli.config.schema import TEMPLATES
        from soup_cli.templates import load_template
        for name in TEMPLATES:
            assert load_template(name) == TEMPLATES[name], (
                f"YAML / inline drift for template {name!r}"
            )


class TestSecurityFallbacks:
    def test_oversized_file_falls_back_to_inline(self, tmp_path, monkeypatch):
        """v0.39.0 — _MAX_TEMPLATE_BYTES guard."""
        import soup_cli.templates as tpl_mod

        fake_dir = tmp_path / "templates"
        fake_dir.mkdir()
        (fake_dir / "manifest.json").write_text(
            json.dumps({"templates": {"chat": "chat.yaml"}, "version": 1})
        )
        # Write a 300 KB file (> 256 KB cap)
        (fake_dir / "chat.yaml").write_text("x" * (300 * 1024))

        monkeypatch.setattr(tpl_mod, "_templates_dir", lambda: fake_dir)
        # Should fall back to inline TEMPLATES["chat"], not return the giant blob.
        body = tpl_mod.load_template("chat")
        from soup_cli.config.schema import TEMPLATES
        assert body == TEMPLATES["chat"]
        assert len(body) < 256 * 1024

    def test_crafted_manifest_outside_dir_falls_back(self, tmp_path, monkeypatch):
        """A manifest entry with a relative path that escapes must be ignored."""
        import soup_cli.templates as tpl_mod

        fake_dir = tmp_path / "templates"
        fake_dir.mkdir()
        # Create a file OUTSIDE the templates dir
        (tmp_path / "secret.yaml").write_text("LEAKED CONTENT")
        # Crafted manifest pointing to it via filename traversal-like component
        # _validate_name will reject any name with `..`/`/`/`\` so this also
        # exercises the manifest-tampering rejection chain.
        (fake_dir / "manifest.json").write_text(
            json.dumps({"templates": {"chat": "../secret.yaml"}, "version": 1})
        )

        monkeypatch.setattr(tpl_mod, "_templates_dir", lambda: fake_dir)
        body = tpl_mod.load_template("chat")
        from soup_cli.config.schema import TEMPLATES
        # Must NOT contain the leaked content; falls back to inline.
        assert "LEAKED CONTENT" not in (body or "")
        assert body == TEMPLATES["chat"]


class TestInitUsesRegistry:
    def test_init_command_lists_template_options(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["init", "--help"])
        # init --help should still succeed after the migration
        assert result.exit_code == 0, (result.output, repr(result.exception))
