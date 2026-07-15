"""v0.71.35 — Compliance pack.

Covers:
  * Compliance init templates (hipaa / soc2 / eu-ai-act / sr-11-7)
  * ``soup card`` — HF model-card autogen from a registry entry
  * ``soup ci init`` — render .github/workflows/soup-gate.yml
"""

from __future__ import annotations

import json
import os

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.templates import list_templates, load_template

runner = CliRunner()

COMPLIANCE_TEMPLATES = ["hipaa", "soc2", "eu-ai-act", "sr-11-7"]


# --------------------------------------------------------------------------- #
# Compliance init templates
# --------------------------------------------------------------------------- #
class TestComplianceTemplates:
    @pytest.mark.parametrize("name", COMPLIANCE_TEMPLATES)
    def test_template_is_listed(self, name):
        assert name in list_templates()

    @pytest.mark.parametrize("name", COMPLIANCE_TEMPLATES)
    def test_template_loads(self, name):
        body = load_template(name)
        assert body is not None and body.strip()

    @pytest.mark.parametrize("name", COMPLIANCE_TEMPLATES)
    def test_template_parses_as_config(self, name):
        """The YAML body must be a valid SoupConfig (train-time keys only)."""
        body = load_template(name)
        cfg = load_config_from_string(body)
        assert cfg.base
        assert cfg.task

    @pytest.mark.parametrize("name", COMPLIANCE_TEMPLATES)
    def test_template_has_compliance_guidance(self, name):
        """Compliance behaviours are CLI flags/commands, not schema keys, so the
        template must carry header-comment guidance pointing users at them."""
        body = load_template(name)
        lowered = body.lower()
        # every compliance template should mention the audit log + at least one
        # provenance command
        assert "audit" in lowered
        assert any(tok in lowered for tok in ("bom", "attest", "repro-receipt", "sign"))

    def test_manifest_lists_compliance_templates(self):
        import soup_cli.templates as tpl

        manifest = tpl._load_manifest()
        for name in COMPLIANCE_TEMPLATES:
            assert name in manifest["templates"]

    def test_init_writes_compliance_template(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        out = tmp_path / "soup.yaml"
        result = runner.invoke(app, ["init", "--template", "hipaa", "-o", str(out), "--force"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.exists()
        cfg = load_config_from_string(out.read_text(encoding="utf-8"))
        assert cfg.base


# --------------------------------------------------------------------------- #
# soup card — model-card autogen from a registry entry
# --------------------------------------------------------------------------- #
def _make_entry(db_path, *, notes=None, with_eval=False, with_artifact=None, parent=False):
    """Seed a registry entry (and optional artifact / lineage) in a temp DB."""
    from soup_cli.registry.store import RegistryStore

    with RegistryStore(db_path=db_path) as store:
        parent_id = None
        if parent:
            parent_id = store.push(
                name="base-run", tag="v1", base_model="Qwen/Qwen2.5-7B-Instruct",
                task="sft", run_id=None, config={"base": "Qwen/Qwen2.5-7B-Instruct"},
                notes=None,
            )
        eid = store.push(
            name="phi-model", tag="v1", base_model="Qwen/Qwen2.5-7B-Instruct",
            task="sft", run_id=None,
            config={"base": "Qwen/Qwen2.5-7B-Instruct", "task": "sft",
                    "training": {"epochs": 3, "lr": 2e-5}},
            notes=notes,
        )
        if parent_id:
            store.add_lineage(child_id=eid, parent_id=parent_id, relation="forked_from")
        if with_artifact is not None:
            store.add_artifact(entry_id=eid, kind=with_artifact[0],
                               path=with_artifact[1], enforce_cwd=True)
    return eid


class TestBuildModelCard:
    def test_pure_card_has_core_sections(self):
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "reg_x", "name": "my-model", "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "task": "sft", "created_at": "2026-07-15T00:00:00", "notes": "clean run",
            "config_hash": "a" * 64, "data_hash": "b" * 64, "run_id": None,
            "tags": ["prod"],
            "config_json": json.dumps({"base": "Qwen/Qwen2.5-7B-Instruct", "task": "sft",
                                       "training": {"epochs": 3, "lr": 2e-5}}),
        }
        md = build_model_card(entry, [], [], [])
        assert md.startswith("---")  # YAML frontmatter
        assert "# my-model" in md
        assert "Qwen/Qwen2.5-7B-Instruct" in md
        assert "## Training" in md

    def test_notes_html_escaped(self):
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "reg_x", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": "<script>alert(1)</script>",
            "config_hash": "", "data_hash": "", "run_id": None, "tags": [],
            "config_json": "{}",
        }
        md = build_model_card(entry, [], [], [])
        assert "<script>" not in md
        assert "&lt;script&gt;" in md

    def test_eval_scorecard_rendered(self):
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": "run1", "tags": [], "config_json": "{}",
        }
        evals = [{"benchmark": "mmlu", "score": 0.61}]
        md = build_model_card(entry, [], evals, [])
        assert "## Evaluation" in md
        assert "mmlu" in md
        assert "0.610" in md

    def test_frontmatter_yaml_safe_base(self):
        """A hostile base_model must not break the YAML frontmatter."""
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": 'evil"\ninjected: true',
            "task": "sft", "created_at": "t", "notes": None,
            "config_hash": "", "data_hash": "", "run_id": None, "tags": [],
            "config_json": "{}",
        }
        md = build_model_card(entry, [], [], [])
        front = md.split("---", 2)[1]
        parsed = yaml.safe_load(front)
        assert "injected" not in parsed  # injection neutralised


class TestCardCli:
    def test_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["card", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_card_happy(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        eid = _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["card", eid, "-o", "CARD.md"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        card = (tmp_path / "CARD.md").read_text(encoding="utf-8")
        assert "phi-model" in card
        assert "Qwen/Qwen2.5-7B-Instruct" in card

    def test_card_with_lineage_and_artifact(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        monkeypatch.chdir(tmp_path)
        gguf = tmp_path / "model.q4_k_m.gguf"
        gguf.write_text("stub", encoding="utf-8")
        eid = _make_entry(db, parent=True, with_artifact=("gguf", str(gguf)))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        result = runner.invoke(app, ["card", eid, "-o", "CARD.md"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        card = (tmp_path / "CARD.md").read_text(encoding="utf-8")
        assert "base-run" in card  # ancestor
        assert "model.q4_k_m.gguf" in card  # artifact link

    def test_card_not_found(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["card", "reg_nonexistent", "-o", "CARD.md"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_card_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        eid = _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        result = runner.invoke(app, ["card", eid, "-o", "../escape.md"])
        assert result.exit_code == 1
        assert not (tmp_path / "escape.md").exists()


class TestPushCardRider:
    def test_push_has_card_option(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["push", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--card" in result.output

    def test_build_card_for_ref_unknown_raises(self, tmp_path, monkeypatch):
        from soup_cli.commands.card import CardError, build_card_for_ref

        db = tmp_path / "reg.db"
        _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(CardError):
            build_card_for_ref("reg_does_not_exist")


# --------------------------------------------------------------------------- #
# soup ci init — render .github/workflows/soup-gate.yml
# --------------------------------------------------------------------------- #
class TestRenderWorkflow:
    def test_render_is_valid_yaml_with_gate_steps(self):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        body = render_soup_gate_workflow(
            data_path="./data/train.jsonl",
            suite_path="expectations.yaml",
            evidence_path="ship_evidence.json",
        )
        doc = yaml.safe_load(body)
        assert "jobs" in doc
        steps = doc["jobs"]["soup-gate"]["steps"]
        runs = " ".join(s.get("run", "") for s in steps)
        assert "soup data validate" in runs
        assert "soup expect" in runs
        assert "soup ship --evidence" in runs

    def test_render_defaults_python_and_branch(self):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        body = render_soup_gate_workflow(
            data_path="data/train.jsonl",
            suite_path="suite.yaml",
            evidence_path="ev.json",
        )
        yaml.safe_load(body)  # must parse
        assert '3.11' in body
        # `on:` parses to True in YAML 1.1 (the "Norway problem"); assert via body
        assert "pull_request" in body

    def test_render_shell_quotes_injection(self):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        body = render_soup_gate_workflow(
            data_path="data/train.jsonl; rm -rf /",
            suite_path="suite.yaml",
            evidence_path="ev.json",
        )
        # the malicious path must be shell-quoted as a single token, not left
        # as a chainable command
        assert "'data/train.jsonl; rm -rf /'" in body

    @pytest.mark.parametrize("bad", ["", "a\nb", "a\x00b", "../escape.jsonl"])
    def test_render_rejects_bad_path(self, bad):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        with pytest.raises((ValueError, TypeError)):
            render_soup_gate_workflow(
                data_path=bad, suite_path="s.yaml", evidence_path="e.json",
            )

    @pytest.mark.parametrize("bad_py", ["3", "3.x", "3.11; rm", "abc"])
    def test_render_rejects_bad_python(self, bad_py):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        with pytest.raises(ValueError):
            render_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
                python_version=bad_py,
            )

    @pytest.mark.parametrize("bad_branch", ["a b", "a$b", "a;b", ""])
    def test_render_rejects_bad_branch(self, bad_branch):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        with pytest.raises(ValueError):
            render_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
                branch=bad_branch,
            )


class TestCiInitCli:
    def test_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["ci", "init", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_ci_init_writes_workflow(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["ci", "init", "--data", "data/train.jsonl"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        wf = tmp_path / ".github" / "workflows" / "soup-gate.yml"
        assert wf.exists()
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        assert "soup-gate" in doc["jobs"]

    def test_ci_init_refuses_overwrite_without_force(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        first = runner.invoke(app, ["ci", "init"])
        assert first.exit_code == 0, (first.output, repr(first.exception))
        second = runner.invoke(app, ["ci", "init"])
        assert second.exit_code == 1
        assert "exists" in second.output.lower()
        forced = runner.invoke(app, ["ci", "init", "--force"])
        assert forced.exit_code == 0, (forced.output, repr(forced.exception))

    def test_ci_init_rejects_bad_python(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["ci", "init", "--python", "3.x"])
        assert result.exit_code == 1
        assert not (tmp_path / ".github" / "workflows" / "soup-gate.yml").exists()


class TestCodeReviewFixes:
    """Regression pins for the v0.71.35 code-review findings."""

    def test_workflow_installs_published_package_not_editable(self):
        """HIGH: `pip install -e ".[dev]"` only works inside the Soup source
        tree, but `soup ci init` targets a DOWNSTREAM fine-tuning repo (no
        pyproject.toml), so the workflow's first step would break for every
        real user."""
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        body = render_soup_gate_workflow(
            data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
        )
        assert "pip install -e" not in body
        assert "pip install soup-cli" in body

    def test_usage_snippet_survives_quote_in_base_model(self):
        """LOW: a `\"` in base_model must not emit broken Python."""
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": 'we"ird/model',
            "task": "sft", "created_at": "t", "notes": None,
            "config_hash": "", "data_hash": "", "run_id": None, "tags": [],
            "config_json": "{}",
        }
        md = build_model_card(entry, [{"kind": "adapter", "path": "a", "sha256": "x"}], [], [])
        # extract the python usage snippet and compile it
        snippet = md.split("```python", 1)[1].split("```", 1)[0]
        compile(snippet, "<card>", "exec")  # raises SyntaxError if broken


class TestSecurityReviewFixes:
    """Regression pins for the v0.71.35 security-review findings."""

    def test_training_section_escapes_hostile_base(self):
        """HIGH: `base`/`scheduler` have no charset validator in SoupConfig, so
        a crafted config could smuggle raw HTML (or a backtick breaking out of
        the code span) into a card published to the public HF Hub."""
        from soup_cli.commands.card import build_model_card

        hostile = '` </code><script>alert(1)</script><code> `'
        entry = {
            "id": "r", "name": "m", "base_model": "safe/base", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": None, "tags": [],
            "config_json": json.dumps({
                "base": hostile, "task": "sft",
                "training": {"epochs": 1, "scheduler": "<img src=x onerror=alert(1)>"},
            }),
        }
        md = build_model_card(entry, [], [], [])
        # The security property is that no HTML TAG survives — angle brackets and
        # backticks are neutralised, so the payload is inert prose. (The bare
        # word "onerror" surviving as plain text is harmless.)
        assert "<script>" not in md
        assert "<img" not in md
        assert "</code>" not in md
        # and the backtick cannot break out of the `code span`
        training_line = next(li for li in md.splitlines() if li.startswith("- **Base model:**"))
        assert training_line.count("`") == 2

    def test_safe_md_cell_strips_control_bytes(self):
        """LOW: an ESC byte in registry text must not survive into the card."""
        from soup_cli.commands.push import _safe_md_cell

        assert "\x1b" not in _safe_md_cell("a\x1b[31mred\x1b[0m")
        assert "\x00" not in _safe_md_cell("a\x00b")

    def test_safe_md_cell_neutralises_backtick(self):
        from soup_cli.commands.push import _safe_md_cell

        assert "`" not in _safe_md_cell("br`eak")

    def test_card_notes_are_truncated(self):
        """LOW/INFO: unbounded registry notes must not render an unbounded card."""
        from soup_cli.commands.card import _MAX_NOTES_CHARS, build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": "A" * (_MAX_NOTES_CHARS + 5_000),
            "config_hash": "", "data_hash": "", "run_id": None, "tags": [],
            "config_json": "{}",
        }
        md = build_model_card(entry, [], [], [])
        assert "[truncated]" in md

    def test_card_rows_are_capped(self):
        from soup_cli.commands.card import _MAX_ROWS, build_model_card

        arts = [
            {"kind": "gguf", "path": f"m{i}.gguf", "sha256": "x"}
            for i in range(_MAX_ROWS + 50)
        ]
        entry = {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": None, "tags": [], "config_json": "{}",
        }
        md = build_model_card(entry, arts, [], [])
        assert f"m{_MAX_ROWS + 10}.gguf" not in md  # beyond the cap
        assert "m0.gguf" in md

    def test_ci_workflow_uses_shared_containment_helper(self):
        """MEDIUM: the hand-rolled S_ISLNK guard missed Windows junctions."""
        import inspect

        from soup_cli.utils import ci_workflow

        src = inspect.getsource(ci_workflow)
        assert "enforce_under_cwd_and_no_symlink" in src
        # no second, weaker copy of the guard (a bare S_ISLNK call misses
        # Windows junctions); a mention in a comment is fine.
        assert "stat.S_ISLNK" not in src

    def test_ci_workflow_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.ci_workflow import write_soup_gate_workflow

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError):
            write_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
                output_path="../escape.yml",
            )


# --------------------------------------------------------------------------- #
# tdd-review gap closures (H1 / M1-M4 / L1-L7)
# --------------------------------------------------------------------------- #
def _model_dir(tmp_path):
    """A minimal full-model dir that `soup push` accepts."""
    d = tmp_path / "out"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    return d


class TestPushCardIntegration:
    """H1 — end-to-end `push --card` wiring (was only --help-tested)."""

    def test_push_card_uploads_registry_card(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        db = tmp_path / "reg.db"
        eid = _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        _model_dir(tmp_path)
        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)

        result = runner.invoke(
            app, ["push", "--model", "out", "--repo", "user/m", "--card", eid]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        readme_calls = [
            c for c in fake_api.upload_file.call_args_list
            if c.kwargs.get("path_in_repo") == "README.md"
        ]
        assert readme_calls, fake_api.upload_file.call_args_list
        body = readme_calls[0].kwargs["path_or_fileobj"].decode("utf-8")
        assert "phi-model" in body  # registry-driven, not the path-based card
        assert "Qwen/Qwen2.5-7B-Instruct" in body

    def test_push_card_error_exits_1_before_network(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        db = tmp_path / "reg.db"
        _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        _model_dir(tmp_path)
        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)

        result = runner.invoke(
            app, ["push", "--model", "out", "--repo", "user/m", "--card", "reg_nope"]
        )
        assert result.exit_code == 1
        assert "--card" in result.output
        # fails fast: no repo creation / upload attempted
        assert not fake_api.create_repo.called
        assert not fake_api.upload_folder.called

    def test_push_card_non_hf_hub_warns_and_ignores(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        db = tmp_path / "reg.db"
        eid = _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        _model_dir(tmp_path)
        uploaded = {}

        def _fake_upload_repo(hub, repo, **kwargs):
            uploaded["hub"] = hub

        monkeypatch.setattr("soup_cli.utils.hubs.upload_repo", _fake_upload_repo)
        result = runner.invoke(
            app,
            ["push", "--model", "out", "--repo", "user/m", "--hub", "modelscope",
             "--card", eid],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "HF-only" in result.output
        assert uploaded.get("hub") == "modelscope"


class TestCardErrorPaths:
    def test_ambiguous_ref_raises_card_error(self, tmp_path, monkeypatch):
        """M1 — every entry id shares the reg_ prefix, so a bare prefix is
        ambiguous once 2 entries exist."""
        from soup_cli.commands.card import CardError, build_card_for_ref

        db = tmp_path / "reg.db"
        monkeypatch.chdir(tmp_path)
        _make_entry(db)
        _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        with pytest.raises(CardError):
            build_card_for_ref("reg_")

    def test_ambiguous_ref_cli_exits_1(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        monkeypatch.chdir(tmp_path)
        _make_entry(db)
        _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        result = runner.invoke(app, ["card", "reg_", "-o", "CARD.md"])
        assert result.exit_code == 1

    def test_card_write_oserror_is_friendly(self, tmp_path, monkeypatch):
        """M2 — `-o .` is a directory: passes containment, fails on os.replace."""
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        eid = _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["card", eid, "-o", "."])
        assert result.exit_code == 1
        assert "cannot write card" in result.output.lower()

    def test_card_overwrites_existing_output(self, tmp_path, monkeypatch):
        """L7 — pin the intended behaviour: regeneration overwrites, no --force."""
        from soup_cli.cli import app

        db = tmp_path / "reg.db"
        eid = _make_entry(db)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CARD.md").write_text("STALE", encoding="utf-8")
        result = runner.invoke(app, ["card", eid, "-o", "CARD.md"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "STALE" not in (tmp_path / "CARD.md").read_text(encoding="utf-8")


class TestBuildModelCardEdges:
    def test_empty_entry_uses_defaults(self):
        """M3 — name -> 'model', no base_model frontmatter line, no crash."""
        from soup_cli.commands.card import build_model_card

        md = build_model_card({}, [], [], [])
        assert "# model" in md
        front = md.split("---", 2)[1]
        assert "base_model:" not in front
        assert "Full model" in md  # no adapter artifact -> full model

    def test_scorecard_drops_none_fields(self):
        """M4 — rows with a None benchmark/score must be dropped, not rendered."""
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": "run1", "tags": [], "config_json": "{}",
        }
        evals = [
            {"benchmark": None, "score": 0.5},
            {"benchmark": "gsm8k", "score": None},
            {"benchmark": "mmlu", "score": 0.42},
        ]
        md = build_model_card(entry, [], evals, [])
        assert "mmlu" in md
        assert "gsm8k" not in md

    def test_scorecard_duplicate_benchmark_last_wins(self):
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": "run1", "tags": [], "config_json": "{}",
        }
        evals = [{"benchmark": "mmlu", "score": 0.1}, {"benchmark": "mmlu", "score": 0.9}]
        md = build_model_card(entry, [], evals, [])
        assert "0.900" in md
        assert "0.100" not in md

    def test_malformed_config_json_does_not_crash(self):
        from soup_cli.commands.card import build_model_card

        entry = {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": None, "tags": [], "config_json": "not json{{",
        }
        md = build_model_card(entry, [], [], [])
        assert "# m" in md

    def test_yaml_dq_unit(self):
        """L5 — direct unit coverage of the frontmatter escaper."""
        from soup_cli.commands.card import _yaml_dq

        assert _yaml_dq("plain") == '"plain"'
        assert _yaml_dq('a"b') == '"a\\"b"'
        assert _yaml_dq("a\\b") == '"a\\\\b"'
        assert "\n" not in _yaml_dq("a\nb")
        assert "\x1b" not in _yaml_dq("a\x1bb")

    def test_truncate_boundary(self):
        """L6 — exactly at the limit must NOT truncate."""
        from soup_cli.commands.card import _truncate

        assert _truncate("abcde", 5) == "abcde"
        assert "[truncated]" in _truncate("abcdef", 5)


class TestCiWorkflowEdges:
    def test_write_rejects_non_bool_overwrite(self):
        from soup_cli.utils.ci_workflow import write_soup_gate_workflow

        with pytest.raises(TypeError):
            write_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
                overwrite="yes",
            )

    @pytest.mark.parametrize("bad_out", ["", None, True])
    def test_write_rejects_bad_output_path(self, bad_out):
        from soup_cli.utils.ci_workflow import write_soup_gate_workflow

        with pytest.raises((ValueError, TypeError)):
            write_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
                output_path=bad_out,
            )

    def test_size_cap_fires(self, tmp_path, monkeypatch):
        """L4 — the 64 KiB cap is unreachable via normal input (path len is
        capped at 4096); pin that the guard still fires when tripped."""
        from soup_cli.utils import ci_workflow

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ci_workflow, "_MAX_FILE_BYTES", 10)
        with pytest.raises(ValueError, match="cap"):
            ci_workflow.write_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
            )

    @pytest.mark.skipif(os.name != "posix", reason="POSIX symlink semantics")
    def test_write_rejects_symlink_output(self, tmp_path, monkeypatch):
        """L1 — live symlink coverage at ci_workflow's own call site."""
        from soup_cli.utils.ci_workflow import write_soup_gate_workflow

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.yml"
        target.write_text("x", encoding="utf-8")
        link = tmp_path / "link.yml"
        link.symlink_to(target)
        with pytest.raises(ValueError):
            write_soup_gate_workflow(
                data_path="d.jsonl", suite_path="s.yaml", evidence_path="e.json",
                output_path="link.yml", overwrite=True,
            )

    def test_ci_init_bad_python_names_the_problem(self, tmp_path, monkeypatch):
        """L3 — assert the message, not just the exit code."""
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["ci", "init", "--python", "3.x"])
        assert result.exit_code == 1
        assert "python_version" in result.output or "3.11" in result.output


class TestAdapterInference:
    """Step-6 smoke found this: a REAL LoRA run with no artifacts attached
    rendered as `Type | Full model` + `library_name: transformers` — a false
    claim in a provenance document and a broken HF Hub card."""

    def _entry(self, config):
        return {
            "id": "r", "name": "m", "base_model": "b", "task": "sft",
            "created_at": "t", "notes": None, "config_hash": "", "data_hash": "",
            "run_id": None, "tags": [], "config_json": json.dumps(config),
        }

    def test_lora_config_without_artifacts_is_adapter(self):
        from soup_cli.commands.card import build_model_card

        md = build_model_card(
            self._entry({"training": {"lora": {"r": 4, "alpha": 8}}}), [], [], []
        )
        assert "LoRA adapter" in md
        assert "library_name: peft" in md
        assert "Full model" not in md

    def test_no_lora_is_full_model(self):
        from soup_cli.commands.card import build_model_card

        md = build_model_card(self._entry({"training": {"epochs": 1}}), [], [], [])
        assert "Full model" in md
        assert "library_name: transformers" in md

    def test_zero_rank_lora_is_full_model(self):
        from soup_cli.commands.card import build_model_card

        md = build_model_card(self._entry({"training": {"lora": {"r": 0}}}), [], [], [])
        assert "Full model" in md

    def test_adapter_artifact_is_definitive(self):
        from soup_cli.commands.card import build_model_card

        md = build_model_card(
            self._entry({"training": {}}),
            [{"kind": "adapter", "path": "a", "sha256": "x"}], [], [],
        )
        assert "LoRA adapter" in md

    def test_dense_export_artifact_beats_lora_config(self):
        """A merged/gguf export is a standalone model even if trained via LoRA."""
        from soup_cli.commands.card import build_model_card

        md = build_model_card(
            self._entry({"training": {"lora": {"r": 8}}}),
            [{"kind": "merged", "path": "m", "sha256": "x"}], [], [],
        )
        assert "Full model" in md

    def test_spectrum_full_ft_is_not_adapter(self):
        """unfrozen_parameters = full FT; the dumped config still carries a
        default lora block, which must NOT be read as 'adapter'."""
        from soup_cli.commands.card import build_model_card

        md = build_model_card(
            self._entry({"training": {"unfrozen_parameters": ["q_proj"],
                                      "lora": {"r": 8}}}), [], [], [],
        )
        assert "Full model" in md

    def test_lisa_full_ft_is_not_adapter(self):
        from soup_cli.commands.card import build_model_card

        md = build_model_card(
            self._entry({"training": {"lisa_enabled": True, "lora": {"r": 8}}}), [], [], [],
        )
        assert "Full model" in md

    def test_malformed_lora_block_does_not_crash(self):
        from soup_cli.commands.card import build_model_card

        md = build_model_card(self._entry({"training": {"lora": "nonsense"}}), [], [], [])
        assert "Full model" in md
        md2 = build_model_card(self._entry({"training": {"lora": {"r": "abc"}}}), [], [], [])
        assert "Full model" in md2


# --------------------------------------------------------------------------- #
# GGUF-on-Windows validation fixes (#70/#144) — both found by the live build
# --------------------------------------------------------------------------- #
class TestGgufWindowsFixes:
    def test_finds_msvc_multi_config_binary(self, tmp_path):
        """MSVC/Xcode are multi-config generators: the binary lands in
        build/bin/Release/, not the flat build/bin/ a Make/Ninja build makes.
        Without this, `soup export --format gguf` cannot find a correctly-built
        llama.cpp on Windows."""
        from soup_cli.commands.export import _find_quantize_binary

        llama = tmp_path / "llama.cpp"
        rel = llama / "build" / "bin" / "Release"
        rel.mkdir(parents=True)
        binary = rel / "llama-quantize.exe"
        binary.write_text("stub", encoding="utf-8")
        found = _find_quantize_binary(llama)
        assert found is not None
        assert found.name == "llama-quantize.exe"
        assert "Release" in str(found)

    def test_flat_single_config_layout_still_found(self, tmp_path):
        """Make/Ninja layout must keep working."""
        from soup_cli.commands.export import _find_quantize_binary

        llama = tmp_path / "llama.cpp"
        bindir = llama / "build" / "bin"
        bindir.mkdir(parents=True)
        (bindir / "llama-quantize").write_text("stub", encoding="utf-8")
        assert _find_quantize_binary(llama) is not None

    def test_missing_binary_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.commands import export as export_mod

        monkeypatch.setattr(export_mod.shutil, "which", lambda _: None)
        llama = tmp_path / "llama.cpp"
        (llama / "build").mkdir(parents=True)
        assert export_mod._find_quantize_binary(llama) is None

    def test_convert_deps_never_install_llama_requirements(self):
        """llama.cpp's requirements.txt pins torch~=2.2.1 against the CPU wheel
        index; installing it downgrades a user's CUDA torch to CPU-only and
        breaks training (observed live: 2.5.1+cu -> 2.2.2+cpu). Soup must only
        install the convert script's EXTRA deps, unpinned."""
        import inspect

        from soup_cli.commands import export as export_mod

        src = inspect.getsource(export_mod._find_llama_cpp)
        assert "requirements.txt" not in src
        assert "torch" not in export_mod._CONVERT_EXTRA_DEPS
        assert "transformers" not in export_mod._CONVERT_EXTRA_DEPS
        assert "gguf" in export_mod._CONVERT_EXTRA_DEPS

    def test_install_convert_deps_is_unpinned_and_targeted(self, monkeypatch):
        from soup_cli.commands import export as export_mod

        calls = {}

        def _fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)
        export_mod._install_convert_deps()
        cmd = calls["cmd"]
        assert "-r" not in cmd  # never a requirements file
        assert all("==" not in tok and "~=" not in tok for tok in cmd)  # unpinned
        assert "gguf" in cmd

    def test_install_convert_deps_failure_is_non_fatal(self, monkeypatch):
        """A pip failure must warn, not abort the export."""
        import subprocess as sp

        from soup_cli.commands import export as export_mod

        def _boom(cmd, **kwargs):
            raise sp.CalledProcessError(1, cmd, stderr="network down")

        monkeypatch.setattr(export_mod.subprocess, "run", _boom)
        export_mod._install_convert_deps()  # must not raise


class TestOllamaModelfileAbsolutePath:
    """`ollama create` resolves a relative FROM against the Modelfile's own
    directory, and Soup writes the Modelfile to a temp dir — so a relative GGUF
    path made Ollama try to PULL it as a remote model:
    "pull model manifest: file does not exist". Found live in the v0.71.35
    GGUF-on-Windows validation."""

    def test_modelfile_from_is_absolute(self, tmp_path, monkeypatch):
        from soup_cli.utils.ollama import create_modelfile

        monkeypatch.chdir(tmp_path)
        gguf = tmp_path / "m.q8_0.gguf"
        gguf.write_text("stub", encoding="utf-8")
        body = create_modelfile("m.q8_0.gguf")  # relative input
        from_line = body.splitlines()[0]
        assert from_line.startswith("FROM ")
        emitted = from_line[len("FROM "):]
        assert os.path.isabs(emitted), emitted
        assert emitted.endswith("m.q8_0.gguf")

    def test_absolute_input_stays_absolute(self, tmp_path):
        from soup_cli.utils.ollama import create_modelfile

        gguf = tmp_path / "m.gguf"
        gguf.write_text("stub", encoding="utf-8")
        body = create_modelfile(str(gguf))
        assert os.path.isabs(body.splitlines()[0][len("FROM "):])

    def test_other_directives_still_render(self, tmp_path, monkeypatch):
        from soup_cli.utils.ollama import create_modelfile

        monkeypatch.chdir(tmp_path)
        body = create_modelfile("m.gguf", template="chatml", system_prompt="be nice")
        assert "TEMPLATE" in body
        assert 'SYSTEM "be nice"' in body


class TestLlamaCppHomeAnchor:
    """GGUF bug #1: SOUP_DIR is the bare name ".soup", so using it relatively
    made the lookup cwd-dependent — ~/.soup/llama.cpp was never found and the
    auto-clone dropped a fresh ~200 MB checkout into whatever directory the
    user ran from. Must anchor to home like tracker.py / registry/store.py."""

    def test_finds_llama_cpp_under_home_not_cwd(self, tmp_path, monkeypatch):
        from pathlib import Path

        from soup_cli.commands import export as export_mod

        fake_home = tmp_path / "home"
        llama = fake_home / ".soup" / "llama.cpp"
        llama.mkdir(parents=True)
        (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")

        workdir = tmp_path / "someproject"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        monkeypatch.delenv("LLAMA_CPP_PATH", raising=False)

        # must resolve to the HOME copy without cloning
        def _no_clone(*a, **k):
            raise AssertionError("must not clone: the home copy already exists")

        monkeypatch.setattr(export_mod.subprocess, "run", _no_clone)

        found = export_mod._find_llama_cpp(None)
        assert Path(found).resolve() == llama.resolve()
        # and no stray .soup litter in the working directory
        assert not (workdir / ".soup").exists()
