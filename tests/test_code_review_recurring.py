"""Regression tests for the recurring-pattern classes in CODE_REVIEW.md.

Windows-8.3-short-name breakage of ``Path.resolve()+relative_to()`` cannot be
reproduced portably, so the containment-migration tests assert the code now
routes through the ``os.path.realpath+commonpath`` helper (``utils.paths``) plus
a behavioural check that containment still works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import soup_cli


def _src(rel: str) -> str:
    return (Path(soup_cli.__file__).parent / rel).read_text(encoding="utf-8")


# ── Path.resolve()+relative_to() containment holdouts → is_under / is_under_cwd ──


def test_containment_holdouts_migrated_to_commonpath_helper():
    for rel, needle in {
        "utils/ollama.py": "is_under_cwd",
        "migrate/common.py": "is_under_cwd",
        "commands/serve.py": "is_under(",
        "commands/export.py": "is_under_cwd",
        "commands/generate.py": "is_under(",
        "data/loader.py": "is_under(",
        "eval/checkpoint_intelligence.py": "is_under(",
    }.items():
        src = _src(rel)
        assert needle in src, f"{rel} not migrated to {needle}"
        # The buggy cwd-containment idiom must be gone from these sites.
        assert ".relative_to(cwd)" not in src, f"{rel} still uses relative_to(cwd)"

    # data.py migrated all three sample/download output-path checks.
    assert _src("commands/data.py").count("is_under_cwd(") >= 3


def test_migrate_output_path_containment_still_works(tmp_path, monkeypatch):
    from soup_cli.migrate.common import validate_output_path

    monkeypatch.chdir(tmp_path)
    ok = validate_output_path(Path("out.yaml"))
    assert ok.name == "out.yaml"
    with pytest.raises(ValueError):
        validate_output_path(Path("../escape.yaml"))


def test_generate_path_within_cwd_behaviour(tmp_path, monkeypatch):
    from soup_cli.commands.generate import _path_within_cwd

    proj = tmp_path / "project"
    proj.mkdir()
    (tmp_path / "project-secrets").mkdir()
    monkeypatch.chdir(proj)
    assert _path_within_cwd((proj / "d.jsonl").resolve(), Path.cwd()) is True
    assert (
        _path_within_cwd((tmp_path / "project-secrets" / "d.jsonl").resolve(), Path.cwd())
        is False
    )


# ── Unescaped Rich markup from external data ──


def test_eval_v0550_escapes_dataset_error():
    assert "escape(str(exc))" in _src("commands/_eval_v0550.py")
    assert 'f"[red]Cannot read dataset:[/] {exc}"' not in _src("commands/_eval_v0550.py")


# ── Symlink-following writes → atomic-write / mkstemp ──


def test_active_sampler_uses_atomic_write():
    src = _src("utils/active_sampler.py")
    assert "atomic_write_text" in src
    assert 'open(output_path, "w"' not in src


def test_ui_train_config_uses_secure_tempfile():
    src = _src("ui/app.py")
    assert "tempfile.mkstemp" in src
    assert 'os.path.join(\n                tempfile.gettempdir(), "soup_ui_config.yaml"' not in src


def test_active_sampler_writes_selected_rows(tmp_path, monkeypatch):
    """Behavioural check: the atomic write still produces the JSONL output."""
    import json

    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}], "logprob": -float(i)}
        for i in range(5)
    ]
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"
    sample_uncertain_rows(str(inp), output_path=str(out), budget=3)
    assert out.exists()
    written = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(written) == 3
