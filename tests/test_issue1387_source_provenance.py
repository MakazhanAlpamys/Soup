"""#1387: one ``source_sha()`` for every benchmark harness that records its tree.

These run real ``git`` against a throwaway checkout holding a stand-in ``soup_cli``
package, so the resolution is exercised end to end rather than through a fake
``subprocess.run``.
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

_HARNESS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


def _load(name: str):
    sys.path.insert(0, str(_HARNESS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            f"_issue1387_{name}", _HARNESS_DIR / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses look their module up here
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_HARNESS_DIR))


provenance = _load("source_provenance")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = tmp_path / "checkout"
    (root / "src" / "soup_cli").mkdir(parents=True)
    (root / "src" / "soup_cli" / "__init__.py").write_text("", encoding="utf-8")
    (root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", "init",
    )
    return root


def _import_soup_cli_from(monkeypatch: pytest.MonkeyPatch, init_file: Path) -> None:
    fake = types.ModuleType("soup_cli")
    fake.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "soup_cli", fake)


def test_clean_tree_records_its_commit(monkeypatch, checkout):
    _import_soup_cli_from(monkeypatch, checkout / "src" / "soup_cli" / "__init__.py")
    assert provenance.source_sha() == _git(checkout, "rev-parse", "HEAD").strip()


def test_tracked_edit_is_dirty(monkeypatch, checkout):
    init_file = checkout / "src" / "soup_cli" / "__init__.py"
    init_file.write_text("x = 1\n", encoding="utf-8")
    _import_soup_cli_from(monkeypatch, init_file)
    assert provenance.source_sha() == _git(checkout, "rev-parse", "HEAD").strip() + "-dirty"


def test_dirty_pathspec_ignores_changes_outside_it(monkeypatch, checkout):
    (checkout / "result.json").write_text("{}", encoding="utf-8")
    _import_soup_cli_from(monkeypatch, checkout / "src" / "soup_cli" / "__init__.py")
    head = _git(checkout, "rev-parse", "HEAD").strip()
    assert provenance.source_sha() == head + "-dirty"
    assert provenance.source_sha(dirty_pathspec="src") == head


def test_untracked_copy_inside_the_checkout_is_unknown(monkeypatch, checkout):
    """A wheel installed into a gitignored .venv/ in the checkout resolves to the
    checkout too, but the code measured is the wheel's, not the commit's."""
    installed = checkout / ".venv" / "lib" / "site-packages" / "soup_cli" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    _import_soup_cli_from(monkeypatch, installed)
    assert provenance.source_sha() == "unknown"


def test_git_archive_tree_is_unknown(monkeypatch, checkout, tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    archive = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD"], cwd=checkout, check=True, capture_output=True
    ).stdout
    zipfile.ZipFile(io.BytesIO(archive)).extractall(extracted)
    _import_soup_cli_from(monkeypatch, extracted / "src" / "soup_cli" / "__init__.py")
    assert provenance.source_sha() == "unknown"


@pytest.mark.parametrize(
    "harness, expected",
    [
        ("variant2_gate", {}),
        ("issue361_nf4_throughput", {}),
        ("head_prefetch_ab", {"dirty_pathspec": "src"}),
    ],
)
def test_every_harness_uses_the_shared_helper(monkeypatch, harness, expected):
    module = _load(harness)
    monkeypatch.setattr(module, "source_sha", lambda **kwargs: kwargs)
    assert module._source_sha() == expected
    assert "def source_sha" not in (_HARNESS_DIR / f"{harness}.py").read_text(encoding="utf-8")
