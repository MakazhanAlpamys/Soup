"""#1174: the head-prefetch A/B driver's JSON must say which card, driver and tree it measured.

Two things I broke, both silent because nothing exercised the driver's JSON before:
the payload's own ``"driver"`` key (the script path) was overwriting
``gpu_facts()``'s ``"driver"`` key (the NVIDIA version) when I spread the dict at
the top level, and ``_source_sha()`` resolved the commit from this file's own
location, not from the ``soup_cli`` tree it actually measured.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _REPO_ROOT / "benchmarks" / "harness"
_HARNESS = _HARNESS_DIR / "head_prefetch_ab.py"


def _load_driver():
    sys.path.insert(0, str(_HARNESS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("head_prefetch_ab", _HARNESS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_HARNESS_DIR))


_driver = _load_driver()


@pytest.mark.parametrize("status", ["", " M src/soup_cli/utils/layer_stream_runtime.py\n"])
def test_git_sha_names_the_imported_tree_and_marks_it_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    """The driver and the ``soup_cli`` it measures can come from different checkouts,
    so I resolve the commit from ``soup_cli.__file__``, not from this driver's file."""
    import soup_cli

    package_file = soup_cli.__file__
    assert package_file is not None
    package_dir = Path(package_file).resolve().parent
    expected_sha = "a" * 40
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        calls.append((command, cwd))
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=f"{tmp_path}\n")
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=f"{expected_sha}\n")
        return SimpleNamespace(stdout=status)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_driver.subprocess, "run", fake_run)

    assert _driver._source_sha() == expected_sha + ("-dirty" if status else "")
    assert calls == [
        (["git", "rev-parse", "--show-toplevel"], package_dir),
        (["git", "rev-parse", "HEAD"], tmp_path),
        (["git", "status", "--porcelain", "--untracked-files=normal"], tmp_path),
    ]


def test_source_sha_is_unknown_without_a_git_toplevel(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        return SimpleNamespace(stdout="\n")

    monkeypatch.setattr(_driver.subprocess, "run", fake_run)
    assert _driver._source_sha() == "unknown"


def test_source_sha_is_unknown_when_git_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        raise FileNotFoundError("no git")

    monkeypatch.setattr(_driver.subprocess, "run", fake_run)
    assert _driver._source_sha() == "unknown"


def test_the_gpu_facts_stay_nested_so_the_nvidia_driver_survives() -> None:
    """``gpu_facts`` returns its own ``"driver"`` key (the NVIDIA version); spreading
    it into a payload that also sets ``"driver"`` to the script path overwrites it."""
    text = _HARNESS.read_text(encoding="utf-8")
    assert '"gpu": stream_probe.gpu_facts(device)' in text
    assert "**stream_probe.gpu_facts(" not in text
