"""Probe commands refuse unsafe, unreadable, and non-UTF-8 evidence locally."""

from pathlib import Path

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()
BASE = "meta-llama/Llama-3-8B"


def _plain(output: str) -> str:
    return " ".join(strip_ansi(output).split())


@pytest.fixture
def workdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.mark.parametrize(
    ("args", "field"),
    [
        pytest.param(("sleeper", BASE, "--evidence", "{path}"), "--evidence", id="sleeper"),
        pytest.param(("truth", BASE, "--evidence", "{path}"), "--evidence", id="truth"),
        pytest.param(("harm", BASE, "--evidence", "{path}"), "--evidence", id="harm"),
        pytest.param(("interference", "{path}"), "--losses", id="interference"),
        pytest.param(
            ("sae-diff", "sae.safetensors", "{path}", "valid.json"), "pre_acts", id="sae-diff-pre"
        ),
        pytest.param(
            ("sae-diff", "sae.safetensors", "valid.json", "{path}"), "post_acts", id="sae-diff-post"
        ),
    ],
)
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("outside", "must stay under cwd"),
        pytest.param("symlink", "must not be a symlink", marks=pytest.mark.requires_symlink),
        ("directory", "cannot read"),
        ("non_utf8", "UTF-8"),
    ],
    ids=["outside", "symlink", "directory", "non-utf8"],
)
def test_probe_refuses_evidence_with_literal_path(
    workdir: Path, args: tuple[str, ...], field: str, failure: str, reason: str
) -> None:
    path = Path("evidence") / "input[red].json"
    if failure == "outside":
        path = Path("..") / path.name
        (workdir / path).write_text("{}", encoding="utf-8")
    else:
        path.parent.mkdir()
        if failure == "symlink":
            target = path.with_name("real.json")
            target.write_text("{}", encoding="utf-8")
            path.symlink_to(target.name)
        elif failure == "directory":
            path.mkdir()
        else:
            path.write_bytes(b"\xff")

    if args[0] == "sae-diff":
        import numpy as np
        from safetensors.numpy import save_file

        save_file({"W_enc": np.ones((1, 1), dtype=np.float32)}, "sae.safetensors")
        Path("valid.json").write_text('{"activations": [[0.0]]}', encoding="utf-8")

    result = runner.invoke(app, ["probe", *(arg.format(path=path) for arg in args)])
    output = _plain(result.output)
    assert result.exit_code == 2, (output, repr(result.exception))
    assert output.startswith((f"{field}:", f"{field} ")), output
    assert f"'{path}'" in output
    assert reason in output


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        pytest.param(None, "cannot read — FileNotFoundError", id="missing"),
        pytest.param(16_777_217, "file too large (16777217 > 16777216 bytes)", id="oversize"),
        pytest.param("?", "invalid JSON — Expecting value: line 1 column 1 (char 0)", id="json"),
        pytest.param("[]", "must be a JSON object (dict)", id="not-object"),
    ],
)
def test_probe_preserves_existing_evidence_refusals(
    workdir: Path, contents: str | int | None, expected: str
) -> None:
    path = workdir / "input.json"
    if isinstance(contents, int):
        with path.open("wb") as handle:
            handle.truncate(contents)
    elif contents is not None:
        path.write_text(contents, encoding="utf-8")

    result = runner.invoke(app, ["probe", "truth", BASE, "--evidence", path.name])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"--evidence: {expected}" in _plain(result.output)
