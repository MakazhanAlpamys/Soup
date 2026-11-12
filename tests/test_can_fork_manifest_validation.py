"""#2728: ``soup can fork`` refuses a manifest every reader would refuse.

``fork_can`` dumped the new manifest as a raw dict, so an ``--author`` with a
control character or over 128 characters, or a source name that ``-fork`` pushes
past 128 characters, produced a ``.can`` that the fork reported as written and
``soup can verify`` / ``inspect`` / ``run`` then rejected. ``pack`` already
refused the same values, because it builds a ``Manifest``.
"""

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore

runner = CliRunner()


def _pack(tmp_path, monkeypatch, name="recipe"):
    from soup_cli.cans.pack import pack_entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    with RegistryStore() as store:
        eid = store.push(name=name, tag="v1", base_model="org/base", task="sft",
                         run_id=None, config={"base": "org/base"})
    out = tmp_path / "src.can"
    pack_entry(entry_id=eid, out_path=str(out))
    return out


@pytest.mark.parametrize(
    "author, expected",
    [
        ("alice\tbob", "--author must not contain control characters"),
        ("alice\nbob", "--author must not contain control characters"),
        ("a" * 200, "--author: 200 characters, over the 128-character limit"),
    ],
)
def test_a_bad_author_is_refused_and_nothing_is_written(
    tmp_path, monkeypatch, author, expected,
):
    from soup_cli.cans.pack import fork_can

    src = _pack(tmp_path, monkeypatch)
    out = tmp_path / "fork.can"
    with pytest.raises(ValueError) as excinfo:
        fork_can(source=str(src), out_path=str(out), modifications=[], author=author)
    assert expected in str(excinfo.value)
    assert not out.exists()


def test_a_name_that_fork_pushes_past_the_limit_is_refused(tmp_path, monkeypatch):
    from soup_cli.cans.pack import fork_can

    src = _pack(tmp_path, monkeypatch, name="m" * 128)
    out = tmp_path / "fork.can"
    with pytest.raises(ValueError) as excinfo:
        fork_can(source=str(src), out_path=str(out), modifications=[], author="alice")
    message = str(excinfo.value)
    assert "name: the forked name is the source name plus '-fork', 133 characters" in message
    assert "over the 128-character limit" in message
    assert "at most 123 characters" in message
    assert not out.exists()


def test_the_longest_forkable_name_forks_and_verifies(tmp_path, monkeypatch):
    from soup_cli.cans.pack import fork_can
    from soup_cli.cans.unpack import inspect_can
    from soup_cli.cans.verify import verify_can

    src = _pack(tmp_path, monkeypatch, name="m" * 123)
    out = tmp_path / "fork.can"
    fork_can(source=str(src), out_path=str(out), modifications=[], author="alice")
    report = verify_can(str(out))
    assert report.manifest_ok and report.config_ok, report
    assert inspect_can(str(out)).name == "m" * 123 + "-fork"


def test_the_cli_exits_1_on_a_bad_author(tmp_path, monkeypatch):
    _pack(tmp_path, monkeypatch)
    result = runner.invoke(app, ["can", "fork", "src.can", "--out", "fork.can",
                                 "--author", "alice\tbob"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "--author" in " ".join(result.output.split())
    assert not (tmp_path / "fork.can").exists()


def test_a_valid_fork_passes_verify(tmp_path, monkeypatch):
    from soup_cli.cans.pack import fork_can
    from soup_cli.cans.unpack import inspect_can
    from soup_cli.cans.verify import verify_can

    src = _pack(tmp_path, monkeypatch)
    out = tmp_path / "fork.can"
    fork_can(source=str(src), out_path=str(out), modifications=[], author="alice")
    report = verify_can(str(out))
    assert report.manifest_ok and report.config_ok, report
    manifest = inspect_can(str(out))
    assert manifest.name == "recipe-fork"
    assert manifest.author == "alice"
