"""A local directory under an allowlisted org-shaped path is not known-safe (#3008).

``is_known_safe`` matched only the start of the name against
``KNOWN_SAFE_PREFIXES``, so a local checkpoint at ``Qwen/my-merge`` counted as a
trusted Hub repo and ``--trust-remote-code`` ran its custom code without the
warning panel. A name that exists on disk is now a local path, as
``from_pretrained`` reads it, and the allowlist answers for Hub repo ids only.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.utils.trust_remote import (
    KNOWN_SAFE_PREFIXES,
    is_known_safe,
    model_requires_trust_remote_code,
    resolve_trust_remote_code,
)
from tests.conftest import strip_ansi

PANEL_TITLE = "REMOTE CODE WARNING"


def _name(prefix: str) -> str:
    return f"{prefix}local-merge"


def _make_local_checkpoint(name: str) -> None:
    path = Path(name)
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        '{"auto_map": {"AutoModelForCausalLM": "modeling_x.Model"}}', encoding="utf-8"
    )


def _resolve(name: str, requested: bool) -> tuple[bool, str]:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    out = resolve_trust_remote_code(
        name, requested=requested, console=console, requires_remote_code=False
    )
    return out, " ".join(strip_ansi(buf.getvalue()).split())


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("prefix", KNOWN_SAFE_PREFIXES)
def test_local_directory_under_prefix_is_not_known_safe(workdir, prefix):
    name = _name(prefix)
    _make_local_checkpoint(name)

    assert not is_known_safe(name)


@pytest.mark.parametrize("prefix", KNOWN_SAFE_PREFIXES)
def test_hub_id_without_local_directory_is_still_known_safe(workdir, prefix):
    """Control: the same spelling with nothing on disk is a Hub repo id."""
    assert is_known_safe(_name(prefix))


@pytest.mark.parametrize("prefix", KNOWN_SAFE_PREFIXES)
def test_local_directory_gets_the_warning_panel(workdir, prefix):
    name = _name(prefix)
    _make_local_checkpoint(name)

    out, text = _resolve(name, requested=True)

    assert out is True
    assert PANEL_TITLE in text
    assert name in text


@pytest.mark.parametrize("prefix", KNOWN_SAFE_PREFIXES)
def test_hub_id_without_local_directory_has_no_panel(workdir, prefix):
    """Control: an allowlisted Hub id keeps its quiet opt-in."""
    out, text = _resolve(_name(prefix), requested=True)

    assert out is True
    assert text == ""


@pytest.mark.parametrize("prefix", KNOWN_SAFE_PREFIXES)
def test_not_requested_stays_silent_for_local_directory(workdir, prefix):
    """Control: without the opt-in nothing is printed and nothing is trusted."""
    name = _name(prefix)
    _make_local_checkpoint(name)

    out, text = _resolve(name, requested=False)

    assert out is False
    assert text == ""


# --- other spellings of a local path (one or two prefixes each) ------------

SPELLING_PREFIXES = ("Qwen/", "meta-llama/")


@pytest.mark.parametrize("prefix", SPELLING_PREFIXES)
def test_trailing_slash_spelling_is_not_known_safe(workdir, prefix):
    name = f"{prefix}x"
    _make_local_checkpoint(name)

    assert not is_known_safe(f"{name}/")


@pytest.mark.parametrize("prefix", SPELLING_PREFIXES)
def test_regular_file_is_not_known_safe(workdir, prefix):
    """``os.path.exists``, not ``isdir``: a file on disk is a local path too."""
    name = f"{prefix}x"
    Path(prefix).mkdir()
    Path(name).write_text("not a checkpoint", encoding="utf-8")

    assert not is_known_safe(name)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("prefix", SPELLING_PREFIXES)
def test_symlink_to_directory_is_not_known_safe(workdir, prefix):
    target = workdir / "real-checkpoint"
    target.mkdir()
    name = f"{prefix}x"
    Path(prefix).mkdir()
    Path(name).symlink_to(target, target_is_directory=True)

    assert not is_known_safe(name)


def test_dot_slash_spelling_is_not_known_safe(workdir):
    """Control: never matched a prefix, before or after."""
    _make_local_checkpoint("Qwen/x")

    assert not is_known_safe("./Qwen/x")


def test_absolute_path_is_not_known_safe(workdir):
    """Control: never matched a prefix, before or after."""
    _make_local_checkpoint("Qwen/x")

    assert not is_known_safe(str((workdir / "Qwen" / "x").resolve()))


# --- the flag stays the gate -------------------------------------------------


def test_local_custom_code_without_the_flag_is_refused(workdir):
    """Control: the change touches only the panel; the opt-in still decides."""
    name = "Qwen/x"
    _make_local_checkpoint(name)
    requires = model_requires_trust_remote_code(name)
    assert requires is True

    with pytest.raises(ValueError, match="--trust-remote-code"):
        resolve_trust_remote_code(
            name, requested=False, console=None, requires_remote_code=requires
        )
