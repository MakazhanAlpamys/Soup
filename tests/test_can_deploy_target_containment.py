"""``_deploy_target(kind="gguf")`` must keep the manifest path inside the can.

The sibling ``kind="ollama"`` branch already walks every discovered ``*.gguf``
through ``os.path.realpath`` + ``os.path.commonpath`` against ``extract_dir``.
The ``gguf`` branch only joined ``extract_dir / target.path`` and checked
``.exists()``, so a crafted manifest reached outside the extract dir three
ways: an absolute ``path`` (``pathlib`` drops the left operand when the right
one is absolute), a ``..`` traversal, or a symlink planted inside the can.

``DeployTarget``'s field validator rejects the first two at parse time, so
these use ``model_construct`` to bypass it — the point is that the deploy step
holds the line on its own rather than trusting its caller. The symlink case is
the one no schema check can see.
"""

from __future__ import annotations

import os
import sys

import pytest

from soup_cli.cans.run import _deploy_target
from soup_cli.cans.schema import DeployTarget


def _target(path: str) -> DeployTarget:
    """A kind=gguf target with the field validator bypassed."""
    return DeployTarget.model_construct(kind="gguf", name=None, path=path)


@pytest.fixture()
def extract_dir(tmp_path):
    d = tmp_path / "extract"
    d.mkdir()
    return d


def test_absolute_path_is_refused(tmp_path, extract_dir):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF")
    assert os.path.isabs(str(outside))

    with pytest.raises(ValueError, match="escapes the can extract dir"):
        _deploy_target(_target(str(outside)), extract_dir)


def test_dotdot_traversal_is_refused(tmp_path, extract_dir):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF")

    with pytest.raises(ValueError, match="escapes the can extract dir"):
        _deploy_target(_target(os.path.join("..", "outside.gguf")), extract_dir)


def test_symlinked_gguf_inside_the_can_is_refused(tmp_path, extract_dir):
    """The case no schema validator can catch: a legal-looking relative path
    whose parent directory is a symlink out of the extract dir."""
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "model.gguf").write_bytes(b"GGUF")
    try:
        os.symlink(str(outside_dir), str(extract_dir / "link"),
                   target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError) as exc:
        pytest.skip(f"symlinks unavailable on this machine: {exc}")

    with pytest.raises(ValueError, match="escapes the can extract dir"):
        _deploy_target(_target("link/model.gguf"), extract_dir)


def test_refusal_names_the_offending_path(extract_dir):
    with pytest.raises(ValueError) as excinfo:
        _deploy_target(_target(os.path.join("..", "evil.gguf")), extract_dir)
    assert "evil.gguf" in str(excinfo.value)


def test_legitimate_relative_gguf_still_works(extract_dir):
    (extract_dir / "sub").mkdir()
    (extract_dir / "sub" / "model.gguf").write_bytes(b"GGUF")

    assert _deploy_target(_target("sub/model.gguf"), extract_dir) == 0


def test_missing_relative_gguf_still_raises_file_not_found(extract_dir):
    with pytest.raises(FileNotFoundError, match="deploy gguf path not found"):
        _deploy_target(_target("model.gguf"), extract_dir)


def test_empty_path_still_rejected(extract_dir):
    with pytest.raises(ValueError, match="requires 'path'"):
        _deploy_target(_target(""), extract_dir)


def test_path_resolving_to_the_extract_dir_itself_is_refused(extract_dir):
    """``.`` is inside the dir but is not a file in the can."""
    with pytest.raises(ValueError, match="escapes the can extract dir"):
        _deploy_target(_target("."), extract_dir)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows drive-absolute")
def test_windows_drive_absolute_is_refused(tmp_path, extract_dir):
    outside = tmp_path / "drive.gguf"
    outside.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="escapes the can extract dir"):
        _deploy_target(_target(str(outside).replace("\\", "/")), extract_dir)


class TestTheSchemaLayerIsStillThere:
    """Defence in depth: the parse-time validator is the first line and stays."""

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../../x.gguf"])
    def test_manifest_validator_rejects_first(self, bad):
        with pytest.raises(ValueError):
            DeployTarget(kind="gguf", path=bad)
