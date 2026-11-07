r"""A rooted path without a drive is never a public identifier, on any Python.

Python 3.13 changed ``ntpath.isabs``: a path that starts with exactly one ``/`` or ``\`` and
names no drive (``/x``, ``\x``) is no longer absolute, where 3.11 and 3.12 call it absolute.
Three checks relied on ``isabs`` alone to refuse such a path, so on 3.13 they began to accept
what they had refused (#1805):

* ``best_of_n_artifact``: the sampler ``model`` and ``revision`` must be public identifiers
  (``/x`` passed on Windows, where ``os.path`` is ``ntpath``);
* ``quest``: a v2 ``base_model`` must not name a local path (``\x`` passed on every OS, the
  check calls ``ntpath.isabs`` itself).

Each check now refuses a leading separator by name. Every case runs twice: with ``isabs`` as
the running interpreter has it, and with ``isabs`` replaced, in the two modules only, by the
3.13 Windows rule. The second run fails on 3.12 too if a check goes back to ``isabs`` alone.
"""

from __future__ import annotations

import ntpath
import os
import sys

import pytest

from soup_cli.utils import best_of_n_artifact, quest
from soup_cli.utils.best_of_n_artifact import validate_sampler_spec
from soup_cli.utils.quest import (
    CALIBRATION_SCALES,
    EXPECTED_MODULES,
    build_metadata,
    validate_metadata,
)

_REAL_NT_ISABS = ntpath.isabs
_SEPARATORS = ("/", "\\")
#: Exactly one leading separator, no drive: absolute on 3.11-3.12, not on 3.13.
ROOTED_WITHOUT_A_DRIVE = ("/x", "\\x", "/org/model", "\\org\\model")


def _isabs_as_on_windows_313(path: str) -> bool:
    """``ntpath.isabs`` with Python 3.13's rule; everything else is the real function."""
    text = os.fspath(path)
    if text[:1] in _SEPARATORS and text[1:2] not in _SEPARATORS:
        return False
    return _REAL_NT_ISABS(text)


class _PathAsOnWindows313:
    """``os.path`` of a Windows 3.13 host, as far as ``isabs`` goes."""

    isabs = staticmethod(_isabs_as_on_windows_313)

    def __getattr__(self, name: str):
        return getattr(os.path, name)


class _OsAsOnWindows313:
    path = _PathAsOnWindows313()

    def __getattr__(self, name: str):
        return getattr(os, name)


class _NtpathAs313:
    isabs = staticmethod(_isabs_as_on_windows_313)

    def __getattr__(self, name: str):
        return getattr(ntpath, name)


@pytest.fixture(params=["this-python", "windows-3.13"])
def isabs_rule(request, monkeypatch):
    """Patch only the two modules under test, never ``os.path`` itself, so nothing else in
    the process sees the simulated rule."""
    if request.param == "windows-3.13":
        monkeypatch.setattr(best_of_n_artifact, "os", _OsAsOnWindows313())
        monkeypatch.setattr(best_of_n_artifact, "ntpath", _NtpathAs313())
        monkeypatch.setattr(quest, "ntpath", _NtpathAs313())
        assert best_of_n_artifact.os.path.isabs("/x") is False
        assert quest.ntpath.isabs("\\x") is False
    return request.param


def _sampler(**changes) -> dict:
    sampler = {
        "kind": "local",
        "model": "org/model",
        "n": 2,
        "temperature": 0.7,
        "max_new_tokens": 16,
        "revision": "main",
        "device": "cpu",
        "seed": 0,
        "trust_remote_code": False,
    }
    sampler.update(changes)
    return sampler


def _quest_metadata(base_model: str) -> dict:
    metadata = build_metadata(
        activation_scales={name: CALIBRATION_SCALES[0] for name in EXPECTED_MODULES},
        base_model="org/model",
        calibration_sha256="ab" * 32,
    )
    metadata["base_model"] = base_model
    return metadata


class TestTheSimulatedRule:
    """CONTROL: the stand-in must be 3.13's rule, or the second run proves nothing."""

    def test_one_leading_separator_without_a_drive_is_not_absolute(self):
        for path in ROOTED_WITHOUT_A_DRIVE:
            assert _isabs_as_on_windows_313(path) is False, path

    def test_a_drive_or_a_unc_share_still_is(self):
        assert _isabs_as_on_windows_313("C:\\x") is True
        assert _isabs_as_on_windows_313("C:/x") is True
        assert _isabs_as_on_windows_313("\\\\server\\share\\x") is True

    @pytest.mark.skipif(sys.version_info < (3, 13), reason="the rule of 3.13 and later")
    def test_it_matches_the_real_function_from_3_13(self):
        for path in (*ROOTED_WITHOUT_A_DRIVE, "C:\\x", "\\\\server\\share\\x", "org/model"):
            assert ntpath.isabs(path) is _isabs_as_on_windows_313(path), path

    @pytest.mark.skipif(sys.version_info >= (3, 13), reason="the rule before 3.13")
    def test_before_3_13_the_real_function_disagrees(self):
        """The difference the checks must not depend on."""
        assert ntpath.isabs("/x") is True
        assert ntpath.isabs("\\x") is True


@pytest.mark.parametrize("path", ROOTED_WITHOUT_A_DRIVE)
class TestBestOfNSampler:
    def test_a_rooted_model_is_not_a_public_identifier(self, isabs_rule, path):
        with pytest.raises(ValueError, match="candidate sampler model must be a public identifier"):
            validate_sampler_spec(_sampler(model=path))

    def test_a_rooted_model_is_refused_for_a_provider_too(self, isabs_rule, path):
        provider = {
            "kind": "provider",
            "provider": "ollama",
            "model": path,
            "n": 2,
            "temperature": 0.7,
            "max_new_tokens": 16,
        }
        with pytest.raises(ValueError, match="candidate sampler model must be a public identifier"):
            validate_sampler_spec(provider)

    def test_a_rooted_revision_is_refused(self, isabs_rule, path):
        with pytest.raises(ValueError, match="candidate sampler revision is invalid"):
            validate_sampler_spec(_sampler(revision=path))


@pytest.mark.parametrize("path", ROOTED_WITHOUT_A_DRIVE)
def test_a_rooted_quest_base_model_is_a_local_path(isabs_rule, path):
    with pytest.raises(ValueError, match="QuEST v2 base_model must not contain a local path"):
        validate_metadata(_quest_metadata(path))


def test_public_identifiers_still_pass(isabs_rule):
    """The new refusal is a leading separator only: an organisation/name id is untouched."""
    spec = validate_sampler_spec(_sampler(model="org/model", revision="v1.0"))
    assert (spec["model"], spec["revision"]) == ("org/model", "v1.0")
    assert validate_metadata(_quest_metadata("org/model"))["base_model"] == "org/model"
