"""A cache-root variable that Soup drops says so, once per process (#2610).

``SOUP_LAYER_STREAM_CACHE_DIR`` and ``SOUP_SPECTRUM_CACHE_DIR`` name the folder the
layer-streaming shard cache and the spectrum scan cache live in. Both are bounded to
``$HOME``, the working directory and the temp directory; a value outside all three
(or one carrying a control character) falls back to ``~/.soup/...``. That boundary is
unchanged. What these tests pin is that the fallback is no longer silent:

* the warning names the variable, the rejected value (control characters made visible,
  markup escaped), the reason and the folder used instead, and the fix;
* it is written once per variable and value, however often a resolver runs;
* an accepted value, an unset or empty variable and an explicit ``cache_dir`` argument
  print nothing;
* the returned folder is still the default, so loosening the rule fails here;
* both variables go through ``utils/cache_roots.py``.

The sandbox fixture points ``$HOME``, the working directory and the temp directory at
three sibling folders of ``tmp_path``, so ``tmp_path / "data-disk"`` is outside all three
on every platform and nothing is written under the real home directory.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.utils import cache_roots

LAYER_VAR = "SOUP_LAYER_STREAM_CACHE_DIR"
SPECTRUM_VAR = "SOUP_SPECTRUM_CACHE_DIR"
OUTSIDE_REASON = "outside $HOME, the working directory and the temp directory"


@pytest.fixture(autouse=True)
def _fresh_once_guard():
    """The once-per-process record is module state: start and leave each test empty."""
    cache_roots._WARNED.clear()
    yield
    cache_roots._WARNED.clear()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = tmp_path / "work"
    temp = tmp_path / "tmp"
    outside = tmp_path / "data-disk" / "cache"
    for folder in (home, work, temp, outside.parent):
        folder.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.delenv(LAYER_VAR, raising=False)
    monkeypatch.delenv(SPECTRUM_VAR, raising=False)
    return SimpleNamespace(home=home, work=work, temp=temp, outside=outside)


@pytest.fixture
def warned(caplog):
    """The WARNING messages the helper wrote, whatever the root logger level is."""
    caplog.set_level(logging.WARNING, logger=cache_roots.logger.name)

    def messages():
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == cache_roots.logger.name and record.levelno == logging.WARNING
        ]

    return messages


def _default_layer_root(sandbox) -> str:
    """The default is returned as ``~/.soup/layer-stream`` is spelled, not realpath'd."""
    return str(sandbox.home / ".soup" / "layer-stream")


def _default_spectrum_root(sandbox) -> str:
    return str(sandbox.home / ".soup" / "spectrum")


def _expected_message(variable: str, shown: str, reason: str, used: str, fix: str) -> str:
    return f"{variable}='{shown}' is ignored: {reason}. Using {used} instead. {fix}"


OUTSIDE_FIX = (
    "To use that folder, run soup from a working directory on the same drive, "
    "or put the folder under $HOME."
)


class TestLayerStreamCacheDir:
    def test_outside_folder_warns_and_still_falls_back(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_cache_root

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside))
        got = resolve_cache_root()

        assert got == _default_layer_root(sandbox)
        assert not os.path.exists(sandbox.outside)
        assert warned() == [
            _expected_message(
                LAYER_VAR,
                str(sandbox.outside),
                "it is " + OUTSIDE_REASON,
                _default_layer_root(sandbox),
                OUTSIDE_FIX,
            )
        ]

    def test_resolve_shard_dir_falls_back_and_warns(self, sandbox, warned, monkeypatch):
        """The call the trainer makes: the shard folder lands under the default root."""
        from soup_cli.utils.layer_shard import resolve_shard_dir

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside))
        chosen = os.path.realpath(resolve_shard_dir("org/model"))

        default_root = os.path.realpath(_default_layer_root(sandbox))
        assert os.path.commonpath([chosen, default_root]) == default_root
        assert not os.path.exists(sandbox.outside)
        messages = warned()
        assert len(messages) == 1
        assert LAYER_VAR in messages[0]
        assert _default_layer_root(sandbox) in messages[0]

    def test_unset_and_empty_are_silent(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_cache_root

        assert resolve_cache_root() == _default_layer_root(sandbox)
        monkeypatch.setenv(LAYER_VAR, "")
        assert resolve_cache_root() == _default_layer_root(sandbox)
        assert warned() == []

    @pytest.mark.parametrize("bound", ["home", "work", "temp"])
    def test_folder_inside_a_bound_is_honoured_silently(
        self, sandbox, warned, monkeypatch, bound
    ):
        from soup_cli.utils.layer_shard import resolve_cache_root

        target = getattr(sandbox, bound) / "stream-cache"
        monkeypatch.setenv(LAYER_VAR, str(target))

        assert resolve_cache_root() == os.path.realpath(str(target))
        assert warned() == []

    def test_explicit_cache_dir_ignores_the_variable_silently(
        self, sandbox, warned, monkeypatch
    ):
        from soup_cli.utils.layer_shard import resolve_cache_root

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside))
        explicit = sandbox.work / "explicit"

        assert resolve_cache_root(str(explicit)) == os.path.realpath(str(explicit))
        assert warned() == []

    def test_repeated_resolution_warns_once(self, sandbox, warned, monkeypatch):
        """train.py, stream_setup.py and doctor.py each resolve the root in one process."""
        from soup_cli.utils.layer_shard import resolve_cache_root, resolve_shard_dir

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside))
        for _ in range(3):
            resolve_cache_root()
        resolve_shard_dir("org/model")
        resolve_shard_dir("org/other")

        assert len(warned()) == 1

    def test_a_changed_value_warns_again(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_cache_root

        second = sandbox.outside.parent / "second"
        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside))
        resolve_cache_root()
        monkeypatch.setenv(LAYER_VAR, str(second))
        resolve_cache_root()
        resolve_cache_root()

        messages = warned()
        assert len(messages) == 2
        assert str(sandbox.outside) in messages[0]
        assert str(second) in messages[1]

    def test_a_warning_for_one_variable_does_not_hide_the_other(
        self, sandbox, warned, monkeypatch
    ):
        from soup_cli.utils.layer_shard import resolve_cache_root
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside))
        monkeypatch.setenv(SPECTRUM_VAR, str(sandbox.outside))
        resolve_cache_root()
        resolve_cache_dir()

        messages = warned()
        assert len(messages) == 2
        assert messages[0].startswith(LAYER_VAR + "=")
        assert messages[1].startswith(SPECTRUM_VAR + "=")


class TestSpectrumCacheDir:
    def test_outside_folder_warns_and_still_falls_back(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        monkeypatch.setenv(SPECTRUM_VAR, str(sandbox.outside))
        got = resolve_cache_dir()

        assert got == _default_spectrum_root(sandbox)
        assert not os.path.exists(sandbox.outside)
        assert warned() == [
            _expected_message(
                SPECTRUM_VAR,
                str(sandbox.outside),
                "it is " + OUTSIDE_REASON,
                _default_spectrum_root(sandbox),
                OUTSIDE_FIX,
            )
        ]

    def test_unset_and_empty_are_silent(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        assert resolve_cache_dir() == _default_spectrum_root(sandbox)
        monkeypatch.setenv(SPECTRUM_VAR, "")
        assert resolve_cache_dir() == _default_spectrum_root(sandbox)
        assert warned() == []

    @pytest.mark.parametrize("bound", ["home", "work", "temp"])
    def test_folder_inside_a_bound_is_honoured_silently(
        self, sandbox, warned, monkeypatch, bound
    ):
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        target = getattr(sandbox, bound) / "spectrum-cache"
        monkeypatch.setenv(SPECTRUM_VAR, str(target))

        assert resolve_cache_dir() == os.path.realpath(str(target))
        assert os.path.isdir(target)
        assert warned() == []

    def test_explicit_cache_dir_ignores_the_variable_silently(
        self, sandbox, warned, monkeypatch
    ):
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        monkeypatch.setenv(SPECTRUM_VAR, str(sandbox.outside))
        explicit = sandbox.work / "explicit"

        assert resolve_cache_dir(str(explicit)) == os.path.realpath(str(explicit))
        assert warned() == []

    def test_repeated_resolution_warns_once(self, sandbox, warned, monkeypatch):
        """``spectrum_scan`` resolves the root from three places in one scan."""
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        monkeypatch.setenv(SPECTRUM_VAR, str(sandbox.outside))
        for _ in range(4):
            resolve_cache_dir()

        assert len(warned()) == 1


class TestRejectedValueIsRenderedSafely:
    @pytest.mark.parametrize(
        "char", ["\x1b", "\r", "\n", "\t", "\x07", "\x01"], ids=lambda c: repr(c)
    )
    def test_control_character_is_rejected_and_made_visible(
        self, sandbox, warned, monkeypatch, char
    ):
        """The value is otherwise valid (under the temp dir): the control character is
        the only reason to refuse it, and the message must not carry it."""
        from soup_cli.utils.layer_shard import resolve_cache_root

        value = str(sandbox.temp / "cache") + char + "tail"
        monkeypatch.setenv(LAYER_VAR, value)

        assert resolve_cache_root() == _default_layer_root(sandbox)
        (message,) = warned()
        assert char not in message
        assert message == _expected_message(
            LAYER_VAR,
            str(sandbox.temp / "cache") + f"\\x{ord(char):02x}" + "tail",
            "it contains control characters",
            _default_layer_root(sandbox),
            "Set it again without the control characters.",
        )

    def test_control_character_in_the_spectrum_variable_is_rejected_too(
        self, sandbox, warned, monkeypatch
    ):
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        monkeypatch.setenv(SPECTRUM_VAR, str(sandbox.temp / "cache") + "\x1b[2J")

        assert resolve_cache_dir() == _default_spectrum_root(sandbox)
        (message,) = warned()
        assert "\x1b" not in message
        assert "\\x1b" in message
        assert "contains control characters" in message
        assert message.startswith(SPECTRUM_VAR + "=")

    @pytest.mark.parametrize(
        "char", ["\x7f", "\x9b", "\u202e", "\u200b"], ids=lambda c: hex(ord(c))
    )
    def test_terminal_active_characters_are_stripped_from_an_outside_value(
        self, sandbox, warned, monkeypatch, char
    ):
        """DEL, a C1 control (U+009B is a one-character CSI on some terminals) and the
        bidi / zero-width format characters are not refused by the rule, so they reach
        the message only through ``for_terminal``."""
        from soup_cli.utils.layer_shard import resolve_cache_root

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside) + char + "x")

        assert resolve_cache_root() == _default_layer_root(sandbox)
        (message,) = warned()
        assert char not in message
        assert OUTSIDE_REASON in message

    def test_rich_markup_in_the_value_is_escaped(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_cache_root

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside) + "[bold red]x[/]")

        resolve_cache_root()
        (message,) = warned()
        assert "\\[bold red]" in message
        assert "\\[/]" in message

    def test_a_very_long_value_is_shortened(self, sandbox, warned, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_cache_root

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside) + "/" + "/".join(["abcdefghij"] * 400))

        resolve_cache_root()
        (message,) = warned()
        assert len(message) < 800
        assert "..." in message
        assert str(sandbox.outside) in message


@pytest.mark.skipif(os.name != "nt", reason="case folding in is_under is a Windows rule")
class TestWindowsCaseFolding:
    def test_a_differently_cased_path_inside_a_bound_is_still_accepted(
        self, sandbox, warned, monkeypatch
    ):
        from soup_cli.utils.layer_shard import resolve_cache_root

        target = str(sandbox.work / "Stream-Cache")
        monkeypatch.setenv(LAYER_VAR, target.swapcase())

        got = resolve_cache_root()
        assert got.lower() == os.path.realpath(target).lower()
        assert warned() == []

    def test_a_differently_cased_outside_path_is_still_refused(
        self, sandbox, warned, monkeypatch
    ):
        from soup_cli.utils.layer_shard import resolve_cache_root

        monkeypatch.setenv(LAYER_VAR, str(sandbox.outside).swapcase())

        assert resolve_cache_root() == _default_layer_root(sandbox)
        assert len(warned()) == 1


class TestOneSharedHelper:
    def test_both_resolvers_call_the_shared_helper(self, sandbox, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_cache_root
        from soup_cli.utils.spectrum_scan import resolve_cache_dir

        seen = []
        real = cache_roots.env_cache_root

        def spy(variable, default):
            seen.append(variable)
            return real(variable, default)

        monkeypatch.setattr(cache_roots, "env_cache_root", spy)
        resolve_cache_root()
        resolve_cache_dir()

        assert seen == [LAYER_VAR, SPECTRUM_VAR]

    def test_the_helper_names_both_variables(self):
        assert cache_roots.LAYER_STREAM_CACHE_ENV == LAYER_VAR
        assert cache_roots.SPECTRUM_CACHE_ENV == SPECTRUM_VAR


_CHILD = """
import sys
from soup_cli.utils.log_level import LogLevel, apply_logging_level, setup_logging
tier = LogLevel(sys.argv[1])
setup_logging(tier)
apply_logging_level(tier)
from soup_cli.utils.layer_shard import resolve_cache_root
print(resolve_cache_root())
"""


class TestVisibleOnADefaultRun:
    """``soup``'s callback runs ``setup_logging`` + ``apply_logging_level`` with the tier
    and installs no root handler, so a ``soup_cli.*`` logger reaches the user through
    the stdlib last-resort handler (stderr, WARNING and above). The child process repeats
    exactly those two calls, with nothing else configured, to show the warning is on
    stderr at the default tier and gone at ``quiet``."""

    @staticmethod
    def _run(sandbox, tier: str):
        package_parent = str(Path(cache_roots.__file__).resolve().parents[2])
        env = {
            **os.environ,
            "PYTHONPATH": package_parent,
            "HOME": str(sandbox.home),
            "USERPROFILE": str(sandbox.home),
            "TMPDIR": str(sandbox.temp),
            "TEMP": str(sandbox.temp),
            "TMP": str(sandbox.temp),
            LAYER_VAR: str(sandbox.outside),
        }
        return subprocess.run(
            [sys.executable, "-c", _CHILD, tier],
            cwd=sandbox.work,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_default_tier_prints_the_warning_on_stderr(self, sandbox):
        done = self._run(sandbox, "normal")

        assert done.returncode == 0, (done.stdout, done.stderr)
        assert LAYER_VAR in done.stderr
        assert "is ignored" in done.stderr
        assert OUTSIDE_REASON in done.stderr
        assert done.stdout.strip() == _default_layer_root(sandbox)

    def test_quiet_tier_keeps_it_off_the_screen(self, sandbox):
        done = self._run(sandbox, "quiet")

        assert done.returncode == 0, (done.stdout, done.stderr)
        assert LAYER_VAR not in done.stderr
        assert done.stdout.strip() == _default_layer_root(sandbox)
