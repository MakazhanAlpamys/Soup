"""#3045: ``soup airgap-bundle --bundle-size-cap`` refuses what it cannot count in bytes.

The command checked ``bundle_size_cap <= 0`` and then converted the cap with
``int(cap * 1024 ** 3)``. ``nan`` and ``inf`` pass ``<= 0``, and ``int()``
raised ``ValueError`` for ``nan`` and ``OverflowError`` for an infinity; a
finite cap above about 1.67e299 GiB overflows to infinity in the
multiplication and raised the same ``OverflowError``. All three reached the
generic error handler, which printed the exception and exited 1. They are now
refused like ``0``, with exit 2 and before anything is read. Every cap that
built a bundle before still builds one.
"""

from __future__ import annotations

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

pytestmark = pytest.mark.unit

_NOT_FINITE = "--bundle-size-cap must be a finite number > 0"
_TOO_LARGE = "--bundle-size-cap is too large to count in bytes"
_NOT_POSITIVE = "--bundle-size-cap must be > 0"

# Every spelling ``float()`` reads as NaN or as positive infinity.
_NON_FINITE_SPELLINGS = ["nan", "NaN", "NAN", "+nan", "-nan", "inf", "+inf", "Infinity", "1e999"]
# Finite in GiB, but the byte count overflows a float: 1.7e299 * 1024**3 > 1.79e308.
_OVERFLOWING = ["1.7e299", "1e300"]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A working directory with a small model folder ``m``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_TELEMETRY", "0")
    monkeypatch.setenv("COLUMNS", "400")
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"safe-weight-bytes")
    return tmp_path


def _bundle(cap: str, model: str = "m"):
    argv = ["airgap-bundle", "-o", "out.tar", "--model", model, "--bundle-size-cap", cap]
    return make_cli_runner().invoke(app, argv)


def _plain(result) -> str:
    """The output without colour, panel borders or line breaks."""
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


# --- refused as a usage error --------------------------------------------------


@pytest.mark.parametrize("cap", _NON_FINITE_SPELLINGS)
def test_nan_and_infinity_are_refused_as_a_usage_error(workdir, cap):
    result = _bundle(cap)

    assert result.exit_code == 2, result.output
    assert _plain(result) == _NOT_FINITE
    assert not (workdir / "out.tar").exists()


@pytest.mark.parametrize("cap", _OVERFLOWING)
def test_a_cap_whose_byte_count_overflows_is_refused_as_a_usage_error(workdir, cap):
    result = _bundle(cap)

    assert result.exit_code == 2, result.output
    assert _plain(result) == _TOO_LARGE
    assert not (workdir / "out.tar").exists()


@pytest.mark.parametrize(("cap", "line"), [("nan", _NOT_FINITE), ("1e300", _TOO_LARGE)])
def test_the_cap_is_refused_before_the_model_folder_is_read(workdir, cap, line):
    result = _bundle(cap, model="no-such-model")

    assert result.exit_code == 2, result.output
    assert _plain(result) == line
    assert "no-such-model" not in _plain(result)


# --- controls that the refusal must not touch -----------------------------------


@pytest.mark.parametrize("cap", ["0", "-1", "-inf"])
def test_zero_and_negative_caps_keep_their_refusal(workdir, cap):
    result = _bundle(cap)

    assert result.exit_code == 2, result.output
    assert _plain(result) == _NOT_POSITIVE
    assert not (workdir / "out.tar").exists()


@pytest.mark.parametrize("cap", ["100", "1e20", "1.6e299"])
def test_a_cap_that_built_a_bundle_still_builds_one(workdir, cap):
    result = _bundle(cap)

    assert result.exit_code == 0, result.output
    assert (workdir / "out.tar").is_file()


def test_a_cap_below_the_model_size_still_stops_the_build(workdir):
    (workdir / "m" / "weights.safetensors").write_bytes(b"x" * 10_000)

    result = _bundle("0.000001")  # about 1073 bytes, below the 10 000-byte model

    assert result.exit_code != 0, result.output
    assert not (workdir / "out.tar").exists()
