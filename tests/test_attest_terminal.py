"""``soup attest`` shows paths, registry ids and error text as typed (#2489).

Every line that carries text from outside the program -- a path the user typed,
a registry id, a statement file name, an exception message, a sidecar's
backend -- keeps ``:name:`` aliases and backslashes before brackets, and drops
control and bidi characters. Fixed messages keep their colours.
"""

import json
import os
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.commands import attest as attest_cmd
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

CONTROLS = "\x9b‮"
VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x" + CONTROLS + "n"]
#: Values that can also be part of a file name on every CI platform.
FILE_VALUES = [
    "plain",
    "x[red]n",
    "x" + CONTROLS + "n",
    pytest.param(
        "x\\[n",
        marks=pytest.mark.skipif(sys.platform == "win32", reason="backslash in filename"),
    ),
    pytest.param(
        "n:rocket:",
        marks=pytest.mark.skipif(sys.platform == "win32", reason="colon in filename"),
    ),
]
SHA = "a" * 64
SIGSTORE_FLAGS = ("--cert-identity", "me@example.com", "--cert-oidc-issuer", "https://issuer")


def shown(value: str) -> str:
    return value.replace(CONTROLS, "")


def plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(attest_cmd, "console", Console(width=1000, height=100))


def _emit(*extra: str):
    args = ["emit", "--stage", "train", "--subject", "synthetic", "--sha", SHA, *extra]
    return runner.invoke(attest_cmd.app, args)


def _verify(statement: str = "stmt.json", *extra: str):
    return runner.invoke(
        attest_cmd.app, ["verify", statement, "--signature", "stmt.json.sig", *extra]
    )


def _fail_with(exc_type, value):
    def fail(*_args, **_kwargs):
        raise exc_type(value)

    return fail


def _ed25519_files(statement: str = "stmt.json") -> str:
    """Emit a real ed25519-signed statement at ``statement``; return the public PEM."""
    from soup_cli.utils.signing import (
        generate_ed25519_private_pem,
        private_key_from_pem,
        public_key_pem,
    )

    private_pem = generate_ed25519_private_pem()
    Path("signing.pem").write_text(private_pem, encoding="utf-8")
    result = _emit("--sign", "ed25519", "--key", "signing.pem", "-o", statement)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return public_key_pem(private_key_from_pem(private_pem))


def _sigstore_sidecar() -> None:
    Path("stmt.json").write_text('{"_type": "x"}', encoding="utf-8")
    Path("stmt.json.sig").write_text(
        json.dumps({"backend": "sigstore", "sigstore_bundle": "{}"}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# soup attest emit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALUES)
def test_an_invalid_statement_is_named_as_typed(monkeypatch, value):
    monkeypatch.setattr(attest_cmd, "AttestationStatement", _fail_with(ValueError, value))

    result = _emit()

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"Invalid attestation: {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_an_output_refusal_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        "soup_cli.utils.paths.enforce_under_cwd_and_no_symlink", _fail_with(ValueError, value)
    )

    result = _emit("-o", "out.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"Invalid output: {shown(value)}" in plain(result)
    assert not Path("out.json").exists()


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("exc_type, code", [(ValueError, 2), (RuntimeError, 1)])
def test_a_sign_failure_is_shown_as_typed(monkeypatch, value, exc_type, code):
    monkeypatch.setattr(attest_cmd, "sign_attestation", _fail_with(exc_type, value))

    result = _emit("-o", "out.json")

    assert result.exit_code == code, (result.output, repr(result.exception))
    assert f"Sign failed: {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_a_write_failure_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(attest_cmd, "write_attestation", _fail_with(OSError, value))

    result = _emit("-o", "out.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"Write failed: {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_receipt_names_the_written_file(value):
    name = f"out-{value}.json"

    result = _emit("-o", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    written = os.path.realpath(name)
    assert f"Wrote attestation -> {shown(written)} (signature: unsigned)" in plain(result)
    assert json.loads(Path(name).read_text(encoding="utf-8"))["_type"]


@pytest.mark.parametrize("value", VALUES)
def test_the_registry_receipt_names_the_entry_as_typed(monkeypatch, value):
    calls = []
    monkeypatch.setattr(
        "soup_cli.registry.attach.attach_artifact",
        lambda entry_id, **kwargs: calls.append(entry_id),
    )

    result = _emit("-o", "out.json", "--attach-to-registry", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == [value]
    assert f"Attached attestation to registry entry {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_a_registry_failure_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        "soup_cli.registry.attach.attach_artifact", _fail_with(ValueError, value)
    )

    result = _emit("-o", "out.json", "--attach-to-registry", "entry")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Error: could not attach to registry: {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_a_registry_import_failure_is_shown_as_typed(monkeypatch, value):
    broken = types.ModuleType("soup_cli.registry.attach")

    def missing(name):
        raise ImportError(value)

    broken.__getattr__ = missing
    monkeypatch.setitem(sys.modules, "soup_cli.registry.attach", broken)

    result = _emit("-o", "out.json", "--attach-to-registry", "entry")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Error: could not import registry attach helper: {shown(value)}" in plain(result)


# ---------------------------------------------------------------------------
# soup attest verify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALUES)
def test_a_path_refusal_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        "soup_cli.utils.paths.enforce_under_cwd_and_no_symlink", _fail_with(ValueError, value)
    )

    result = _verify()

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert shown(value) in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_an_unreadable_input_is_shown_as_typed(monkeypatch, value):
    _sigstore_sidecar()
    monkeypatch.setattr(json, "load", _fail_with(ValueError, value))

    result = _verify()

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"Could not read inputs: {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "exc_type, code, label",
    [
        (RuntimeError, 1, "Sigstore verification unavailable"),
        (ValueError, 3, "Sigstore attestation INVALID"),
    ],
)
def test_a_sigstore_failure_is_shown_as_typed(monkeypatch, value, exc_type, code, label):
    _sigstore_sidecar()
    monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", _fail_with(exc_type, value))

    result = _verify("stmt.json", *SIGSTORE_FLAGS)

    assert result.exit_code == code, (result.output, repr(result.exception))
    assert f"{label}: {shown(value)}" in plain(result)


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_sigstore_receipt_names_the_statement(monkeypatch, value):
    _sigstore_sidecar()
    name = f"stmt-{value}.json"
    Path(name).write_text('{"_type": "x"}', encoding="utf-8")
    monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", lambda *a, **k: None)

    result = _verify(name, *SIGSTORE_FLAGS)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Attestation Sigstore signature valid ({shown(name)})" in plain(result)


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_ed25519_receipt_names_the_statement(value):
    name = f"stmt-{value}.json"
    _ed25519_files(name)
    os.replace(name + ".sig", "stmt.json.sig")

    result = _verify(name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Attestation signature valid ({shown(name)})" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_a_sidecar_backend_is_quoted_as_written(value):
    Path("stmt.json").write_text('{"_type": "x"}', encoding="utf-8")
    Path("stmt.json.sig").write_text(
        json.dumps({"backend": value, "signature": "00"}), encoding="utf-8"
    )

    result = _verify()

    assert result.exit_code == 3, (result.output, repr(result.exception))
    # repr() turns the two controls into their escape text, so nothing is dropped.
    assert f"Signature backend {value!r} is not cryptographically verifiable" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_an_unreadable_public_key_is_shown_as_typed(monkeypatch, value):
    _ed25519_files()
    Path("trusted.pub").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "soup_cli.utils.signing.read_public_key_file", _fail_with(ValueError, value)
    )

    result = _verify("stmt.json", "--public-key", "trusted.pub")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"Could not read --public-key: {shown(value)}" in plain(result)


# ---------------------------------------------------------------------------
# The colours stay as they were
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(monkeypatch):
    monkeypatch.setattr(
        attest_cmd,
        "console",
        Console(width=1000, height=100, force_terminal=True, color_system="standard"),
    )


def test_the_emit_receipt_keeps_its_styles(coloured):
    result = _emit("-o", "out.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    path = os.path.realpath("out.json")
    expected = f"\x1b[32mWrote attestation\x1b[0m -> {path} \x1b[2m(signature: unsigned)\x1b[0m"
    assert expected in result.output


def test_the_registry_receipt_keeps_its_styles(coloured, monkeypatch):
    monkeypatch.setattr("soup_cli.registry.attach.attach_artifact", lambda *a, **k: None)

    result = _emit("-o", "out.json", "--attach-to-registry", "entry")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    expected = "\x1b[32mAttached\x1b[0m attestation to registry entry \x1b[1mentry\x1b[0m"
    assert expected in result.output


def test_a_registry_failure_keeps_its_red_label(coloured, monkeypatch):
    monkeypatch.setattr(
        "soup_cli.registry.attach.attach_artifact", _fail_with(ValueError, "boom")
    )

    result = _emit("-o", "out.json", "--attach-to-registry", "entry")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "\x1b[31mError:\x1b[0m could not attach to registry: boom" in result.output


@pytest.mark.parametrize(
    "target, exc_type, line",
    [
        ("AttestationStatement", ValueError, "Invalid attestation: boom"),
        ("sign_attestation", RuntimeError, "Sign failed: boom"),
        ("write_attestation", OSError, "Write failed: boom"),
    ],
)
def test_every_emit_error_line_is_red(coloured, monkeypatch, target, exc_type, line):
    monkeypatch.setattr(attest_cmd, target, _fail_with(exc_type, "boom"))

    result = _emit("-o", "out.json")

    assert result.exit_code != 0
    assert f"\x1b[31m{line}\x1b[0m" in result.output


def test_the_ed25519_receipt_keeps_its_styles(coloured):
    _ed25519_files()

    result = _verify()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    expected = "\x1b[32mAttestation signature valid\x1b[0m \x1b[2m(stmt.json)\x1b[0m"
    assert expected in result.output


def test_the_backend_warning_stays_yellow(coloured):
    Path("stmt.json").write_text('{"_type": "x"}', encoding="utf-8")
    Path("stmt.json.sig").write_text(
        json.dumps({"backend": "hmac", "signature": "00"}), encoding="utf-8"
    )

    result = _verify()

    assert result.exit_code == 3
    expected = "\x1b[33mSignature backend 'hmac' is not cryptographically verifiable.\x1b[0m"
    assert expected in result.output


@pytest.mark.parametrize(
    "exc_type, line",
    [
        (RuntimeError, "Sigstore verification unavailable: boom"),
        (ValueError, "Sigstore attestation INVALID: boom"),
    ],
)
def test_every_sigstore_error_line_is_red(coloured, monkeypatch, exc_type, line):
    _sigstore_sidecar()
    monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", _fail_with(exc_type, "boom"))

    result = _verify("stmt.json", *SIGSTORE_FLAGS)

    assert f"\x1b[31m{line}\x1b[0m" in result.output
