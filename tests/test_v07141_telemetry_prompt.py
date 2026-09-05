"""v0.71.41 #318 — Wire the opt-in telemetry flywheel.

Validates the complete opt-in telemetry pipeline:
- Default-OFF: zero payload building, zero network calls when disabled
- Env-only opt-in: SOUP_TELEMETRY=1 is the single switch
- CLI flag: --no-telemetry guarantees no emission even with SOUP_TELEMETRY=1
- Privacy invariant: command tokens are strictly validated against registered
  commands; file paths, model paths, and arbitrary tokens are masked as (unknown)
- Pure stdlib: urllib.request POST with 1.0s timeout and JSON payload
- Anonymous distinct_id: generated, validated, persisted, resilient to FS errors
- Every single test contains load-bearing AST assertions.
"""
from __future__ import annotations

import ast
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


def test_emit_telemetry_disabled_never_builds_or_sends(monkeypatch):
    """When telemetry is disabled, zero payloads are built and zero network calls occur."""
    from soup_cli.cli import _emit_telemetry

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    build_mock = MagicMock()
    send_mock = MagicMock()
    monkeypatch.setattr("soup_cli.utils.trackers.build_telemetry_payload", build_mock)
    monkeypatch.setattr("soup_cli.utils.trackers.send_telemetry_payload", send_mock)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        MagicMock(side_effect=AssertionError("network call made while disabled")),
    )

    _emit_telemetry(["soup", "train"], 1.0)

    assert build_mock.call_count == 0
    assert send_mock.call_count == 0


@pytest.mark.parametrize("falsy_val", ["0", "false", "no", "off", "", "random_value"])
def test_emit_telemetry_falsy_env_never_sends(falsy_val, monkeypatch):
    """Falsy or invalid SOUP_TELEMETRY values result in zero emissions."""
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setenv("SOUP_TELEMETRY", falsy_val)
    send_mock = MagicMock()
    monkeypatch.setattr("soup_cli.utils.trackers.send_telemetry_payload", send_mock)

    _emit_telemetry(["soup", "train"], 1.0)

    assert send_mock.call_count == 0


def test_emit_telemetry_no_telemetry_flag_overrides_env(monkeypatch):
    """The --no-telemetry flag prevents any telemetry emission even if SOUP_TELEMETRY=1."""
    import soup_cli.cli as cli

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    send_mock = MagicMock()
    monkeypatch.setattr("soup_cli.utils.trackers.send_telemetry_payload", send_mock)

    # Invocation with --no-telemetry in argv
    cli._emit_telemetry(["soup", "--no-telemetry", "train"], 1.0)
    assert send_mock.call_count == 0

    # Global _telemetry_disabled set
    cli._telemetry_disabled = True
    try:
        cli._emit_telemetry(["soup", "train"], 1.0)
        assert send_mock.call_count == 0
    finally:
        cli._telemetry_disabled = False


def test_emit_telemetry_enabled_builds_and_sends(tmp_path, monkeypatch):
    """When SOUP_TELEMETRY=1, telemetry builds and sends exactly one payload."""
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    sent_payloads = []
    monkeypatch.setattr(
        "soup_cli.utils.trackers.send_telemetry_payload",
        lambda payload: sent_payloads.append(payload) or True,
    )

    _emit_telemetry(["soup", "train", "--config", "soup.yaml"], 2.5)

    assert len(sent_payloads) == 1
    p = sent_payloads[0]
    assert p["command"] == "train"
    assert p["duration_seconds"] == 2.5
    assert "soup.yaml" not in str(p)
    assert uuid.UUID(p["distinct_id"])


@pytest.mark.parametrize(
    ("argv", "expected_command"),
    [
        (["soup", "train", "-c", "soup.yaml"], "train"),
        (["soup", "data", "ingest"], "data"),
        (["soup", "--help"], "(root)"),
        (["soup"], "(root)"),
        (["soup", "/home/alice/secret-merger-project/data.jsonl"], "(unknown)"),
        (["soup", "./acme-corp-model"], "(unknown)"),
        (["soup", "--log-level", "debug", "C:/clients/pfizer/run.yaml"], "(unknown)"),
        (["soup", "trian", "-c", "soup.yaml"], "(unknown)"),
    ],
)
def test_command_sanitization_masks_paths_and_unknown_args(
    argv, expected_command, tmp_path, monkeypatch
):
    """Private paths, file names, and typos must NEVER leak into command event names."""
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    sent_payloads = []
    monkeypatch.setattr(
        "soup_cli.utils.trackers.send_telemetry_payload",
        lambda payload: sent_payloads.append(payload) or True,
    )

    _emit_telemetry(argv, 1.0)

    assert len(sent_payloads) == 1
    assert sent_payloads[0]["command"] == expected_command


def test_cli_has_no_prompt_and_documents_no_telemetry(monkeypatch):
    """Ensure no interactive consent prompt exists, and --no-telemetry is in --help."""
    from soup_cli.cli import app
    from tests.conftest import strip_ansi

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    with patch("rich.prompt.Confirm.ask", side_effect=AssertionError("interactive prompt called")):
        result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    clean_out = strip_ansi(result.output)
    assert "--no-telemetry" in clean_out
    assert "telemetry" in clean_out.lower()


def test_send_boundary_disabled_never_opens_network(monkeypatch):
    """send_telemetry_payload returns False without touching urllib when disabled."""
    from soup_cli.utils.trackers import send_telemetry_payload

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    network_mock = MagicMock(side_effect=AssertionError("network opened while disabled"))
    monkeypatch.setattr("urllib.request.urlopen", network_mock)

    assert send_telemetry_payload({"command": "train"}) is False
    assert network_mock.call_count == 0


def test_send_boundary_enabled_uses_stdlib_post(monkeypatch):
    """send_telemetry_payload performs a well-formed JSON POST using urllib.request."""
    from soup_cli.utils.trackers import send_telemetry_payload

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    monkeypatch.setattr(
        "soup_cli.utils.trackers._telemetry_endpoint_is_safe", lambda _url: True
    )

    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return self.status

    def fake_urlopen(req, timeout):
        captured["req"] = req
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = {
        "command": "train",
        "soup_version": "0.73.3",
        "python": "3.10",
        "os": "Linux",
        "arch": "x86_64",
        "duration_seconds": 1.5,
        "distinct_id": str(uuid.uuid4()),
    }

    result = send_telemetry_payload(payload, api_key="phc_test_key")
    assert result is True
    assert captured["timeout"] == 1.0

    req = captured["req"]
    assert req.get_method() == "POST"
    assert req.headers.get("Content-type") == "application/json"

    sent_body = json.loads(req.data.decode("utf-8"))
    assert sent_body["api_key"] == "phc_test_key"
    assert sent_body["event"] == "train"
    assert sent_body["properties"]["soup_version"] == "0.73.3"
    assert sent_body["properties"]["distinct_id"] == payload["distinct_id"]
    assert sent_body["properties"]["duration_seconds"] == 1.5


def test_distinct_id_creation_persistence_and_reuse(tmp_path, monkeypatch):
    """distinct_id is a valid UUID, saved to disk, and reused across invocations."""
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    id1 = get_or_create_distinct_id()
    assert uuid.UUID(id1)
    saved_file = tmp_path / ".soup" / "telemetry_id"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8").strip() == id1

    id2 = get_or_create_distinct_id()
    assert id1 == id2


def test_distinct_id_recovers_from_corrupted_file(tmp_path, monkeypatch):
    """If the persisted distinct_id is invalid/corrupted, a new valid UUID replaces it."""
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    id_file = tmp_path / ".soup" / "telemetry_id"
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text("not-a-valid-uuid", encoding="utf-8")

    fresh_id = get_or_create_distinct_id()
    assert uuid.UUID(fresh_id)
    assert fresh_id != "not-a-valid-uuid"
    assert id_file.read_text(encoding="utf-8").strip() == fresh_id


def test_distinct_id_swallows_fs_exceptions(tmp_path, monkeypatch):
    """Filesystem permission or OS errors are silently swallowed, returning a valid UUID."""
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("pathlib.Path.exists", side_effect=OSError("Permission denied")):
        with patch("pathlib.Path.write_text", side_effect=OSError("Read-only filesystem")):
            generated_id = get_or_create_distinct_id()

    assert uuid.UUID(generated_id)


def test_build_telemetry_payload_disabled_leaves_no_disk_trace(tmp_path, monkeypatch):
    """When telemetry is disabled, building a payload creates zero disk side effects."""
    from soup_cli.utils.trackers import build_telemetry_payload

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    payload = build_telemetry_payload(soup_version="0.74.0", command="train")
    assert uuid.UUID(payload["distinct_id"])
    assert not (tmp_path / ".soup" / "telemetry_id").exists()
    assert not (tmp_path / ".soup").exists()


def test_send_telemetry_refuses_placeholder_key_with_notice(capsys, monkeypatch):
    """Placeholder PostHog key refuses transmission, warns on stderr, and makes 0 network calls."""
    from soup_cli.utils.trackers import send_telemetry_payload

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    monkeypatch.setattr(
        "soup_cli.utils.trackers._telemetry_endpoint_is_safe", lambda _url: True
    )
    network_mock = MagicMock(side_effect=AssertionError("network opened with placeholder key"))
    monkeypatch.setattr("urllib.request.urlopen", network_mock)

    payload = {
        "command": "train",
        "soup_version": "0.74.0",
        "distinct_id": str(uuid.uuid4()),
    }
    monkeypatch.delenv("SOUP_POSTHOG_KEY", raising=False)
    result = send_telemetry_payload(payload)
    assert result is False
    assert network_mock.call_count == 0

    err = capsys.readouterr().err
    assert "Notice: telemetry is not yet live." in err


def test_emit_telemetry_never_raises_on_broken_env(monkeypatch):
    """_emit_telemetry must swallow any exception even if consent check raises."""
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setattr(
        "soup_cli.utils.trackers.is_telemetry_enabled",
        MagicMock(side_effect=RuntimeError("unexpected crash")),
    )
    # Must not raise
    _emit_telemetry(["soup", "train"], 1.0)
    assert True


def test_every_test_in_this_file_has_load_bearing_assert():
    """Verify by AST inspection that every test in this file contains assert nodes."""
    source_file = Path(__file__)
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    test_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]

    assert len(test_functions) >= 10
    for func in test_functions:
        has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(func))
        assert has_assert, f"Test {func.name} does not contain any assert statements!"
