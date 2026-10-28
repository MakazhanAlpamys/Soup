"""Tests for Lambda Cloud controller remote log tail printing on failure (#1705).

Verifies that the rendered Lambda controller:
1. Embeds output redirection via tee under pipefail in user_data.
2. Captures remote log tail over SSH when remote training exits with non-zero status.
3. Passes short timeout, capture_output=True, check=False, and utf-8 replace encoding
   to subprocess.run.
4. Prints the log tail strictly before verifying instance termination.
5. Proceeds with instance termination even if the log tail fetch fails, times out, or errors.
6. Leaves the controller exit status unchanged on fetch errors (rc == 1).
7. Sanitizes untrusted remote output (both stdout and stderr failure notices) by stripping
   ESC, C0, DEL, and C1 control sequences (such as OSC window title sequences) and capping
   failure details at 200 characters.
8. Enforces line cap (at most 50 lines) independently of byte cap.
9. Enforces byte cap (at most 50 KB) independently of line cap.
10. Handles multi-byte characters split across the 50 KB byte-cap boundary without decode error.
11. Prints no log tail on successful runs (exit 0).
"""

from __future__ import annotations

import re
import subprocess
import types
from pathlib import Path

from soup_cli.cloud.lambda_labs import render_lambda_stub

_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: data.jsonl\n  format: chatml\n"
    "output: ./out\n"
)


def _setup_env(tmp_path: Path, monkeypatch) -> str:
    key = tmp_path / "lambda-key"
    key.write_text("test-only", encoding="utf-8")
    monkeypatch.setenv("LAMBDA_API_KEY", "secret")
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "my-key")
    monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
    monkeypatch.setenv("LAMBDA_REGION", "us-tx-1")
    return render_lambda_stub(
        _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.75.1"
    )


def _exec(
    stub: str, fake_run_fn, captured_prints: list[str]
) -> tuple[int, list]:
    namespace: dict[str, object] = {"__name__": "controller"}
    exec(stub, namespace)
    # ssh and scp are taken as found under their own names, so the fake replies can
    # be keyed on argv[0]; tests/test_external_tool_paths.py covers the PATH search.
    namespace["_find_tool"] = lambda name: name

    terminations: list[tuple] = []
    clock = {"now": 0.0}

    def fake_request(api_key, path, *, method="GET", payload=None):
        if path.endswith("/launch"):
            return {"data": {"instance_ids": ["instance-1"]}}
        if path.endswith("/terminate"):
            terminations.append((api_key, path, payload))
            return {"data": {}}
        return {"data": {}}

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    namespace["_request_json"] = fake_request
    namespace["_wait_for_ip"] = lambda *_: "192.0.2.1"
    namespace["_wait_for_termination"] = lambda *_: None
    namespace["time"] = types.SimpleNamespace(
        sleep=fake_sleep,
        monotonic=lambda: clock["now"],
    )
    namespace["print"] = lambda *args, **kw: captured_prints.append(
        " ".join(str(a) for a in args)
    )
    namespace["subprocess"] = types.SimpleNamespace(
        run=fake_run_fn,
        TimeoutExpired=subprocess.TimeoutExpired,
    )

    rc = namespace["main"]()
    return rc, terminations


class TestLambdaLogTailOnFailure:
    def test_user_data_pipes_train_log_under_pipefail(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        namespace: dict[str, object] = {}
        exec(stub, namespace)
        user_data = str(namespace["_USER_DATA"])

        pipefail_stmt = "set -euo pipefail"
        tee_stmt = "soup train --config soup.yaml --yes 2>&1 | tee /home/ubuntu/soup/train.log"

        assert pipefail_stmt in user_data
        assert tee_stmt in user_data
        assert user_data.index(pipefail_stmt) < user_data.index(tee_stmt)

    def test_remote_exit_1_prints_log_tail_before_instance_termination(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        tail_calls: list[tuple[list[str], dict]] = []

        sample_tail = (
            "Traceback (most recent call last):\n"
            "  File 'train.py', line 42, in <module>\n"
            "FileNotFoundError: data.jsonl not found\n"
        )

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                cmd_str = argv[-1]
                if "tail" in cmd_str:
                    tail_calls.append((argv, kw))
                    return subprocess.CompletedProcess(argv, 0, sample_tail, "")
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 1
        assert len(terminations) == 1
        assert terminations[0][2] == {"instance_ids": ["instance-1"]}

        # Verify tail command arguments & options
        assert len(tail_calls) == 1
        tail_argv, tail_kw = tail_calls[0]
        assert tail_kw.get("timeout") == 15
        assert tail_kw.get("encoding") == "utf-8"
        assert tail_kw.get("errors") == "replace"
        assert tail_kw.get("capture_output") is True
        assert tail_kw.get("check") is False

        # Derive expected log path from user_data so the tee destination and tail path cannot drift
        namespace: dict[str, object] = {}
        exec(stub, namespace)
        user_data = str(namespace["_USER_DATA"])
        match = re.search(r"\|\s*tee\s+(\S+)", user_data)
        assert match is not None
        tee_path = match.group(1)

        key_path = str(tmp_path / "lambda-key")
        assert tail_argv == [
            "ssh",
            "-i",
            key_path,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            "ubuntu@192.0.2.1",
            f"tail -n 50 {tee_path}",
        ]

        # Verify log tail is in printed output
        assert any("FileNotFoundError: data.jsonl not found" in p for p in prints)

        # Verify log tail is printed strictly BEFORE instance termination verified
        tail_idx = next(
            i for i, p in enumerate(prints) if "FileNotFoundError: data.jsonl not found" in p
        )
        term_idx = next(
            i for i, p in enumerate(prints) if "Lambda instance termination verified" in p
        )
        assert tail_idx < term_idx

    def test_stubbed_fetch_timeout_still_terminates_instance_with_exit_1(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    raise subprocess.TimeoutExpired(argv, 15)
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 1
        assert len(terminations) == 1
        notices = [p for p in prints if "Could not fetch remote log tail" in p]
        assert len(notices) == 1
        assert any("Lambda instance termination verified" in p for p in prints)

    def test_stubbed_fetch_ssh_stderr_strips_control_bytes_and_caps_200(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        raw_stderr = "\x1b]0;PWNED\x07\x1b[2Jremote failure: " + ("x" * 300)

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(argv, 255, "", raw_stderr)
                return subprocess.CompletedProcess(argv, 0, "status: error\n2\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 1
        assert len(terminations) == 1
        notices = [p for p in prints if "Could not fetch remote log tail" in p]
        assert len(notices) == 1
        notice = notices[0]

        # Verify control bytes stripped from stderr notice
        assert "\x1b" not in notice
        assert "\x07" not in notice
        assert "[2Jremote failure:" in notice

        # Prefix is 'Could not fetch remote log tail: ' (32 chars) + at most 200 detail chars
        detail = notice.replace("Could not fetch remote log tail: ", "")
        assert len(detail) <= 200
        assert any("Lambda instance termination verified" in p for p in prints)

    def test_stubbed_fetch_oserror_still_terminates_instance_with_exit_1(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        raw_err = "\x1b[31mssh binary execution error\x1b[0m: " + ("y" * 300)

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    raise OSError(raw_err)
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 1
        assert len(terminations) == 1
        notices = [p for p in prints if "Could not fetch remote log tail" in p]
        assert len(notices) == 1
        notice = notices[0]

        # Verify control bytes stripped and length capped
        assert "\x1b" not in notice
        assert "ssh binary execution error" in notice
        detail = notice.replace("Could not fetch remote log tail: ", "")
        assert len(detail) <= 200
        assert any("Lambda instance termination verified" in p for p in prints)

    def test_log_tail_strips_control_bytes_and_osc_title_escape(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        raw_log = (
            "\x1b]0;Attacker Title\x07"
            "Normal error line with \ttab\n"
            "\x1b[31mRed text\x1b[0m\x7f\x80\x9f\n"
        )

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(argv, 0, raw_log, "")
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, _ = _exec(stub, fake_run, captured_prints=prints)
        assert rc == 1

        tail_block = next(p for p in prints if "Remote log tail" in p)

        # Control bytes must be completely stripped
        assert "\x1b" not in tail_block
        assert "\x07" not in tail_block
        assert "\x7f" not in tail_block
        assert "\x80" not in tail_block
        assert "\x9f" not in tail_block

        # Legitimate content and allowed controls (tab, newline) preserved
        assert "Attacker TitleNormal error line with \ttab" in tail_block
        assert "[31mRed text[0m" in tail_block

    def test_log_tail_enforces_line_cap_with_more_than_50_short_lines(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        # 100 short lines (~15 bytes each, total ~1,500 bytes << 50 KB byte cap)
        short_lines = [f"short line {i:03d}" for i in range(100)]
        log_text = "\n".join(short_lines) + "\n"

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(argv, 0, log_text, "")
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, _ = _exec(stub, fake_run, captured_prints=prints)
        assert rc == 1

        tail_block = next(p for p in prints if "Remote log tail" in p)
        content_lines = [line for line in tail_block.splitlines() if line.startswith("short line")]

        # Exactly 50 lines kept
        assert len(content_lines) == 50
        # Newest line is present
        assert "short line 099" in content_lines
        # Oldest line (000) was trimmed
        assert "short line 000" not in content_lines
        # 50th line from end is the earliest kept
        assert "short line 050" in content_lines
        assert "short line 049" not in content_lines

    def test_log_tail_enforces_byte_cap_with_few_very_long_lines(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        # 3 lines of 30,000 bytes each (total 90,000 bytes > 50,000 bytes cap, but only 3 lines)
        line_0 = "oldest line: " + ("a" * 29980)
        line_1 = "middle line: " + ("b" * 29980)
        line_2 = "newest line: " + ("c" * 29980)
        oversized_log = f"{line_0}\n{line_1}\n{line_2}\n"

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(argv, 0, oversized_log, "")
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, _ = _exec(stub, fake_run, captured_prints=prints)
        assert rc == 1

        tail_block = next(p for p in prints if "Remote log tail" in p)
        # Content after the header must be within 50,000 bytes
        content_part = tail_block.split("\n", 1)[1]
        assert len(content_part.encode("utf-8")) <= 50_000

        # Newest line is kept
        assert "newest line:" in tail_block
        # Oldest line is trimmed away by byte cap
        assert "oldest line:" not in tail_block

    def test_log_tail_byte_cut_splits_multibyte_utf8_char_without_decode_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        # 3-byte UTF-8 character (Euro sign €: b'\xe2\x82\xac') placed so the
        # 50,000-byte tail cut slices into it, leaving an orphan continuation byte at index 0.
        prefix = "start of remote log\n"
        char_3b = "\u20ac"  # 3 bytes: \xe2 \x82 \xac

        # Suffix of 49,999 bytes: the cut at -50,000 bytes lands on char_3b's 3rd byte (0xac).
        final_marker = "UniqueFatalError: out of bounds\n"
        padding = "z" * (49_999 - len(final_marker.encode("utf-8")))
        suffix = padding + final_marker

        raw_log = prefix + char_3b + suffix

        encoded = raw_log.encode("utf-8")
        assert len(encoded) > 50_000
        cut = encoded[-50_000:]
        assert len(cut) == 50_000
        assert cut[0] == 0xAC  # UTF-8 continuation byte

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(argv, 0, raw_log, "")
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, _ = _exec(stub, fake_run, captured_prints=prints)
        assert rc == 1

        tail_block = next(p for p in prints if "Remote log tail" in p)
        assert "UniqueFatalError: out of bounds" in tail_block
        assert not any("Could not fetch remote log tail" in p for p in prints)

    def test_successful_run_prints_no_log_tail(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        commands: list[list[str]] = []

        def fake_run(argv, **kw):
            commands.append(argv)
            if argv[0] == "ssh":
                return subprocess.CompletedProcess(argv, 0, "status: done\n0\n", "")
            if argv[0] == "scp":
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 0
        assert len(terminations) == 1
        assert not any(cmd[0] == "ssh" and "tail" in cmd[-1] for cmd in commands)
        assert any(cmd[0] == "scp" for cmd in commands)
        assert not any("Remote log tail" in p for p in prints)
        assert not any("Could not fetch remote log tail" in p for p in prints)
        assert any("Training completed" in p for p in prints)
