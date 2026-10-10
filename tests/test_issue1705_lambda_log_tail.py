"""Tests for Lambda Cloud controller remote log tail printing on failure (#1705).

Verifies that the rendered Lambda controller:
1. Captures remote log tail over SSH when remote training exits with non-zero status.
2. Prints the log tail strictly before verifying instance termination.
3. Proceeds with instance termination even if the log tail fetch fails, times out, or errors.
4. Leaves the controller exit status unchanged on fetch errors (rc == 1).
5. Sanitizes untrusted remote output by stripping ESC, C0, DEL, and C1 control sequences
   (such as OSC window title sequences) while preserving tab and newline.
6. Enforces bounds: at most 50 lines and 50 KB byte cap.
7. Prints no log tail on successful runs (exit 0).
"""

from __future__ import annotations

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
    def test_remote_exit_1_prints_log_tail_before_instance_termination(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        commands: list[list[str]] = []

        sample_tail = (
            "Traceback (most recent call last):\n"
            "  File 'train.py', line 42, in <module>\n"
            "FileNotFoundError: data.jsonl not found\n"
        )

        def fake_run(argv, **kw):
            commands.append(argv)
            if argv[0] == "ssh":
                # Check whether this is poll or log tail
                cmd_str = argv[-1]
                if "tail" in cmd_str:
                    return subprocess.CompletedProcess(argv, 0, sample_tail, "")
                # status poll
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 1
        assert len(terminations) == 1
        assert terminations[0][2] == {"instance_ids": ["instance-1"]}

        # Verify tail command was called
        tail_cmds = [cmd for cmd in commands if cmd[0] == "ssh" and "tail" in cmd[-1]]
        assert len(tail_cmds) == 1
        assert "tail -n 50 /home/ubuntu/soup/train.log" in tail_cmds[0][-1]

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
        # Printed failure notice is concise
        notices = [p for p in prints if "Could not fetch remote log tail" in p]
        assert len(notices) == 1
        # Termination was still verified
        assert any("Lambda instance termination verified" in p for p in prints)

    def test_stubbed_fetch_ssh_error_still_terminates_instance_with_exit_1(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(
                        argv, 255, "", "Connection to 192.0.2.1 closed by remote host."
                    )
                return subprocess.CompletedProcess(argv, 0, "status: error\n2\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec(stub, fake_run, captured_prints=prints)

        assert rc == 1
        assert len(terminations) == 1
        notices = [p for p in prints if "Could not fetch remote log tail" in p]
        assert len(notices) == 1
        assert "Connection to 192.0.2.1 closed by remote host." in notices[0]
        assert any("Lambda instance termination verified" in p for p in prints)

    def test_log_tail_strips_control_bytes_and_osc_title_escape(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        # OSC window title escape sequence + ANSI color codes + C0/C1 control bytes
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

    def test_log_tail_enforces_line_and_byte_caps(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []

        # 100 lines, each 1000 characters (total 100,000 bytes, exceeds 50 KB cap)
        lines_100 = [f"line-{i:03d} " + ("x" * 980) for i in range(100)]
        oversized_log = "\n".join(lines_100) + "\n"

        def fake_run(argv, **kw):
            if argv[0] == "ssh":
                if "tail" in argv[-1]:
                    return subprocess.CompletedProcess(argv, 0, oversized_log, "")
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, _ = _exec(stub, fake_run, captured_prints=prints)
        assert rc == 1

        tail_block = next(p for p in prints if "Remote log tail" in p)
        # Byte cap check: printed block must be bounded
        assert len(tail_block.encode("utf-8")) <= 60_000

        # Line cap check: at most 50 log lines
        log_lines = tail_block.splitlines()
        # First line is the header 'Remote log tail (last 50 lines):'
        content_lines = [line for line in log_lines if line.startswith("line-")]
        assert len(content_lines) <= 50
        # The newest line must be present (line-099)
        assert any("line-099" in line for line in content_lines)
        # Older lines (e.g. line-000) must be trimmed
        assert not any("line-000" in line for line in content_lines)

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
        # No tail command sent
        assert not any(cmd[0] == "ssh" and "tail" in cmd[-1] for cmd in commands)
        # SCP was run
        assert any(cmd[0] == "scp" for cmd in commands)
        # No log tail or fetch notice printed
        assert not any("Remote log tail" in p for p in prints)
        assert not any("Could not fetch remote log tail" in p for p in prints)
        assert any("Training completed" in p for p in prints)
