"""How the loop starts its children and names how one ended (review 5, F2).

``soup loop watch --detach`` on Windows: the ``soup train`` child died with exit 3221225786
(``0xC000013A``, ``STATUS_CONTROL_C_EXIT``) and nothing in its log but ``Loading config from
...``, while the same command in the foreground ran. The detached daemon has no console, so a
console program it starts gets a NEW console with a WINDOW of its own; that is the one place a
console control event can reach it. Every child of the loop (train, ship, the VRAM probe, the
tokenizer probe) is now started with ``CREATE_NO_WINDOW`` on Windows; POSIX is unchanged. A child
that ends by a signal or an NTSTATUS is reported by name and code in ``last_failure``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

import soup_cli.utils.loop_quantization as lq
import soup_cli.utils.loop_spawn as spawn
import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_detach import detached_popen_kwargs
from soup_cli.utils.loop_state import LoopState


class TestFlags:
    def test_windows_children_get_no_window(self):
        assert spawn.child_popen_kwargs("nt") == {"creationflags": 0x08000000}
        assert spawn.CREATE_NO_WINDOW == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    def test_posix_children_get_nothing_added(self):
        assert spawn.child_popen_kwargs("posix") == {}

    def test_the_default_is_this_systems(self):
        assert spawn.child_popen_kwargs() == spawn.child_popen_kwargs(os.name)

    def test_the_detached_daemon_itself_keeps_its_flags(self):
        flags = detached_popen_kwargs("nt")["creationflags"]
        assert flags & 0x00000008 and flags & 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        assert detached_popen_kwargs("posix") == {
            "stdin": subprocess.DEVNULL, "start_new_session": True,
        }


class TestDescribeExit:
    @pytest.mark.parametrize("code", [0, 1, 2, 3, 255])
    def test_an_ordinary_code_is_just_the_code(self, code):
        assert spawn.describe_exit(code, "nt") == f"exit code {code}"
        assert spawn.describe_exit(code, "posix") == f"exit code {code}"

    def test_a_console_control_exit_is_named_with_its_hex_code(self):
        text = spawn.describe_exit(3221225786)

        assert text.startswith("exit code 3221225786 (STATUS_CONTROL_C_EXIT 0xC000013A: ")
        assert "console Ctrl+C, Ctrl+Break or close event" in text

    def test_the_signed_reading_of_the_same_status_is_the_same_name(self):
        assert "STATUS_CONTROL_C_EXIT 0xC000013A" in spawn.describe_exit(3221225786 - 2**32)

    @pytest.mark.parametrize(
        ("code", "name"),
        [
            (0xC0000005, "STATUS_ACCESS_VIOLATION"),
            (0xC0000409, "STATUS_STACK_BUFFER_OVERRUN"),
            (0xC0000135, "STATUS_DLL_NOT_FOUND"),
            (0xC0000374, "STATUS_HEAP_CORRUPTION"),
        ],
    )
    def test_the_common_statuses_have_names(self, code, name):
        assert f"{name} 0x{code:08X}" in spawn.describe_exit(code)

    def test_an_unknown_status_still_shows_its_hex_code(self):
        assert spawn.describe_exit(0xC0000999) == f"exit code {0xC0000999} (NTSTATUS 0xC0000999)"

    def test_a_posix_signal_is_named(self):
        assert spawn.describe_exit(-9, "posix") == "exit code -9 (killed by signal SIGKILL)"
        assert spawn.describe_exit(-15, "posix") == "exit code -15 (killed by signal SIGTERM)"

    def test_an_unknown_signal_number_is_still_shown(self):
        assert spawn.describe_exit(-60, "posix") == "exit code -60 (killed by signal 60)"

    def test_a_small_negative_code_on_windows_is_not_a_signal(self):
        assert spawn.describe_exit(-1, "nt") == "exit code -1"

    @pytest.mark.parametrize("value", [None, "1", 1.0, True, False, [1]])
    def test_anything_that_is_not_an_int_has_no_exit_code(self, value):
        assert spawn.describe_exit(value) == "no exit code"


class _Child:
    def __init__(self, returncode):
        self.returncode = returncode
        self.kwargs = {}

    def __call__(self, argv, **kwargs):
        self.kwargs = kwargs
        kwargs["stdout"].write(b"Loading config from x.yaml...\n")
        return type("R", (), {"returncode": self.returncode, "stderr": None})()


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    row = {"prompt": "q", "chosen": "a", "rejected": "b"}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return tmp_path


def _state():
    return LoopState(served_model="org/base", eval_suite="e", baseline="b", base_model="org/base")


class TestEveryChildGetsTheFlags:
    FLAGS = {"creationflags": 0x08000000}

    def test_the_train_child(self, cwd, monkeypatch):
        monkeypatch.setattr(ls, "child_popen_kwargs", lambda: dict(self.FLAGS))
        child = _Child(0)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)

        ls.train_dpo_from_pairs(_state(), {"pairs_path": "pairs.jsonl", "pairs_harvested": 1})

        assert child.kwargs["creationflags"] == 0x08000000

    def test_the_ship_child(self, cwd, monkeypatch):
        monkeypatch.setattr(ls, "child_popen_kwargs", lambda: dict(self.FLAGS))
        child = _Child(2)
        monkeypatch.setattr(ls, "_SHIP_RUNNER", child)
        adapter = cwd / "adapter"
        adapter.mkdir()

        ls.verify_with_ship(
            _state(), {"adapter_path": str(adapter), "run_id": "r1", "skipped": False}
        )

        assert child.kwargs["creationflags"] == 0x08000000

    def test_the_vram_probe(self, monkeypatch):
        seen = {}

        def run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, '{"free": null}\n', "")

        monkeypatch.setattr(lq, "child_popen_kwargs", lambda: dict(self.FLAGS))
        monkeypatch.setattr(subprocess, "run", run)

        REAL_FREE_VRAM()

        assert seen["creationflags"] == 0x08000000


#: The real function, taken before the suite's autouse stub replaces it for a test.
REAL_FREE_VRAM = lq.free_vram_bytes


class TestFailureIsNamed:
    def test_a_train_child_that_ended_by_a_console_event_says_so(self, cwd, monkeypatch):
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", _Child(3221225786))

        out = ls.train_dpo_from_pairs(
            _state(), {"pairs_path": "pairs.jsonl", "pairs_harvested": 1}
        )

        assert out["failed"] is True
        assert (
            "train failed (exit code 3221225786 (STATUS_CONTROL_C_EXIT 0xC000013A: " in out["notes"]
        )
        assert "Loading config from x.yaml..." in out["notes"]  # the log's tail follows

    def test_the_status_row_carries_the_name_not_a_bare_number(self, cwd, monkeypatch):
        from soup_cli.utils.loop_daemon import WatchConfig, run_once

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", _Child(3221225786))
        config = WatchConfig(
            poll_interval_sec=1.0, max_iterations=1,
            state_path=str(cwd / ".soup" / "loop.yaml"),
            canary_stats_path=str(cwd / ".soup" / "canary-stats.json"),
            iteration_dir=str(cwd / ".soup-loops"),
            harvest_fn=lambda state: {
                "pairs_harvested": 1, "traces_collected": 1, "pairs_path": "pairs.jsonl",
                "new_pairs": 1,
            },
            train_fn=ls.train_dpo_from_pairs,
            gate_fn=lambda state, ctx: {"gate_verdict": "SKIPPED"},
            deploy_fn=lambda state, ctx: {"deployed": False},
            cost_fn=lambda state: 0.0,
        )

        state, _, _ = run_once(_state(), config)

        assert "train failed (exit code 3221225786 (STATUS_CONTROL_C_EXIT 0xC000013A" in (
            state.last_failure
        )

    def test_a_ship_child_that_died_is_named_too(self, cwd, monkeypatch):
        monkeypatch.setattr(ls, "_SHIP_RUNNER", _Child(3221225786))
        adapter = cwd / "adapter"
        adapter.mkdir()

        out = ls.verify_with_ship(
            _state(), {"adapter_path": str(adapter), "run_id": "r1", "skipped": False}
        )

        assert out["gate_verdict"] == "SKIPPED"
        assert "STATUS_CONTROL_C_EXIT 0xC000013A" in out["notes"]


_STUB_CHILD = textwrap.dedent(
    """
    import ctypes, os, sys, time
    marker, seconds = sys.argv[1], float(sys.argv[2])
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    window = int(kernel.GetConsoleWindow() or 0)
    time.sleep(seconds)
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(str(window))
    print("stub child done", flush=True)
    """
)

_MIDDLE = textwrap.dedent(
    """
    import subprocess, sys
    import soup_cli.utils.loop_stages as ls
    stub, marker, log, result = sys.argv[1:5]
    done = ls._run_logged(
        subprocess.run, [sys.executable, stub, marker, "1.5"], log_path=log, timeout=120
    )
    with open(result, "w", encoding="utf-8") as handle:
        handle.write(str(done.returncode))
    """
)


@pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows matter")
class TestADetachedDaemonRunsItsChildToTheEnd:
    """The real spawn helper under the daemon's own flags, with a stub child (no GPU, no soup)."""

    def test_a_stub_child_of_a_detached_process_runs_with_no_console_window(self, tmp_path):
        stub = tmp_path / "stub_child.py"
        stub.write_text(_STUB_CHILD, encoding="utf-8")
        marker, log, result = tmp_path / "marker", tmp_path / "child.log", tmp_path / "rc"
        middle = subprocess.Popen(  # the daemon: no console, its own process group
            [sys.executable, "-c", _MIDDLE, str(stub), str(marker), str(log), str(result)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **detached_popen_kwargs("nt"),
        )
        deadline = time.monotonic() + 120
        while not result.exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        middle.wait(timeout=30)

        assert result.exists(), "the detached daemon stand-in wrote no result"
        assert result.read_text(encoding="utf-8") == "0"
        assert marker.exists(), "the stub child did not run to the end"
        assert marker.read_text(encoding="utf-8") == "0", "the child got a console WINDOW"
        assert "stub child done" in log.read_text(encoding="utf-8")  # its output is in the log
