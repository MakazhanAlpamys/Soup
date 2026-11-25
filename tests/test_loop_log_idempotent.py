"""The loop's log lines are idempotent: a restart does not append a second record.

``run_once`` appends the iteration, version and progress lines and only then does ``watch``
write ``loop.yaml``. A kill in between left the canary active in loop.yaml, so the next cycle
promoted again and wrote a second promote line with the same ``to_version``. Now an iteration
line (same ``iteration_id`` and ``outcome``) and a version line (same ``to_version``, ``cause``
and ``iteration_id``) that are already in the file are not written again; a reader also ignores
a duplicate. The checks run against the real crash: the state write is made to fail once.
"""

from __future__ import annotations

import json
import os

import pytest

import soup_cli.utils.loop_daemon as loop_daemon
import soup_cli.utils.loop_log as loop_log_module
from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, run_once, watch
from soup_cli.utils.loop_log import LoopLog
from soup_cli.utils.loop_state import LoopState, read_state, write_state


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _iteration(iteration_id="iter-20261110T120000-aaaaaaaa", outcome="canary"):
    return {"schema_version": 1, "id": "i-1", "iteration_id": iteration_id, "outcome": outcome}


def _version(to_version=2, cause="promote", iteration_id="iter-1"):
    return {"schema_version": 1, "id": "v-1", "to_version": to_version, "cause": cause,
            "iteration_id": iteration_id}


class TestLogLevel:
    def test_deeply_nested_line_does_not_break_dedup_or_later_appends(self, cwd):
        log = LoopLog(str(cwd / "logs"))
        path = cwd / "logs" / "versions.jsonl"

        assert log.version(_version()) is True
        with path.open("ab") as handle:
            handle.write(b"[" * 10_000 + b"]" * 10_000 + b"\n")

        assert log.version({**_version(), "id": "duplicate"}) is True
        assert log.version(_version(to_version=3)) is True

        valid = []
        for raw in path.read_bytes().splitlines():
            try:
                record = json.loads(raw)
            except (ValueError, RecursionError):
                continue
            if isinstance(record, dict):
                valid.append(record)
        assert [record["to_version"] for record in valid] == [2, 3]

    def test_the_same_iteration_and_outcome_is_written_once(self, cwd):
        log = LoopLog(str(cwd / "logs"))

        assert log.iteration(_iteration()) is True
        assert log.iteration({**_iteration(), "id": "i-2"}) is True

        assert len(_lines(cwd / "logs" / "iterations.jsonl")) == 1

    def test_a_new_outcome_of_the_same_iteration_is_written(self, cwd):
        log = LoopLog(str(cwd / "logs"))

        log.iteration(_iteration(outcome="canary"))
        log.iteration(_iteration(outcome="promoted"))

        outcomes = [line["outcome"] for line in _lines(cwd / "logs" / "iterations.jsonl")]
        assert outcomes == ["canary", "promoted"]

    def test_the_same_version_change_is_written_once(self, cwd):
        log = LoopLog(str(cwd / "logs"))

        log.version(_version())
        log.version({**_version(), "id": "v-2"})
        log.version(_version(to_version=3))

        versions = [line["to_version"] for line in _lines(cwd / "logs" / "versions.jsonl")]
        assert versions == [2, 3]

    def test_a_line_without_the_key_fields_is_always_written(self, cwd):
        log = LoopLog(str(cwd / "logs"))

        log.iteration({"schema_version": 1, "id": "a"})
        log.iteration({"schema_version": 1, "id": "b"})

        assert len(_lines(cwd / "logs" / "iterations.jsonl")) == 2

    def test_a_torn_or_foreign_line_in_the_file_does_not_stop_the_check(self, cwd):
        path = cwd / "logs" / "iterations.jsonl"
        path.parent.mkdir()
        path.write_text('{"half"\nnot json\n[1,2]\n' + json.dumps(_iteration()) + "\n")
        log = LoopLog(str(cwd / "logs"))

        log.iteration(_iteration())

        assert len([line for line in path.read_text().splitlines() if "iter-" in line]) == 1

    def test_progress_lines_are_not_deduplicated(self, cwd):
        log = LoopLog(str(cwd / "logs"))

        log.progress({"schema_version": 1, "id": "p", "iteration_id": "x", "stage": "train"})
        log.progress({"schema_version": 1, "id": "p2", "iteration_id": "x", "stage": "train"})

        assert len(_lines(cwd / "logs" / "progress.jsonl")) == 2

    def test_large_log_is_scanned_once_then_only_new_bytes_are_read(self, cwd, monkeypatch):
        path = cwd / "logs" / "iterations.jsonl"
        path.parent.mkdir()
        target = _iteration("iter-target", "skipped")
        rows = [
            {"schema_version": 1, "iteration_id": f"iter-{i}", "outcome": "skipped"}
            for i in range(20_000)
        ]
        rows.append(target)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        reads = []
        real_read = loop_log_module.os.read

        def record_read(fd, size):
            data = real_read(fd, size)
            reads.append(len(data))
            return data

        monkeypatch.setattr(loop_log_module.os, "read", record_read)
        log = LoopLog(str(path.parent))
        assert log.iteration(target) is True
        assert sum(reads) >= path.stat().st_size

        reads.clear()
        later = _iteration("iter-later", "skipped")
        assert log.iteration(later) is True
        assert log.iteration(later) is True

        assert sum(reads) < 2048
        assert sum(line.get("iteration_id") == "iter-later" for line in _lines(path)) == 1

    def test_incremental_index_observes_an_append_from_another_writer(self, cwd):
        first = LoopLog(str(cwd / "logs"))
        warm = _iteration("iter-warm", "skipped")
        target = _iteration("iter-external", "canary")
        first.iteration(warm)
        assert first.iteration({**warm, "id": "different-id"}) is True

        loop_log_module.append_json_line(str(cwd / "logs" / "iterations.jsonl"), target)
        assert first.iteration(target) is True

        assert sum(line.get("iteration_id") == "iter-external"
                   for line in _lines(cwd / "logs" / "iterations.jsonl")) == 1

    def test_restart_rebuilds_the_idempotency_index_from_existing_logs(self, cwd):
        first = LoopLog(str(cwd / "logs"))
        target = _iteration("iter-restart", "promoted")
        first.iteration(target)

        restarted = LoopLog(str(cwd / "logs"))
        assert restarted.iteration({**target, "id": "different-id"}) is True

        assert len(_lines(cwd / "logs" / "iterations.jsonl")) == 1

    def test_truncating_a_log_invalidates_the_cached_index(self, cwd):
        path = cwd / "logs" / "iterations.jsonl"
        log = LoopLog(str(path.parent))
        target = _iteration("iter-after-truncate", "failed")
        log.iteration(target)
        assert log.iteration({**target, "id": "different-id"}) is True
        path.write_bytes(b"")

        assert log.iteration(target) is True
        assert len(_lines(path)) == 1

    def test_same_size_rewrite_invalidates_the_cached_index(self, cwd):
        path = cwd / "logs" / "iterations.jsonl"
        log = LoopLog(str(path.parent))
        old = _iteration("iter-same-size", "canary")
        replacement = _iteration("iter-same-size", "failed")
        log.iteration(old)
        assert log.iteration({**old, "id": "different-id"}) is True
        old_stat = path.stat()
        old_size = path.stat().st_size
        path.write_bytes((json.dumps(replacement) + "\n").encode("utf-8"))
        rewritten = path.stat()
        os.utime(path, ns=(rewritten.st_atime_ns, old_stat.st_mtime_ns + 2_000_000_000))
        assert path.stat().st_size == old_size
        assert path.stat().st_mtime_ns != old_stat.st_mtime_ns

        assert log.iteration(old) is True
        assert len(_lines(path)) == 2

    def test_replacing_log_inode_invalidates_the_cached_index(self, cwd):
        path = cwd / "logs" / "iterations.jsonl"
        log = LoopLog(str(path.parent))
        target = _iteration("iter-replaced", "canary")
        replacement = _iteration("iter-replaced", "failed")
        log.iteration(target)
        assert log.iteration({**target, "id": "different-id"}) is True

        replacement_path = path.with_name("iterations-replacement.jsonl")
        replacement_path.write_bytes((json.dumps(replacement) + "\n").encode("utf-8"))
        os.replace(replacement_path, path)

        assert log.iteration(target) is True
        rows = _lines(path)
        assert sum(row.get("outcome") == "canary" for row in rows) == 1
        assert sum(row.get("outcome") == "failed" for row in rows) == 1

    def test_changed_bytes_before_old_end_are_detected_when_log_grows(self, cwd):
        path = cwd / "logs" / "iterations.jsonl"
        log = LoopLog(str(path.parent))
        target = _iteration("iter-anchor", "canary")
        replacement = _iteration("iter-anchor", "failed")
        log.iteration(target)
        assert log.iteration({**target, "id": "different-id"}) is True
        original_size = path.stat().st_size

        replacement_bytes = (json.dumps(replacement) + "\n").encode("utf-8")
        assert len(replacement_bytes) == original_size
        path.write_bytes(replacement_bytes)
        loop_log_module.append_json_line(
            str(path), _iteration("iter-after-anchor", "skipped")
        )

        assert log.iteration(target) is True
        rows = _lines(path)
        assert sum(row.get("outcome") == "canary" for row in rows) == 1
        assert sum(row.get("outcome") == "failed" for row in rows) == 1


class TestEveryIterationGetsItsLine:
    """Two iterations that share a parent are two iterations (review 5, F4).

    The dedup key of a line with a ``parent_iteration_id`` used to be (parent, outcome): every
    iteration of a loop whose production did not change shares that parent, so only the first
    ``failed`` (or ``skipped``, or ``canary``) line of a production version was ever written.
    """

    def _line(self, iteration_id, outcome, parent="iter-20261110T120000-seed0000"):
        return {"schema_version": 1, "iteration_id": iteration_id, "outcome": outcome,
                "parent_iteration_id": parent}

    @pytest.mark.parametrize("outcome", ["failed", "skipped", "canary"])
    def test_two_iterations_with_one_parent_are_two_lines(self, cwd, outcome):
        log = LoopLog(str(cwd / "logs"))

        log.iteration(self._line("iter-20261110T120001-aaaaaaaa", outcome))
        log.iteration(self._line("iter-20261110T120002-bbbbbbbb", outcome))

        ids = [line["iteration_id"] for line in _lines(cwd / "logs" / "iterations.jsonl")]
        assert ids == ["iter-20261110T120001-aaaaaaaa", "iter-20261110T120002-bbbbbbbb"]

    def test_the_same_iteration_and_outcome_is_still_written_once(self, cwd):
        log = LoopLog(str(cwd / "logs"))

        log.iteration(self._line("iter-20261110T120001-aaaaaaaa", "failed"))
        log.iteration(self._line("iter-20261110T120001-aaaaaaaa", "failed"))

        assert len(_lines(cwd / "logs" / "iterations.jsonl")) == 1

    def test_six_failed_cycles_leave_six_failed_lines(self, cwd):
        config = WatchConfig(
            poll_interval_sec=1.0, max_iterations=1,
            state_path=str(cwd / ".soup" / "loop.yaml"),
            canary_stats_path=str(cwd / ".soup" / "canary-stats.json"),
            iteration_dir=str(cwd / ".soup-loops"),
            harvest_fn=lambda state: {"pairs_harvested": 3, "traces_collected": 3},
            train_fn=lambda state, h: {
                "run_id": None, "skipped": False, "adapter_path": None, "failed": True,
                "notes": "train failed (exit code 1): the same message",
            },
            gate_fn=lambda state, t: {"gate_verdict": "SKIPPED"},
            deploy_fn=lambda state, o: {"deployed": False},
            cost_fn=lambda state: 0.0,
        )
        state = LoopState(served_model="base", eval_suite="e", baseline="b", status="running")
        log = LoopLog(str(cwd / ".soup-loops"))

        for _ in range(6):
            state, _, _ = run_once(state, config, log=log)

        lines = _lines(cwd / ".soup-loops" / "iterations.jsonl")
        failed = [line for line in lines if line["outcome"] == "failed"]
        assert len(failed) == 6, [line["outcome"] for line in lines]
        assert len({line["iteration_id"] for line in failed}) == 6


class _Killed(BaseException):
    """A process kill. Not an ``Exception``: no handler of the daemon sees it.

    An ``OSError`` from the state write is a transient error the daemon rides out and notes
    (tests/test_loop_daemon_state_errors.py), so it no longer simulates a crash.
    """


class TestACrashBetweenTheLogAndTheState:
    def _config(self, root):
        return WatchConfig(
            poll_interval_sec=1.0, max_iterations=1, state_path=str(root / ".soup" / "loop.yaml"),
            canary_stats_path=str(root / ".soup" / "canary-stats.json"),
            iteration_dir=str(root / ".soup-loops"),
            harvest_fn=lambda state: {"pairs_harvested": 3, "traces_collected": 3},
            train_fn=lambda state, h: {"run_id": "run-1", "skipped": False,
                                       "adapter_path": "out/a1"},
            gate_fn=lambda state, t: {"gate_verdict": "OK"},
            deploy_fn=lambda state, o: {
                "deployed": True, "canary_active": "cand", "canary_adapter": "out/a1",
                "canary_traffic_pct": 5.0, "canary_rollout_id": "r1",
            },
            cost_fn=lambda state: 0.0,
        )

    def test_a_restart_after_a_crash_does_not_promote_twice_in_the_logs(self, cwd, monkeypatch):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(
            LoopState(served_model="base", eval_suite="e", baseline="b", status="running"), path
        )
        config = self._config(cwd)
        watch(config)  # the iteration trains and starts a canary
        assert read_state(path).canary_active == "cand"
        policy = CanaryPolicy(stable="base", canary="cand", traffic_pct=5.0)
        for _ in range(30):  # a healthy canary: the next cycle promotes it
            for bucket in ("stable", "canary"):
                record_bucket_outcome(
                    policy, bucket, True, rollout_id="r1",
                    path=str(cwd / ".soup" / "canary-stats.json"),
                )
        real_merge = loop_daemon.merge_state

        def killed(*args, **kwargs):
            raise _Killed("killed before loop.yaml was written")

        monkeypatch.setattr(loop_daemon, "merge_state", killed)
        with pytest.raises(_Killed):
            watch(config)  # promotes: the lines are appended, loop.yaml is not
        monkeypatch.setattr(loop_daemon, "merge_state", real_merge)
        assert read_state(path).canary_active == "cand", "the crash left the canary active"

        watch(config)  # the restart promotes again

        assert read_state(path).production_name == "cand"
        versions = _lines(cwd / ".soup-loops" / "versions.jsonl")
        promotes = [line for line in versions if line.get("cause") == "promote"]
        assert len(promotes) == 1, versions
        iterations = _lines(cwd / ".soup-loops" / "iterations.jsonl")
        # (the first "promoted" line is the log's own start: the served model as version 1)
        promoted = [
            line for line in iterations
            if line.get("outcome") == "promoted" and line.get("to_version") == 2
        ]
        assert len(promoted) == 1, iterations
