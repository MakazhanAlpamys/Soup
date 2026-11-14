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

import pytest

import soup_cli.utils.loop_daemon as loop_daemon
from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, watch
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
            raise OSError("killed before loop.yaml was written")

        monkeypatch.setattr(loop_daemon, "merge_state", killed)
        with pytest.raises(OSError):
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
