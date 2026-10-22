"""Platform selection and useful failures for the real two-rank Gloo checks."""

import runpy
import subprocess
import sys
from unittest.mock import Mock

import pytest

from tests import test_distributed_base_revision as ranks


@pytest.mark.parametrize("platform,skipped", [("linux", False), ("darwin", True), ("win32", True)])
def test_only_real_gloo_cases_are_limited_to_linux(monkeypatch, platform, skipped):
    monkeypatch.setattr(sys, "platform", platform)
    namespace = runpy.run_path(ranks.__file__)
    real = namespace["test_two_real_cpu_processes_share_identity_and_receipts"]
    gates = [mark for mark in real.pytestmark if mark.name == "skipif"]
    assert len(gates) == 1
    assert gates[0].args == (skipped,)
    assert "Linux" in gates[0].kwargs["reason"]
    assert "Gloo" in gates[0].kwargs["reason"]
    assert not namespace.get("pytestmark"), "simulated ranks must run on every platform"
    for name, value in namespace.items():
        if name.startswith("test_") and value is not real:
            assert not any(mark.name in {"skip", "skipif"}
                           for mark in getattr(value, "pytestmark", ())), name


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_rank_failure_reports_both_logs_and_reaps_the_children(tmp_path, rank, failure):
    logs = [tmp_path / f"rank-{index}.log" for index in range(2)]
    for index, path in enumerate(logs):
        path.write_text(f"rank {index} diagnostic", encoding="utf-8")
    processes = [Mock(), Mock()]
    for index, process in enumerate(processes):
        process.poll.return_value = 0 if index < rank else None
        process.wait.return_value = 0
    if failure == "timeout":
        processes[rank].wait.side_effect = [subprocess.TimeoutExpired("worker", 75), -9]
    else:
        processes[rank].wait.return_value = 7
        processes[rank].poll.return_value = 7
    with pytest.raises(pytest.fail.Exception) as error:
        ranks._wait_for_ranks(processes, logs)
    message = str(error.value)
    assert f"rank {rank}" in message
    assert ("timed out" if failure == "timeout" else "exited with 7") in message
    for index in range(2):
        assert f"rank {index} diagnostic" in message
        process = processes[index]
        if process.poll.return_value is None:
            process.kill.assert_called_once_with()
            assert process.wait.call_args.kwargs == {"timeout": 10}
        else:
            process.kill.assert_not_called()


def test_successful_ranks_are_waited_for_without_being_killed(tmp_path):
    processes = [Mock(), Mock()]
    for process in processes:
        process.wait.return_value = process.poll.return_value = 0
    # Successful children need no diagnostic reads.
    ranks._wait_for_ranks(processes, [tmp_path / "absent-0", tmp_path / "absent-1"])
    for process in processes:
        process.wait.assert_called_once_with(timeout=75)
        process.kill.assert_not_called()
