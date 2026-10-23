"""Rank-zero base identity coordination, including real CPU process groups."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.utils import run_provenance as provenance

BASE = "owner/base"
SHA = "a" * 40
linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason=(
        "real two-process CPU/Gloo checks run on Linux only; macOS env:// rendezvous "
        "can time out on loaded runners. Simulated-rank checks run on every platform."
    ),
)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("source", ["hub_commit", "offline", "resolution_failed"])
def test_each_rank_uses_only_the_coordinators_nullable_result(monkeypatch, rank, source):
    import accelerate
    import accelerate.utils

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    if source == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    probe = Mock(return_value=SimpleNamespace(sha=SHA))
    if source == "resolution_failed":
        probe.side_effect = OSError("private provider diagnostic")
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=probe))
    # Two one-process nodes: each is its node's local main process, but only
    # global rank zero may resolve. A local-main check would resolve twice.
    state = SimpleNamespace(num_processes=2, is_main_process=rank == 0, is_local_main_process=True)
    factory = Mock(return_value=state)
    monkeypatch.setattr(accelerate, "PartialState", factory)
    payload = {"schema_version": 1, "run_id": "coordinator", "base_model": BASE,
               "base_revision": SHA if source == "hub_commit" else None,
               "base_revision_source": source, "training_data_sha256": None}

    def broadcast(objects, from_process):
        assert from_process == 0
        assert objects == ([payload] if rank == 0 else [None])
        objects[0] = payload

    send = Mock(side_effect=broadcast)
    monkeypatch.setattr(accelerate.utils, "broadcast_object_list", send)
    identity = provenance.BaseRevision(BASE)
    provenance.synchronize_base_revision(identity, cpu=True)
    expected = {"revision": SHA} if source == "hub_commit" else {}
    assert identity.load_kwargs(BASE) == expected
    assert identity.load_kwargs(BASE) == expected
    assert identity.load_kwargs("other/teacher") == {}
    assert identity.source == source
    assert probe.call_count == (1 if rank == 0 and source != "offline" else 0)
    assert factory.call_args.kwargs["cpu"] is True
    assert factory.call_args.kwargs["timeout"].total_seconds() == 1800
    send.assert_called_once()


@pytest.mark.parametrize("cpu", [False, True])
def test_initial_group_timeout_is_not_shorter_than_trainers_collective_default(monkeypatch, cpu):
    import accelerate
    from transformers import TrainingArguments

    monkeypatch.setenv("WORLD_SIZE", "2")
    factory = Mock(return_value=SimpleNamespace(num_processes=2))
    monkeypatch.setattr(accelerate, "PartialState", factory)
    provenance._distributed_state(provenance.BaseRevision(BASE), cpu=cpu)
    # PartialState shares the first group's timeout with Trainer. Compare with
    # Trainer's actual default, not the constant being tested in Soup.
    trainer_seconds = TrainingArguments.__dataclass_fields__["ddp_timeout"].default
    assert factory.call_args.kwargs["timeout"].total_seconds() >= trainer_seconds
    assert factory.call_args.kwargs["cpu"] is cpu


def test_single_process_stays_lazy_and_unsupported_backends_stay_unbound(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    probe = Mock(side_effect=AssertionError("unexpected metadata request"))
    monkeypatch.setattr(provenance, "get_hf_api", probe)
    identity = provenance.BaseRevision(BASE)
    provenance.synchronize_base_revision(identity)
    assert identity._pending is True
    unsupported = provenance.BaseRevision(BASE, binding_supported=False)
    provenance.synchronize_base_revision(unsupported)
    assert unsupported.load_kwargs(BASE) == {}
    probe.assert_not_called()


@pytest.mark.parametrize("bad", [None, {"schema_version": 9}, {
    "schema_version": 1, "run_id": "coordinator", "base_model": "different/base",
    "base_revision": SHA, "base_revision_source": "hub_commit",
}])
def test_bad_broadcast_cannot_fall_back_to_independent_resolution(monkeypatch, bad):
    import accelerate
    import accelerate.utils

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(accelerate, "PartialState", lambda **kwargs: SimpleNamespace(
        num_processes=2, is_main_process=False,
    ))
    monkeypatch.setattr(accelerate.utils, "broadcast_object_list",
                        lambda values, **kwargs: values.__setitem__(0, bad))
    probe = Mock(side_effect=AssertionError("must not resolve on a worker"))
    monkeypatch.setattr(provenance, "get_hf_api", probe)
    with pytest.raises(ValueError):
        provenance.synchronize_base_revision(provenance.BaseRevision(BASE), cpu=True)
    probe.assert_not_called()


def test_missing_process_group_does_not_claim_independent_bindings(monkeypatch):
    import accelerate

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(accelerate, "PartialState", lambda **kw: SimpleNamespace(num_processes=1))
    with pytest.raises(RuntimeError, match="process group"):
        provenance.synchronize_base_revision(provenance.BaseRevision(BASE), cpu=True)


@pytest.mark.parametrize("lost", [False, True])
def test_a_lost_receipt_on_any_rank_downgrades_every_participant(monkeypatch, lost):
    import accelerate
    import accelerate.utils

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(accelerate, "PartialState", lambda **kw: SimpleNamespace(num_processes=2))
    identity = provenance.BaseRevision(BASE)
    identity.revision, identity.source, identity._pending = SHA, "hub_commit", False
    record = identity.metadata("coordinator")
    other = {**record, "base_revision": None, "base_revision_source": "unsupported_backend"}
    monkeypatch.setattr(accelerate.utils, "gather_object",
                        lambda values: [*values, other if lost else record])
    provenance.confirm_distributed_base_revision(identity)
    assert identity.revision == (None if lost else SHA)
    assert identity.source == ("unsupported_backend" if lost else "hub_commit")


def test_collective_failure_is_not_reinterpreted_as_offline(monkeypatch):
    import accelerate
    import accelerate.utils

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(accelerate, "PartialState", lambda **kw: SimpleNamespace(
        num_processes=2, is_main_process=False,
    ))
    monkeypatch.setattr(accelerate.utils, "broadcast_object_list",
                        Mock(side_effect=RuntimeError("collective failed")))
    with pytest.raises(RuntimeError, match="collective failed"):
        provenance.synchronize_base_revision(provenance.BaseRevision(BASE), cpu=True)


WORKER = '''
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch.distributed as distributed
from soup_cli.utils import run_provenance as provenance

root, rank = Path(sys.argv[1]), int(sys.argv[2])
mode, initialize = sys.argv[3], sys.argv[4] == "True"
calls = []
def model_info(base, timeout):
    calls.append(base)
    if mode == "failed":
        raise OSError("secret should never be broadcast")
    return SimpleNamespace(sha=("a" if rank == 0 else "b") * 40)
provenance.get_hf_api = lambda: SimpleNamespace(model_info=model_info)
if mode == "offline" and rank == 0:
    os.environ["HF_HUB_OFFLINE"] = "1"
if initialize:
    distributed.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                                  rank=rank, world_size=2, timeout=timedelta(seconds=45))
try:
    identity = provenance.BaseRevision("owner/base")
    provenance.synchronize_base_revision(identity, cpu=True)
    with provenance.base_revision_scope(identity):
        receivers = [provenance.base_revision_kwargs("owner/base") for _ in range(3)]
        assert provenance.base_revision_kwargs("owner/teacher") == {}
    model = SimpleNamespace(config=SimpleNamespace(
        _commit_hash="b" * 40 if mode == "lost" and rank == 1 else None))
    provenance.confirm_distributed_base_revision(identity, model=model)
    provenance.write_run_provenance(root / f"artifact-{rank}", identity.metadata("run"))
    (root / f"rank-{rank}.json").write_text(json.dumps({
        "metadata": provenance.read_run_provenance(root / f"artifact-{rank}"),
        "receivers": receivers, "calls": calls,
    }), encoding="utf-8")
finally:
    if distributed.is_initialized():
        distributed.destroy_process_group()
'''


def _wait_for_ranks(processes: list[subprocess.Popen], log_paths: list[Path]) -> None:
    """Reap every child before reporting a failed rank and both ranks' diagnostics."""
    failure = None
    try:
        for rank, process in enumerate(processes):
            try:
                returncode = process.wait(timeout=75)
            except subprocess.TimeoutExpired:
                failure = f"rank {rank} timed out after 75 seconds"
                break
            if returncode != 0:
                failure = f"rank {rank} exited with {returncode}"
                break
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    if failure is not None:
        diagnostics = "\n".join(
            f"--- rank {rank} ---\n{path.read_text(encoding='utf-8', errors='replace')}"
            for rank, path in enumerate(log_paths)
        )
        pytest.fail(f"{failure}\n{diagnostics}", pytrace=False)


@linux_only
@pytest.mark.parametrize("mode", ["bound", "offline", "failed", "lost"])
@pytest.mark.parametrize("initialize", [False, True])
def test_two_real_cpu_processes_share_identity_and_receipts(tmp_path, mode, initialize):
    distributed = pytest.importorskip("torch.distributed")
    if not distributed.is_gloo_available():
        pytest.skip("installed torch has no CPU Gloo backend")
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    processes, logs, log_paths = [], [], []
    try:
        for rank in range(2):
            env = os.environ.copy()
            env.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                       MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), ACCELERATE_USE_CPU="true",
                       USE_LIBUV="0", OMP_NUM_THREADS="1")
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "ACCELERATE_USE_FSDP",
                         "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_MIXED_PRECISION"):
                env.pop(name, None)
            log_path = tmp_path / f"rank-{rank}.log"
            log = log_path.open("w", encoding="utf-8")
            logs.append(log)
            log_paths.append(log_path)
            processes.append(subprocess.Popen(
                [sys.executable, "-u", str(worker), str(tmp_path),
                 str(rank), mode, str(initialize)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            ))
        _wait_for_ranks(processes, log_paths)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert rows[0]["metadata"] == rows[1]["metadata"]
    expected_source = {"bound": "hub_commit", "offline": "offline", "failed": "resolution_failed",
                       "lost": "unsupported_backend"}[mode]
    assert rows[0]["metadata"]["base_revision_source"] == expected_source
    assert rows[0]["metadata"]["base_revision"] == (SHA if mode == "bound" else None)
    assert rows[0]["calls"] == ([] if mode == "offline" else [BASE])
    assert rows[1]["calls"] == []
    expected_kwargs = {"revision": SHA} if mode in {"bound", "lost"} else {}
    assert all(row["receivers"] == [expected_kwargs] * 3 for row in rows)
