"""Regression tests for #3370: rank-aware experiment tracking on multi-GPU CLI launches.

All tests run cleanly on CPU without GPUs, model weights, network calls, or
``SOUP_MCP_RUN_ID`` modifications, using isolated SQLite databases in ``tmp_path``
and two-process ``subprocess.Popen`` workers.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.utils.launcher import is_chief_rank

_REPO_ROOT = Path(__file__).resolve().parents[1]

_WORKER_SCRIPT = '''
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
mode = sys.argv[2]
extra_flag = sys.argv[3] if len(sys.argv) > 3 else ""

rank = int(os.environ.get("RANK", "0"))

# Avoid huggingface/git network or subprocess surprises; capture calls cleanly.
import soup_cli.commands.train as train_cmd
import soup_cli.data.loader as data_loader
import soup_cli.trainer.dispatch as trainer_dispatch
import soup_cli.utils.code_revision as code_rev_mod
import soup_cli.utils.gpu as gpu_mod
import soup_cli.utils.run_provenance as provenance
from soup_cli.monitoring.callback import SoupTrainerCallback

gpu_mod.detect_device = lambda: ("cpu", "CPU")
gpu_mod.get_gpu_info = lambda: {}

# Verify collectives remain active on every rank in two-worker runs via a file barrier.
world_size = int(os.environ.get("WORLD_SIZE", "1"))
collective_calls = []
profile_calls = []

def _barrier(tag: str, timeout_s: float = 15.0) -> None:
    if world_size < 2:
        return
    (root / f"barrier-{tag}-{rank}").write_text("1", encoding="utf-8")
    deadline = time.monotonic() + timeout_s
    for peer in range(world_size):
        target = root / f"barrier-{tag}-{peer}"
        while not target.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"rank {rank} timed out waiting for peer {peer} at {tag}")
            time.sleep(0.02)

def _sync_base(identity, cpu=False):
    collective_calls.append("sync")
    _barrier("sync")
    identity.revision = "c" * 40
    identity.source = "hub_commit"
    identity._pending = False

def _confirm_base(identity, model=None):
    collective_calls.append("confirm")
    _barrier("confirm")

provenance.synchronize_base_revision = _sync_base
provenance.confirm_distributed_base_revision = _confirm_base
code_rev_mod.capture_code_revision = lambda *a, **kw: code_rev_mod.CodeRevision(
    commit="b" * 40, dirty=False
)

data_loader.load_dataset = lambda *a, **kw: {
    "train": [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
    ] * 4
}

if extra_flag == "profile":
    import soup_cli.utils.profiling as profiling_mod

    def _fake_profile(*, output_dir, run_id, schedule=None):
        profile_calls.append((rank, str(output_dir), run_id))
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        return nullcontext()

    profiling_mod.profile_training = _fake_profile

class _FakeWrapper:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.model = None
        self.trainer = None

    def setup(self, dataset):
        pass

    def train(self, display=None, tracker=None, run_id="", resume_from_checkpoint=None):
        if mode == "chief_fails" and rank == 0:
            raise RuntimeError("chief crashed during train")
        if mode == "worker_aborts":
            if rank == 1:
                (root / "worker-aborted").write_text("worker 1 OOM", encoding="utf-8")
                raise RuntimeError("worker 1 OOM")
            deadline = time.monotonic() + 15.0
            while not (root / "worker-aborted").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("timed out waiting for worker abort signal")
                time.sleep(0.02)
            raise RuntimeError("distributed collective aborted: worker 1 exited")
        if mode == "isolated_worker_fail" and rank == 1:
            raise RuntimeError("isolated worker failure")
        if mode == "gloo_worker_crash":
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="gloo")
            dist.barrier()
            if rank == 1:
                raise RuntimeError("rank 1 gloo worker crashed")
            dist.barrier()

        cb = SoupTrainerCallback(display=display, tracker=tracker, run_id=run_id)
        losses = [2.0, 1.0] if rank == 0 else [9.9, 8.8]
        for step_idx, loss_val in enumerate(losses, start=1):
            state = SimpleNamespace(
                global_step=step_idx,
                epoch=float(step_idx),
                max_steps=len(losses),
                is_world_process_zero=(rank == 0),
            )
            cb.on_log(
                SimpleNamespace(),
                state,
                SimpleNamespace(),
                logs={"loss": loss_val, "learning_rate": 1e-4},
            )
        out_dir = Path(self.cfg.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "total_steps": len(losses),
            "duration_secs": 1.5 if rank == 0 else 4.0,
            "output_dir": str(out_dir),
        }

trainer_dispatch.build_trainer = lambda cfg, **kw: _FakeWrapper(cfg, **kw)

exit_code = 0
cli_args = ["train", "--config", "soup.yaml", "--yes", "--no-reexec"]
if extra_flag == "profile":
    cli_args.append("--profile")

from typer.testing import CliRunner
from soup_cli.cli import app

cli_result = CliRunner().invoke(app, cli_args)
exit_code = cli_result.exit_code
sys.stdout.write(cli_result.output)
if cli_result.exception is not None and not isinstance(cli_result.exception, SystemExit):
    (root / f"exc-{rank}.txt").write_text(
        f"{type(cli_result.exception).__name__}: {cli_result.exception}",
        encoding="utf-8",
    )

(root / f"worker-{rank}.json").write_text(
    json.dumps({
        "pid": os.getpid(),
        "exit_code": exit_code,
        "collective_calls": collective_calls,
        "profile_calls": profile_calls,
    }),
    encoding="utf-8",
)
sys.exit(exit_code)
'''


def _setup_workspace(tmp_path: Path, db_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "data.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "content": "ok"}]}\n' * 4,
        encoding="utf-8",
    )
    (work / "soup.yaml").write_text(
        "base: sshleifer/tiny-gpt2\n"
        "task: sft\n"
        "experiment_name: exp-3370\n"
        "output: ./out\n"
        "data:\n"
        "  train: data.jsonl\n"
        "  format: chatml\n"
        "  val_split: 0.0\n"
        "training:\n"
        "  epochs: 1\n"
        "  batch_size: 1\n",
        encoding="utf-8",
    )
    script = tmp_path / "worker.py"
    script.write_text(_WORKER_SCRIPT, encoding="utf-8")
    return work


def _worker_env(work: Path, db_path: Path, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_USE_FSDP",
        "SOUP_MCP_RUN_ID",
    ):
        env.pop(key, None)
    pythonpath = str(_REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=pythonpath,
        SOUP_DB_PATH=str(db_path),
        **extra,
    )
    return env


def _read_db(db_path: Path) -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        runs = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY created_at ASC")]
        metrics = [dict(r) for r in conn.execute("SELECT * FROM metrics ORDER BY id ASC")]
        return runs, metrics
    finally:
        conn.close()


def _wait_workers(procs: list[subprocess.Popen], timeout: float = 30.0) -> list[int]:
    codes = []
    for proc in procs:
        try:
            codes.append(proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=5)
            raise
    return codes


class TestIsChiefRank:
    """is_chief_rank mirrors _should_run_diagnose_gate_on_rank cluster-chief rules."""

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({}, True),
            ({"RANK": "0", "LOCAL_RANK": "0"}, True),
            ({"RANK": "1", "LOCAL_RANK": "0"}, False),
            ({"RANK": "8", "LOCAL_RANK": "0"}, False),
            ({"RANK": "", "LOCAL_RANK": "0"}, True),
            ({"RANK": "", "LOCAL_RANK": "1"}, False),
            ({"LOCAL_RANK": "2"}, False),
            ({"RANK": "bad"}, True),
            ({"LOCAL_RANK": "bad"}, True),
        ],
    )
    def test_rank_resolution(self, monkeypatch, env, expected):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert is_chief_rank() is expected


class TestCallbackAndPpoMetricGuards:
    """Defense-in-depth rank gating on SoupTrainerCallback and PPO manual loops."""

    def test_callback_only_writes_metrics_and_sse_on_chief_rank(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.utils import train_event_buffer

        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "cb.db"))
        tracker = ExperimentTracker(db_path=tmp_path / "cb.db")
        run_id = tracker.start_run({"base": "m", "task": "sft"}, "cpu", "CPU", {})

        events = []
        monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)

        # Even if caller didn't strip tracker/run_id, non-chief must not write metrics/SSE.
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        display_worker = MagicMock()
        cb_worker = SoupTrainerCallback(display=display_worker, tracker=tracker, run_id=run_id)
        cb_worker.on_log(
            MagicMock(),
            SimpleNamespace(global_step=1, epoch=0.5, is_world_process_zero=True),
            MagicMock(),
            logs={"loss": 9.9, "learning_rate": 1e-4},
        )
        cb_worker.on_log(
            MagicMock(),
            SimpleNamespace(global_step=1, epoch=0.5, is_world_process_zero=True),
            MagicMock(),
            logs={"train_runtime": 1.0, "train_steps_per_second": 2.0},
        )
        assert tracker.get_metrics(run_id) == []
        assert events == []
        assert display_worker.update.call_count == 2

        # is_world_process_zero=False must also block writes even if env is unset.
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        cb_worker.on_log(
            MagicMock(),
            SimpleNamespace(global_step=1, epoch=0.5, is_world_process_zero=False),
            MagicMock(),
            logs={"loss": 8.8, "learning_rate": 1e-4},
        )
        assert tracker.get_metrics(run_id) == []
        assert events == []

        # Chief rank writes metrics and emits SSE normally.
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        display_chief = MagicMock()
        cb_chief = SoupTrainerCallback(display=display_chief, tracker=tracker, run_id=run_id)
        cb_chief.on_log(
            MagicMock(),
            SimpleNamespace(global_step=1, epoch=0.5, is_world_process_zero=True),
            MagicMock(),
            logs={"loss": 1.25, "learning_rate": 1e-4},
        )
        rows = tracker.get_metrics(run_id)
        assert len(rows) == 1
        assert rows[0]["loss"] == pytest.approx(1.25)
        assert len(events) == 1

    def test_ppo_manual_loop_logs_metrics_only_on_chief_rank(self, tmp_path, monkeypatch):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        tracker = ExperimentTracker(db_path=tmp_path / "ppo.db")
        run_id = tracker.start_run({"base": "m", "task": "ppo"}, "cpu", "CPU", {})
        cfg = load_config_from_string(
            "base: m\n"
            "task: ppo\n"
            f"output: {tmp_path / 'out'}\n"
            "data: {train: ./d.jsonl, format: chatml}\n"
        )
        class _FakePromptDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {"prompt_text": ["hello"]}

        class _QueryTensor:
            def __len__(self):
                return 1

            def unsqueeze(self, dim):
                return [1]

        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        wrapper._output_dir = str(tmp_path / "out")
        wrapper._num_epochs = 1
        wrapper._batch_size = 1
        wrapper._max_length = 32
        wrapper._train_ds = _FakePromptDataset()
        wrapper.tokenizer = SimpleNamespace(
            encode=lambda p, return_tensors="pt": SimpleNamespace(
                squeeze=lambda: _QueryTensor()
            ),
            save_pretrained=lambda out: None,
        )
        wrapper._compute_rewards = lambda q, r, b: [SimpleNamespace()]
        wrapper.trainer = SimpleNamespace(
            generate=lambda *a, **k: SimpleNamespace(squeeze=lambda: [1, 2]),
            step=lambda *a, **k: {"ppo/loss/total": 0.75, "ppo/learning_rate": 1e-5},
            save_model=lambda out: None,
        )

        class _FakeTorch:
            @staticmethod
            def stack(rewards):
                return SimpleNamespace(mean=lambda: SimpleNamespace(item=lambda: 1.0))

        monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        wrapper._train_manual(display=None, tracker=tracker, run_id=run_id)
        assert tracker.get_metrics(run_id) == []

        monkeypatch.setenv("RANK", "0")
        wrapper._train_manual(display=None, tracker=tracker, run_id=run_id)
        rows = tracker.get_metrics(run_id)
        assert len(rows) == 1
        assert rows[0]["loss"] == pytest.approx(0.75)


class TestTwoWorkerCliLifecycle:
    """Real two-process CPU subprocess tests against soup_cli.commands.train.train."""

    def test_two_worker_cli_launch_creates_one_run_one_metric_series_and_provenance(
        self, tmp_path
    ):
        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), "normal"],
                cwd=work,
                env=_worker_env(
                    work,
                    db_path,
                    RANK=str(rank),
                    LOCAL_RANK=str(rank),
                    WORLD_SIZE="2",
                    LOCAL_WORLD_SIZE="2",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for rank in range(2)
        ]
        codes = _wait_workers(procs)
        assert codes == [0, 0], [p.stderr.read() for p in procs]

        w0 = json.loads((tmp_path / "worker-0.json").read_text(encoding="utf-8"))
        w1 = json.loads((tmp_path / "worker-1.json").read_text(encoding="utf-8"))
        assert w0["collective_calls"] == ["sync", "confirm", "confirm"]
        assert w1["collective_calls"] == ["sync", "confirm", "confirm"]

        # Worker stdout must not print broken "soup runs show" without a run ID.
        worker1_stdout = procs[1].stdout.read()
        assert "soup runs show" not in worker1_stdout

        runs, metrics = _read_db(db_path)
        assert len(runs) == 1
        row = runs[0]
        assert row["status"] == "completed"
        assert row["pid"] == w0["pid"]
        assert row["initial_loss"] == pytest.approx(2.0)
        assert row["final_loss"] == pytest.approx(1.0)
        assert row["training_data_sha256"] is not None
        assert row["code_commit"] == "b" * 40
        assert row["base_revision"] == "c" * 40
        assert row["base_revision_source"] == "hub_commit"

        # Metrics come strictly from chief rank (losses 2.0, 1.0 -- never worker's 9.9, 8.8).
        assert [(m["step"], m["loss"]) for m in metrics] == [(1, 2.0), (2, 1.0)]

        # `soup runs` lists exactly one run.
        cli_runs = subprocess.run(
            [sys.executable, "-m", "soup_cli.cli", "runs"],
            cwd=work,
            env=_worker_env(work, db_path),
            capture_output=True,
            text=True,
            check=True,
        )
        assert cli_runs.stdout.count(row["run_id"]) == 1
        provenance_file = work / "out" / "soup_run.json"
        assert provenance_file.is_file()
        assert json.loads(provenance_file.read_text(encoding="utf-8"))["run_id"] == row["run_id"]

    def test_two_worker_with_profile_only_profiles_chief_rank(self, tmp_path):
        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), "normal", "profile"],
                cwd=work,
                env=_worker_env(
                    work,
                    db_path,
                    RANK=str(rank),
                    LOCAL_RANK=str(rank),
                    WORLD_SIZE="2",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for rank in range(2)
        ]
        codes = _wait_workers(procs)
        assert codes == [0, 0]

        w0 = json.loads((tmp_path / "worker-0.json").read_text(encoding="utf-8"))
        w1 = json.loads((tmp_path / "worker-1.json").read_text(encoding="utf-8"))
        assert len(w0["profile_calls"]) == 1 and w0["profile_calls"][0][0] == 0
        assert w1["profile_calls"] == []

    def test_chief_failure_not_overwritten_by_worker_completion(self, tmp_path):
        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), "chief_fails"],
                cwd=work,
                env=_worker_env(
                    work,
                    db_path,
                    RANK=str(rank),
                    LOCAL_RANK=str(rank),
                    WORLD_SIZE="2",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for rank in range(2)
        ]
        codes = _wait_workers(procs)
        assert codes[0] != 0 and codes[1] == 0

        runs, metrics = _read_db(db_path)
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"
        assert "chief crashed during train" in (runs[0]["error_message"] or "")
        assert metrics == []

    def test_simulated_worker_abort_propagated_to_chief_records_single_failed_run(
        self, tmp_path
    ):
        """Simulates cross-worker abort propagation via a filesystem marker.

        Unlike a real ``torchrun``/``accelerate`` supervisor, this test uses a
        marker file so rank 0 raises after rank 1 aborts, exercising chief
        failure persistence and single-run ownership on CPU without ``torch``.
        """
        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), "worker_aborts"],
                cwd=work,
                env=_worker_env(
                    work,
                    db_path,
                    RANK=str(rank),
                    LOCAL_RANK=str(rank),
                    WORLD_SIZE="2",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for rank in range(2)
        ]
        codes = _wait_workers(procs)
        assert codes[0] != 0 and codes[1] != 0

        runs, _ = _read_db(db_path)
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"
        assert "worker 1 exited" in (runs[0]["error_message"] or "")

    def test_torchrun_gloo_worker_failure_teardown_never_leaves_completed_run(
        self, tmp_path
    ):
        """Real torch.distributed.run supervisor teardown on rank-1 crash (Gloo CPU)."""
        if sys.platform != "linux":
            pytest.skip("Real torch.distributed.run Gloo subprocess test requires Linux")
        pytest.importorskip("torch")
        dist = pytest.importorskip("torch.distributed")
        if not dist.is_gloo_available():
            pytest.skip("torch.distributed Gloo backend is not available")

        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        env = _worker_env(
            work,
            db_path,
            ACCELERATE_USE_CPU="true",
            USE_LIBUV="0",
            OMP_NUM_THREADS="1",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint=127.0.0.1:{port}",
                str(script),
                str(tmp_path),
                "gloo_worker_crash",
            ],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode != 0

        runs, metrics = _read_db(db_path)
        assert len(runs) == 1
        row = runs[0]
        assert row["status"] != "completed"
        assert row["status"] in {"failed", "running"}
        assert not (row["status"] == "completed" and row["error_message"])
        assert metrics == []

    def test_isolated_non_chief_failure_writes_nothing_to_db(self, tmp_path):
        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=db_path)
        chief_id = tracker.start_run({"base": "m", "task": "sft"}, "cpu", "CPU", {})
        tracker.finish_run(chief_id, 1.5, 0.5, 5, 2.0, str(work / "out"), status="completed")

        proc = subprocess.run(
            [sys.executable, str(script), str(tmp_path), "isolated_worker_fail"],
            cwd=work,
            env=_worker_env(work, db_path, RANK="1", LOCAL_RANK="1", WORLD_SIZE="2"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        runs, _ = _read_db(db_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == chief_id
        assert runs[0]["status"] == "completed"

    def test_single_process_cli_tracking_unchanged(self, tmp_path):
        db_path = tmp_path / "experiments.db"
        work = _setup_workspace(tmp_path, db_path)
        script = tmp_path / "worker.py"

        proc = subprocess.run(
            [sys.executable, str(script), str(tmp_path), "normal"],
            cwd=work,
            env=_worker_env(work, db_path),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr

        runs, metrics = _read_db(db_path)
        assert len(runs) == 1
        row = runs[0]
        assert row["status"] == "completed"
        assert row["initial_loss"] == pytest.approx(2.0)
        assert row["final_loss"] == pytest.approx(1.0)
        assert row["training_data_sha256"] is not None
        assert row["code_commit"] == "b" * 40
        assert [(m["step"], m["loss"]) for m in metrics] == [(1, 2.0), (2, 1.0)]
        assert row["run_id"] in proc.stdout
        assert f"soup runs show {row['run_id']}" in proc.stdout


class TestMlxRankAwareTelemetry:
    """Defense-in-depth rank gating for MLX tracker/SSE writes (#3370)."""

    def test_mlx_sft_callback_only_writes_metrics_and_sse_on_chief_rank(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils import train_event_buffer
        from tests.test_issue23_mlx_display_bridge import (
            _REPORT_1,
            _RecordingDisplay,
            _RecordingTracker,
        )
        from tests.test_issue634_mlx_resume import (
            _FakeMlxModel,
            _install_fake_mlx,
            _mlx_wrapper,
        )

        _install_fake_mlx(monkeypatch)
        events = []
        monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)

        def _fake_train(**kwargs):
            callback = kwargs.get("training_callback")
            if callback is not None:
                callback.on_val_loss_report({"iteration": 1, "val_loss": 2.5})
                callback.on_train_loss_report(_REPORT_1)

        sys.modules["mlx_lm.tuner.trainer"].train = _fake_train

        def _make_wrapper(name: str):
            out = tmp_path / name
            wrapper = _mlx_wrapper(out, epochs=1, lr=1e-4, batch_size=1)
            wrapper.model = _FakeMlxModel()
            wrapper.tokenizer = object()
            wrapper._dataset = {"train": [{"text": "hi"}] * 48, "val": []}
            return wrapper

        # Non-chief worker: display still updates, tracker and SSE stay silent.
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        worker_display = _RecordingDisplay()
        worker_tracker = _RecordingTracker()
        _make_wrapper("non_chief").train(
            display=worker_display, tracker=worker_tracker, run_id="run-mlx"
        )
        assert len(worker_display.updates) == 2
        assert worker_tracker.metrics == []
        assert events == []

        # Chief worker: tracker and SSE both record eval + train telemetry.
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        chief_display = _RecordingDisplay()
        chief_tracker = _RecordingTracker()
        _make_wrapper("chief").train(
            display=chief_display, tracker=chief_tracker, run_id="run-mlx"
        )
        assert len(chief_display.updates) == 2
        assert [m["kind"] for m in chief_tracker.metrics] == ["eval", "train"]
        assert len(events) == 2

    def test_mlx_dpo_report_only_writes_metrics_and_sse_on_chief_rank(
        self, monkeypatch
    ):
        from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper
        from soup_cli.utils import train_event_buffer
        from tests.test_issue23_mlx_display_bridge import (
            _RecordingDisplay,
            _RecordingTracker,
        )

        events = []
        monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)
        wrapper = MLXDPOTrainerWrapper.__new__(MLXDPOTrainerWrapper)

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        worker_display = _RecordingDisplay()
        worker_tracker = _RecordingTracker()
        wrapper._report(
            worker_display,
            worker_tracker,
            "run-dpo",
            step=2,
            epoch=0.5,
            loss=0.4,
            lr=1e-5,
            speed=1.5,
            metrics={"accuracy": 0.9},
            val={"val_loss": 0.35},
        )
        assert len(worker_display.updates) == 1
        assert worker_tracker.metrics == []
        assert events == []

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        chief_display = _RecordingDisplay()
        chief_tracker = _RecordingTracker()
        wrapper._report(
            chief_display,
            chief_tracker,
            "run-dpo",
            step=2,
            epoch=0.5,
            loss=0.4,
            lr=1e-5,
            speed=1.5,
            metrics={"accuracy": 0.9},
            val={"val_loss": 0.35},
        )
        assert len(chief_display.updates) == 1
        assert len(chief_tracker.metrics) == 1
        assert len(events) == 1

    def test_mlx_grpo_log_val_reward_only_writes_on_chief_rank(self, monkeypatch):
        from soup_cli.trainer.mlx_grpo import log_val_reward
        from tests.test_issue23_mlx_display_bridge import _RecordingTracker

        tracker = _RecordingTracker()

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        log_val_reward(tracker, "run-grpo", step=4, epoch=1.0, lr=1e-5, val_reward=0.75)
        assert tracker.metrics == []

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        log_val_reward(tracker, "run-grpo", step=4, epoch=1.0, lr=1e-5, val_reward=0.75)
        assert len(tracker.metrics) == 1
        assert tracker.metrics[0]["val_reward"] == pytest.approx(0.75)
        assert tracker.metrics[0]["kind"] == "eval"
