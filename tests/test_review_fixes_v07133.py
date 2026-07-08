"""Regression tests for the v0.71.32 full-tree review findings.

Covers the 6 HIGH + MEDIUM/LOW findings surfaced by the review round:
vLLM trust_remote_code gate, vision image path containment, functional
multi-adapter serving, --dry-run/--gpus re-exec guard + dropped-flag forwarding,
MLX optimizer construction, Rich-markup escaping in `soup data`, the ASR infer
metric-guard / control-strip / task-validation / exit-code fixes, the distill KD
off-by-one shift, the load_config_from_string ValueError contract, `soup runs`
escaping, the constant-time UI token compare, and the atomic-write centralisation.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner


# --------------------------------------------------------------------------
# HIGH-1 — vLLM serve must not force trust_remote_code=True
# --------------------------------------------------------------------------
class TestVllmTrustRemoteCode:
    def _install_fake_vllm(self, monkeypatch):
        captured: dict = {}

        class _FakeEngineArgs:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class _FakeEngine:
            @classmethod
            def from_engine_args(cls, args):
                return object()

        fake = types.ModuleType("vllm")
        fake.AsyncEngineArgs = _FakeEngineArgs
        fake.AsyncLLMEngine = _FakeEngine
        monkeypatch.setitem(sys.modules, "vllm", fake)
        return captured

    def test_default_is_false(self, monkeypatch):
        from soup_cli.utils.vllm import create_vllm_engine

        captured = self._install_fake_vllm(monkeypatch)
        create_vllm_engine(model_path="some/model")
        assert captured["trust_remote_code"] is False

    def test_true_is_threaded_through(self, monkeypatch):
        from soup_cli.utils.vllm import create_vllm_engine

        captured = self._install_fake_vllm(monkeypatch)
        create_vllm_engine(model_path="some/model", trust_remote_code=True)
        assert captured["trust_remote_code"] is True

    def test_serve_vllm_passes_resolved_trust(self, monkeypatch):
        # _serve_vllm must forward its trust_remote_code down to the engine.
        import soup_cli.commands.serve as serve

        seen: dict = {}

        def _fake_engine(**kwargs):
            seen.update(kwargs)
            return object(), "m"

        monkeypatch.setattr(
            "soup_cli.utils.vllm.create_vllm_engine", _fake_engine
        )
        monkeypatch.setattr(
            "soup_cli.utils.vllm.create_vllm_app",
            lambda **kw: object(),
        )
        serve._serve_vllm(
            model_path=Path("m"),
            base_model=None,
            is_adapter=False,
            max_tokens_default=128,
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        assert seen.get("trust_remote_code") is True


# --------------------------------------------------------------------------
# HIGH-2 — vision dataset image paths must be containment-checked
# --------------------------------------------------------------------------
class TestVisionImageContainment:
    def test_absolute_traversal_dropped(self, tmp_path):
        from soup_cli.data.loader import _validate_vision_images

        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "ok.png").write_bytes(b"x")
        rows = [
            {"image": "ok.png", "messages": []},
            {"image": "/etc/passwd", "messages": []},
        ]
        out = _validate_vision_images([dict(r) for r in rows], img_dir)
        kept = [Path(r["image"]).name for r in out]
        assert "ok.png" in kept
        assert "passwd" not in kept
        assert len(out) == 1

    def test_relative_traversal_dropped(self, tmp_path):
        from soup_cli.data.loader import _validate_vision_images

        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        secret = tmp_path / "secret.png"
        secret.write_bytes(b"x")
        out = _validate_vision_images(
            [{"image": "../secret.png", "messages": []}], img_dir
        )
        assert out == []


# --------------------------------------------------------------------------
# HIGH-3 — multi-adapter serving actually switches adapters
# --------------------------------------------------------------------------
class _FakeAdapterModel:
    """Records set_adapter + disable_adapter usage."""

    def __init__(self):
        self.set_calls: list = []
        self.disable_calls = 0
        self.active = None

    def set_adapter(self, name):
        self.set_calls.append(name)
        self.active = name

    def disable_adapter(self):
        model = self

        class _Ctx:
            def __enter__(self):
                model.disable_calls += 1
                return self

            def __exit__(self, *exc):
                return False

        return _Ctx()


class TestAdapterScope:
    def _run(self, requested, active, names):
        import threading

        from soup_cli.commands.serve import _adapter_scope

        model = _FakeAdapterModel()
        lock = threading.Lock()
        with _adapter_scope(model, lock, names, requested, active):
            pass
        return model

    def test_requested_overrides_active(self):
        model = self._run("code", "chat", {"chat", "code"})
        assert model.set_calls == ["code"]
        assert model.disable_calls == 0

    def test_falls_back_to_active(self):
        model = self._run(None, "chat", {"chat", "code"})
        assert model.set_calls == ["chat"]

    def test_unknown_name_uses_base(self):
        model = self._run("nope", None, {"chat"})
        assert model.set_calls == []
        assert model.disable_calls == 1

    def test_none_selects_base(self):
        model = self._run(None, None, {"chat"})
        assert model.disable_calls == 1

    def test_no_adapters_is_noop(self):
        model = self._run("chat", "chat", set())
        assert model.set_calls == []
        assert model.disable_calls == 0


class TestLoadNamedAdapters:
    def test_wraps_plain_model_then_loads_rest(self, monkeypatch):
        import soup_cli.commands.serve as serve

        loaded: dict = {"from_pretrained": [], "load_adapter": []}

        class _FakePeft:
            def __init__(self, name="default"):
                self.name = name

            @classmethod
            def from_pretrained(cls, model, path, adapter_name=None):
                loaded["from_pretrained"].append((path, adapter_name))
                return cls(adapter_name)

            def load_adapter(self, path, adapter_name=None):
                loaded["load_adapter"].append((path, adapter_name))

            def eval(self):
                return self

        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = _FakePeft
        monkeypatch.setitem(sys.modules, "peft", fake_peft)

        plain = object()  # not a PeftModel → must be wrapped
        model, names = serve._load_named_adapters(
            plain, {"chat": "a/path", "code": "b/path"}
        )
        assert names == {"chat", "code"}
        # First adapter wraps the base; the rest are load_adapter'd.
        assert loaded["from_pretrained"] == [("a/path", "chat")]
        assert loaded["load_adapter"] == [("b/path", "code")]


# --------------------------------------------------------------------------
# HIGH-4 — --dry-run must NOT re-exec, and dropped flags must forward
# --------------------------------------------------------------------------
def _write_sft_config(tmp_path):
    (tmp_path / "soup.yaml").write_text(
        "base: test/model\n"
        "task: sft\n"
        "data: {train: data.jsonl, format: alpaca}\n"
        "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n",
        encoding="utf-8",
    )


def _force_two_gpus(monkeypatch):
    from soup_cli.commands import train as train_cmd
    from soup_cli.utils import launcher as launcher_mod
    from soup_cli.utils import topology as topo_mod

    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    topo = {"gpu_count": 2, "interconnect": "PCIe"}
    monkeypatch.setattr(topo_mod, "detect_topology", lambda: topo)
    monkeypatch.setattr(topo_mod, "resolve_num_gpus", lambda spec: 2)
    monkeypatch.setattr(train_cmd, "detect_topology", lambda: topo, raising=False)
    monkeypatch.setattr(
        train_cmd, "resolve_num_gpus", lambda spec: 2, raising=False
    )
    monkeypatch.setattr(launcher_mod, "is_in_distributed", lambda: False)


class TestDryRunNoReexec:
    def test_dry_run_does_not_execvp(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_sft_config(tmp_path)
        _force_two_gpus(monkeypatch)

        called = {"execvp": False}

        def _fake_execvp(file, argv):
            called["execvp"] = True
            raise SystemExit(99)

        monkeypatch.setattr("os.execvp", _fake_execvp)

        CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--gpus", "2", "--dry-run"]
        )
        assert called["execvp"] is False, (
            "os.execvp was called on a --dry-run multi-GPU invocation"
        )


class TestReexecForwardsFlags:
    def test_capture_flags_forwarded(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_sft_config(tmp_path)
        _force_two_gpus(monkeypatch)

        captured: dict = {}

        def _fake_execvp(file, argv):
            captured["argv"] = list(argv)
            raise SystemExit(99)

        monkeypatch.setattr("os.execvp", _fake_execvp)

        CliRunner().invoke(
            app,
            [
                "train", "--config", "soup.yaml", "--gpus", "2", "--yes",
                "--capture-activations", "model.layers.5",
                "--capture-prompts", "p.jsonl",
            ],
        )
        argv = captured.get("argv", [])
        assert "--capture-activations" in argv
        assert "model.layers.5" in argv
        assert "--capture-prompts" in argv
        assert "p.jsonl" in argv


# --------------------------------------------------------------------------
# HIGH-5 — MLX SFT must build a real optimizer (not pass None)
# --------------------------------------------------------------------------
class TestMlxOptimizer:
    def test_train_passes_non_none_optimizer(self, tmp_path, monkeypatch):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        cfg = load_config_from_string(
            "base: mlx-community/tiny\n"
            "task: sft\n"
            "backend: mlx\n"
            "data: {train: d.jsonl, format: chatml}\n"
            "training: {epochs: 1, lr: 2e-4, batch_size: 1}\n"
            f"output: {json.dumps(str(tmp_path / 'out'))}\n"
        )
        wrapper = MLXSFTTrainerWrapper(cfg)
        wrapper.model = object()
        wrapper.tokenizer = object()
        wrapper._dataset = {"train": [{"messages": []}], "val": []}
        monkeypatch.setattr(wrapper, "_require_mlx", lambda: None)

        seen: dict = {}

        def _fake_train(**kwargs):
            seen.update(kwargs)

        class _FakeArgs:
            def __init__(self, **kwargs):
                pass

        sentinel = object()

        trainer_mod = types.ModuleType("mlx_lm.tuner.trainer")
        trainer_mod.TrainingArgs = _FakeArgs
        trainer_mod.train = _fake_train
        tuner_mod = types.ModuleType("mlx_lm.tuner")
        mlx_lm_mod = types.ModuleType("mlx_lm")
        opt_mod = types.ModuleType("mlx.optimizers")
        opt_mod.AdamW = lambda learning_rate=None: sentinel
        mlx_root = types.ModuleType("mlx")

        monkeypatch.setitem(sys.modules, "mlx", mlx_root)
        monkeypatch.setitem(sys.modules, "mlx.optimizers", opt_mod)
        monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
        monkeypatch.setitem(sys.modules, "mlx_lm.tuner", tuner_mod)
        monkeypatch.setitem(sys.modules, "mlx_lm.tuner.trainer", trainer_mod)

        wrapper.train()
        assert "optimizer" in seen
        assert seen["optimizer"] is sentinel


# --------------------------------------------------------------------------
# HIGH-6 — `soup data inspect` must escape dataset-derived Rich markup
# --------------------------------------------------------------------------
class TestDataInspectEscape:
    def test_bracket_slash_does_not_crash(self, tmp_path, monkeypatch):
        from soup_cli.commands.data import app as data_app

        monkeypatch.chdir(tmp_path)
        data_file = tmp_path / "d.jsonl"
        # A stray '[/]' + a crafted link tag in ordinary data.
        rows = [
            {"instruction": "hi [/] there", "output": "[link=http://x]clickme[/]"},
            {"instruction": "normal", "output": "fine"},
        ]
        data_file.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        result = CliRunner().invoke(data_app, ["inspect", "d.jsonl"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Literal markup survives as text (escaped), not interpreted.
        assert "[/]" in result.output


# --------------------------------------------------------------------------
# ASR — infer.py metric guard / control-strip / task-validate / exit code
# --------------------------------------------------------------------------
def _write_asr_input(tmp_path, rows):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


class TestAsrInferMetricGuard:
    def test_oversized_reference_does_not_crash(self, tmp_path, monkeypatch):
        import soup_cli.commands.infer as infer
        from soup_cli.cli import app as cli_app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.wav").write_bytes(b"x")
        (tmp_path / "b.wav").write_bytes(b"x")
        big = "word " * 300_000  # > _MAX_RAW_CHARS
        _write_asr_input(
            tmp_path,
            [
                {"audio": "a.wav", "text": big},        # metric must be skipped
                {"audio": "b.wav", "text": "hello"},    # metric ok
            ],
        )
        monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", lambda p: "hello")

        result = CliRunner().invoke(
            cli_app,
            ["infer", "--task", "asr", "--model", "whatever",
             "--input", "in.jsonl", "--output", "out.jsonl"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        lines = [
            json.loads(x)
            for x in (tmp_path / "out.jsonl").read_text().splitlines() if x
        ]
        assert len(lines) == 2  # both rows still transcribed
        # Oversized row present but unscored; the small row scored.
        assert "wer" not in lines[0]
        assert "wer" in lines[1]


class TestAsrSkipControlStrip:
    def test_hostile_filename_is_stripped(self, tmp_path, monkeypatch):
        import soup_cli.commands.infer as infer
        from soup_cli.cli import app as cli_app

        monkeypatch.chdir(tmp_path)
        # A filename carrying a raw ESC byte; transcriber raises so the skip
        # warning path (which prints the name) runs.
        (tmp_path / "clip.wav").write_bytes(b"x")

        def _boom(_p):
            raise ValueError("bad \x1b[31maudio")

        monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", _boom)
        _write_asr_input(tmp_path, [{"audio": "clip.wav"}])
        result = CliRunner().invoke(
            cli_app,
            ["infer", "--task", "asr", "--model", "m",
             "--input", "in.jsonl", "--output", "out.jsonl"],
        )
        # All rows skipped → exit 2 (L2), and no raw ESC reaches the terminal.
        assert result.exit_code == 2
        assert "\x1b" not in result.output


class TestAsrTaskValidation:
    def test_bad_asr_task_rejected_upfront(self, tmp_path, monkeypatch):
        from soup_cli.cli import app as cli_app

        monkeypatch.chdir(tmp_path)
        _write_asr_input(tmp_path, [{"audio": "a.wav"}])
        result = CliRunner().invoke(
            cli_app,
            ["infer", "--task", "asr", "--model", "m", "--input", "in.jsonl",
             "--output", "out.jsonl", "--asr-task", "translat"],
        )
        assert result.exit_code == 2
        assert "transcribe" in result.output


class TestAsrAllSkippedExit:
    def test_all_rows_skipped_exit_2(self, tmp_path, monkeypatch):
        import soup_cli.commands.infer as infer
        from soup_cli.cli import app as cli_app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.wav").write_bytes(b"x")

        def _boom(_p):
            raise OSError("cannot decode")

        monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", _boom)
        _write_asr_input(tmp_path, [{"audio": "a.wav"}])
        result = CliRunner().invoke(
            cli_app,
            ["infer", "--task", "asr", "--model", "m",
             "--input", "in.jsonl", "--output", "out.jsonl"],
        )
        assert result.exit_code == 2
        assert not (tmp_path / "out.jsonl").exists()


# --------------------------------------------------------------------------
# MEDIUM — distill KD term must shift like the CE term
# --------------------------------------------------------------------------
class TestDistillKdShift:
    def test_mask_aligns_to_predicted_token(self):
        import torch

        from soup_cli.trainer.distill import _compute_distill_term

        # seq=4, vocab=3. Teacher/student differ ONLY at position 1 (which,
        # after the causal shift, predicts token at index 2).
        student = torch.zeros(1, 4, 3)
        teacher = torch.zeros(1, 4, 3)
        teacher[0, 1, 0] = 10.0  # divergence localised at position 1

        # Case A: only the PREDICTED token (index 2) is trained → after the
        # shift, position 1 is included → non-zero divergence.
        labels_a = torch.tensor([[-100, -100, 5, -100]])
        loss_a = _compute_distill_term(
            student, teacher, "forward_kl", 1.0, labels=labels_a
        )
        assert float(loss_a) > 0.0

        # Case B: only the INPUT token at index 1 is trained (its prediction,
        # index 1's target = index 2, is NOT) → position 1 excluded → ~0.
        labels_b = torch.tensor([[-100, 5, -100, -100]])
        loss_b = _compute_distill_term(
            student, teacher, "forward_kl", 1.0, labels=labels_b
        )
        assert float(loss_b) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# MEDIUM — load_config_from_string ValueError-only contract
# --------------------------------------------------------------------------
class TestLoaderNonMapping:
    def test_bare_list_raises_valueerror(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="mapping"):
            load_config_from_string("- a\n- b\n")

    def test_scalar_raises_valueerror(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="mapping"):
            load_config_from_string("42\n")


# --------------------------------------------------------------------------
# MEDIUM — `soup runs` escapes config-derived strings
# --------------------------------------------------------------------------
class TestRunsEscape:
    def _install_fake_tracker(self, monkeypatch, run):
        class _FakeTracker:
            def __init__(self, *a, **k):
                pass

            def list_runs(self, limit=50):
                return [run]

            def get_run(self, run_id):
                return run

            def get_eval_results(self, run_id=None):
                return []

            def get_metrics(self, run_id):
                return []

            def close(self):
                pass

        monkeypatch.setattr(
            "soup_cli.experiment.tracker.ExperimentTracker", _FakeTracker
        )

    def _crafted_run(self):
        return {
            "run_id": "abcd1234",
            "experiment_name": "exp [/] boom",
            "base_model": "org/[link=http://x]evil[/]",
            "task": "sft",
            "status": "completed",
            "created_at": "2026-07-08T10:00:00",
            "total_steps": 10,
            "duration_secs": 60,
            "device_name": "GPU",
            "device": "cuda",
            "gpu_memory": "4GB",
            "output_dir": "out",
            "config_json": None,
        }

    def test_list_does_not_crash(self, monkeypatch):
        from soup_cli.commands.runs import app as runs_app

        self._install_fake_tracker(monkeypatch, self._crafted_run())
        result = CliRunner().invoke(runs_app, [])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_show_does_not_crash(self, monkeypatch):
        from soup_cli.commands.runs import app as runs_app

        self._install_fake_tracker(monkeypatch, self._crafted_run())
        result = CliRunner().invoke(runs_app, ["show", "abcd1234", "--no-plot"])
        assert result.exit_code == 0, (result.output, repr(result.exception))


# --------------------------------------------------------------------------
# MEDIUM — constant-time UI token compare (behaviour preserved)
# --------------------------------------------------------------------------
class TestConstantTimeToken:
    def test_wrong_token_401_right_token_ok(self):
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        from soup_cli.ui.app import create_app, set_auth_token

        set_auth_token("a" * 32)
        app = create_app()
        client = TestClient(app)
        # A mutating endpoint requires the Bearer token.
        bad = client.post(
            "/api/train/start",
            headers={"Authorization": "Bearer wrong"},
            json={"config": "base: x"},
        )
        assert bad.status_code == 401
        # Correct token gets past the auth gate (may fail later on validation,
        # but never with 401).
        ok = client.post(
            "/api/train/start",
            headers={"Authorization": "Bearer " + "a" * 32},
            json={"config": "not: valid: config"},
        )
        assert ok.status_code != 401


# --------------------------------------------------------------------------
# MEDIUM — atomic writes for vscode_setup + lr_finder
# --------------------------------------------------------------------------
class TestAtomicWrites:
    def test_vscode_writes_and_honours_force(self, tmp_path, monkeypatch):
        from soup_cli.utils.vscode_setup import write_vscode_launch

        monkeypatch.chdir(tmp_path)
        out = write_vscode_launch(config_path="soup.yaml", target_dir=".vscode")
        assert Path(out).exists()
        payload = json.loads(Path(out).read_text())
        assert "configurations" in payload
        # Second write without force must refuse.
        with pytest.raises(FileExistsError):
            write_vscode_launch(config_path="soup.yaml", target_dir=".vscode")
        # force=True overwrites atomically.
        write_vscode_launch(
            config_path="other.yaml", target_dir=".vscode", force=True
        )
        assert Path(out).exists()

    def test_lr_finder_writes_under_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.lr_finder import save_lr_finder_report

        monkeypatch.chdir(tmp_path)
        out = tmp_path / "lr.json"
        save_lr_finder_report([1e-5, 1e-4, 1e-3, 1e-2], [3.0, 2.0, 2.5, 4.0], out)
        payload = json.loads(out.read_text())
        assert "recommended_lr" in payload

    def test_lr_finder_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.lr_finder import save_lr_finder_report

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "escape.json"
        with pytest.raises(ValueError):
            save_lr_finder_report([1e-5, 1e-4], [3.0, 2.0], outside)
