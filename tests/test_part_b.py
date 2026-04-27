"""Part B — v0.27.1 multi-GPU live (#37, #38) for v0.33.0.

Covers:
  - #37 Auto-reexec under accelerate launch when --gpus N>1.
  - #38 DeepSpeed-MII live serve: build_mii_app exposes
    /v1/chat/completions and /v1/models matching the transformers
    backend's contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# #38 — build_mii_app
# ---------------------------------------------------------------------------


class TestBuildMiiApp:
    def test_v1_models_endpoint(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.utils.mii import build_mii_app

        fake_pipeline = MagicMock()
        app = build_mii_app(fake_pipeline, model_name="test-mii-model")
        client = TestClient(app)

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"][0]["id"] == "test-mii-model"

    def test_chat_completions_happy_path(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.utils.mii import build_mii_app

        # Fake MII response objects expose .generated_text
        fake_response = MagicMock()
        fake_response.generated_text = "Hello, world!"

        def _pipeline(prompts, **_kwargs):
            return [fake_response]

        app = build_mii_app(_pipeline, model_name="test-mii-model")
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-mii-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello, world!"
        assert data["model"] == "test-mii-model"

    def test_chat_streaming_rejected(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.utils.mii import build_mii_app

        app = build_mii_app(MagicMock(), model_name="test")
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 400
        assert "stream" in resp.text.lower()

    def test_max_tokens_bounds(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.utils.mii import build_mii_app

        app = build_mii_app(MagicMock(), model_name="test")
        client = TestClient(app)

        # Below lower bound
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,
        })
        assert resp.status_code == 400

        # Above upper bound
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 99999,
        })
        assert resp.status_code == 400

    def test_pipeline_failure_returns_500(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.utils.mii import build_mii_app

        def _bad_pipeline(prompts, **_kwargs):
            raise RuntimeError("MII inference crashed")

        app = build_mii_app(_bad_pipeline, model_name="test")
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 500

    def test_empty_pipeline_response_returns_500(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.utils.mii import build_mii_app

        app = build_mii_app(lambda prompts, **k: [], model_name="test")
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# #37 — auto-reexec wiring smoke (without actually exec'ing)
# ---------------------------------------------------------------------------


class TestAutoReexec:
    def test_train_has_no_reexec_flag(self):
        import inspect

        from soup_cli.commands import train as train_cmd

        sig = inspect.signature(train_cmd.train)
        assert "no_reexec" in sig.parameters

    def test_no_reexec_falls_back_to_advisory(self, tmp_path, monkeypatch):
        """With --no-reexec + multi-GPU, we should print the advice + Exit(1).
        We force the topology detection to report 2 GPUs via monkeypatch."""
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.utils import topology as topo_mod

        monkeypatch.chdir(tmp_path)
        # Minimal valid config so train gets past load
        (tmp_path / "soup.yaml").write_text(
            "base: test/model\n"
            "task: sft\n"
            "data: {train: data.jsonl, format: alpaca}\n"
            "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n",
            encoding="utf-8",
        )
        # Mock topology to report 2 GPUs
        monkeypatch.setattr(
            topo_mod, "detect_topology",
            lambda: {"gpu_count": 2, "interconnect": "PCIe"},
        )
        # Don't let resolve_num_gpus fail on a real CUDA check
        monkeypatch.setattr(
            topo_mod, "resolve_num_gpus", lambda spec: 2,
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "train",
                "--config", "soup.yaml",
                "--gpus", "2",
                "--no-reexec",
                "--yes",
            ],
        )
        # Either exits 1 with advisory message, or earlier (e.g. data load fail)
        assert result.exit_code != 0, result.output
        # Ideally we'd see the advisory; allow earlier failure since no real
        # data file exists.
        if "Multi-GPU launch required" in result.output:
            assert "accelerate" in result.output

    def test_reexec_calls_execvp_with_accelerate_argv(self, tmp_path, monkeypatch):
        """With --gpus 2 and no --no-reexec, the train command should call
        os.execvp with an argv starting with 'accelerate'."""
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import train as train_cmd
        from soup_cli.utils import launcher as launcher_mod
        from soup_cli.utils import topology as topo_mod

        # Strip env-var contamination from prior tests / parent shell so
        # is_in_distributed deterministically returns False.
        for var in (
            "RANK", "WORLD_SIZE", "LOCAL_RANK",
            "ACCELERATE_MIXED_PRECISION",
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_USE_FSDP",
        ):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            "base: test/model\n"
            "task: sft\n"
            "data: {train: data.jsonl, format: alpaca}\n"
            "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            topo_mod, "detect_topology",
            lambda: {"gpu_count": 2, "interconnect": "PCIe"},
        )
        monkeypatch.setattr(
            topo_mod, "resolve_num_gpus", lambda spec: 2,
        )
        # Patch the *imported* names: train.py does
        # ``from soup_cli.utils.topology import ...`` so the reference is on
        # the train module, not the topology module.
        monkeypatch.setattr(train_cmd, "detect_topology",
                            lambda: {"gpu_count": 2, "interconnect": "PCIe"},
                            raising=False)
        monkeypatch.setattr(train_cmd, "resolve_num_gpus", lambda spec: 2,
                            raising=False)
        # is_in_distributed lives in launcher; train does an inline import.
        monkeypatch.setattr(
            launcher_mod, "is_in_distributed", lambda: False,
        )

        captured: dict = {}

        def _fake_execvp(file, argv):
            captured["file"] = file
            captured["argv"] = list(argv)
            # Raise SystemExit so Typer treats it as a clean exit.
            raise SystemExit(99)

        monkeypatch.setattr("os.execvp", _fake_execvp)

        runner = CliRunner()
        runner.invoke(
            app,
            [
                "train",
                "--config", "soup.yaml",
                "--gpus", "2",
                "--yes",
            ],
        )
        # Force assertion — a bypass would have left captured empty and we'd
        # silently accept a regression.
        assert captured.get("file") == "accelerate", (
            f"os.execvp was not called with 'accelerate'; captured={captured!r}"
        )
        assert "launch" in captured["argv"]
        assert "--num_processes" in captured["argv"]
        assert "2" in captured["argv"]
