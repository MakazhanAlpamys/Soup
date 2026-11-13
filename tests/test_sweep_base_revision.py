"""A sweep arm owns one nullable base identity, including its portable artifact."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console

from soup_cli.utils import run_provenance as provenance
from tests.conftest import make_cli_runner, strip_ansi

BASE = "owner/base"
OTHER = "owner/teacher"
FIRST = "a" * 40
SECOND = "b" * 40
UNKNOWN = "Base revision: unknown (existing loading behavior preserved)"


def _plain(text):
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def sweep_cli(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from soup_cli.cli import app
    from soup_cli.commands import sweep
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.trainer import dispatch
    from soup_cli.utils import gpu

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "RANK", "WORLD_SIZE",
                "ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(gpu, "detect_device", lambda: ("cpu", "CPU"))
    monkeypatch.setattr(gpu, "get_gpu_info", lambda: {})
    monkeypatch.setattr(gpu, "configure_mps_async_load", lambda device: None)
    (tmp_path / "rows.jsonl").write_text(
        '{"instruction":"hello", "output":"world"}\n', encoding="utf-8",
    )
    config = {
        "base": BASE, "data": {"train": "rows.jsonl", "format": "alpaca", "val_split": 0},
        "training": {"epochs": 1, "quantization": "none", "batch_size": 1},
        "output": "artifacts",
    }
    state = SimpleNamespace(config=config, calls=[], lookups=[], failure=None, setup_loads=True,
                            real_build_trainer=dispatch.build_trainer)

    def model_info(base, **kwargs):
        state.lookups.append(base)
        return SimpleNamespace(sha=FIRST if len(state.lookups) == 1 else SECOND)

    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=model_info))

    class Trainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def load(self, stage, base=None):
            base = base or self.cfg.base
            kwargs = provenance.base_revision_kwargs(base)
            state.calls.append((self.cfg.experiment_name, stage, base, kwargs))

        def setup(self, dataset):
            assert len(dataset["train"]) == 1
            if state.setup_loads:
                for stage in ("tokenizer", "config", "processor", "model"):
                    self.load(stage)
            if state.failure == "setup":
                raise RuntimeError("setup failed after base load")

        def train(self, **kwargs):
            self.load("training-model")
            self.load("teacher", OTHER)
            if state.failure == "train":
                raise RuntimeError("training failed after base load")
            output = Path(self.cfg.output) / self.cfg.experiment_name
            output.mkdir(parents=True)
            (output / "receiver.json").write_text(json.dumps(state.calls), encoding="utf-8")
            return {"output_dir": str(output), "initial_loss": 1.0, "final_loss": 0.5,
                    "total_steps": 1, "duration_secs": 1.0}

    monkeypatch.setattr(dispatch, "build_trainer", lambda cfg, **kwargs: Trainer(cfg))

    def invoke(param="lr=0.001,0.002", *, colour=False, extra=()):
        monkeypatch.setattr(sweep, "console", Console(
            width=500, height=100, force_terminal=colour, no_color=False,
        ))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        return make_cli_runner().invoke(app, [
            "--no-audit-log", "--no-telemetry", "sweep", "--config", "soup.yaml",
            "--yes", "--param", param, *extra,
        ])

    def runs():
        tracker = ExperimentTracker(tmp_path / "runs.db")
        try:
            return sorted(tracker.list_runs(), key=lambda row: row["experiment_name"])
        finally:
            tracker.close()

    return SimpleNamespace(invoke=invoke, runs=runs, state=state, root=tmp_path)


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
def test_each_arm_pins_its_setup_loads_and_records_the_commit(sweep_cli, colour):
    result = sweep_cli.invoke(colour=colour)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    setup = [call for call in sweep_cli.state.calls if call[1] in {
        "tokenizer", "config", "processor", "model",
    }]
    assert setup == [
        (arm, stage, BASE, {"revision": revision})
        for arm, revision in (("sweep_1", FIRST), ("sweep_2", SECOND))
        for stage in ("tokenizer", "config", "processor", "model")
    ]
    assert sweep_cli.state.lookups == [BASE, BASE]
    rows = sweep_cli.runs()
    assert [(row["base_model"], row["base_revision"], row["base_revision_source"])
            for row in rows] == [(BASE, FIRST, "hub_commit"), (BASE, SECOND, "hub_commit")]
    assert all(row["training_data_sha256"] is None for row in rows)
    for revision in (FIRST, SECOND):
        assert _plain(result.output).count(f"Base revision: {revision} (bound Hub commit)") == 1
    assert provenance.base_revision_kwargs(BASE) == {}


@pytest.mark.parametrize("setup_loads", [True, False], ids=["setup-load", "training-only-load"])
def test_training_loads_and_copied_final_artifact_keep_the_arms_identity(sweep_cli, setup_loads):
    sweep_cli.state.setup_loads = setup_loads
    result = sweep_cli.invoke(param="lr=0.001", colour=True)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert [call for call in sweep_cli.state.calls if call[1] == "training-model"] == [
        ("sweep_1", "training-model", BASE, {"revision": FIRST}),
    ]
    assert [call for call in sweep_cli.state.calls if call[1] == "teacher"] == [
        ("sweep_1", "teacher", OTHER, {}),
    ]
    assert sweep_cli.state.lookups == [BASE]
    row, = sweep_cli.runs()
    assert (row["status"], row["base_revision"], row["base_revision_source"]) == (
        "completed", FIRST, "hub_commit",
    )
    assert row["training_data_sha256"] is None
    assert f"Base revision: {FIRST} (bound Hub commit)" in _plain(result.output)
    artifact = sweep_cli.root / row["output_dir"]
    copied = sweep_cli.root / "copied"
    shutil.copytree(artifact, copied)
    shutil.rmtree(artifact)
    metadata = provenance.read_run_provenance(copied)
    assert metadata == {
        "schema_version": 1, "run_id": row["run_id"], "base_model": BASE,
        "base_revision": FIRST, "base_revision_source": "hub_commit",
        "training_data_sha256": None,
    }
    # This is the same public reader used by export and merge; no original DB
    # or artifact directory is supplied to recover the recorded pin.
    assert provenance.recorded_base_kwargs(str(copied), BASE) == {"revision": FIRST}
    assert provenance.recorded_base_kwargs(str(copied), OTHER) == {}
    assert provenance.base_revision_kwargs(BASE) == {}


def test_arms_can_name_different_bases_without_reusing_another_arms_identity(sweep_cli):
    result = sweep_cli.invoke(param=f"base={BASE},owner/second-base")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert sweep_cli.state.lookups == [BASE, "owner/second-base"]
    rows = sweep_cli.runs()
    assert [(row["base_model"], row["base_revision"]) for row in rows] == [
        (BASE, FIRST), ("owner/second-base", SECOND),
    ]
    for row in rows:
        output = sweep_cli.root / row["output_dir"]
        assert provenance.recorded_base_kwargs(str(output), row["base_model"]) == {
            "revision": row["base_revision"],
        }
        assert provenance.recorded_base_kwargs(str(output), OTHER) == {}


@pytest.mark.parametrize("mode,source", [
    ("local", "local"), ("cache-path", "cache_path"),
    ("HF_HUB_OFFLINE", "offline"), ("TRANSFORMERS_OFFLINE", "offline"),
    ("lookup-error", "resolution_failed"), ("invalid-sha", "resolution_failed"),
    ("mlx", "unsupported_backend"), ("unsloth", "unsupported_backend"),
    ("non-hf", "unsupported_backend"), ("distributed", "unsupported_backend"),
])
def test_unknown_and_unsupported_sources_keep_loading_without_a_pin(
    sweep_cli, monkeypatch, mode, source,
):
    if mode in {"local", "cache-path"}:
        path = sweep_cli.root / (f"snapshots/{FIRST}" if mode == "cache-path" else "local")
        path.mkdir(parents=True)
        sweep_cli.state.config["base"] = str(path)
    elif mode in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}:
        monkeypatch.setenv(mode, "1")
    elif mode in {"mlx", "unsloth"}:
        sweep_cli.state.config["backend"] = mode
    elif mode == "non-hf":
        sweep_cli.state.config["training"]["hub"] = "modelscope"
    elif mode == "distributed":
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
    else:
        def model_info(base, **kwargs):
            sweep_cli.state.lookups.append(base)
            if mode == "lookup-error":
                raise OSError("provider unavailable: do not print this")
            return SimpleNamespace(sha="moving-branch")

        monkeypatch.setattr(
            provenance, "get_hf_api", lambda: SimpleNamespace(model_info=model_info),
        )

    result = sweep_cli.invoke(param="lr=0.001", colour=True)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert sweep_cli.state.calls and all(call[3] == {} for call in sweep_cli.state.calls)
    assert sweep_cli.state.lookups == ([BASE] if mode in {"lookup-error", "invalid-sha"} else [])
    row, = sweep_cli.runs()
    expected_revision = FIRST if mode == "cache-path" else None
    assert (row["status"], row["base_revision"], row["base_revision_source"]) == (
        "completed", expected_revision, source,
    )
    output = sweep_cli.root / row["output_dir"]
    metadata = provenance.read_run_provenance(output)
    assert (metadata["base_revision"], metadata["base_revision_source"]) == (
        expected_revision, source,
    )
    assert metadata["base_model"] == sweep_cli.state.config["base"]
    assert metadata["training_data_sha256"] is None and row["training_data_sha256"] is None
    assert provenance.recorded_base_kwargs(str(output), metadata["base_model"]) == {}
    line = ("Base revision: cache-path metadata; local bytes are unverified"
            if mode == "cache-path" else UNKNOWN)
    assert _plain(result.output).count(line) == 1
    assert "provider unavailable" not in result.output
    assert provenance.base_revision_kwargs(BASE) == {}


@pytest.mark.parametrize("phase", ["setup", "train"])
def test_failed_arms_stay_failed_without_a_final_provenance_file(sweep_cli, phase):
    sweep_cli.state.failure = phase
    result = sweep_cli.invoke()

    assert result.exit_code == 1, result.output
    assert f"{phase if phase == 'setup' else 'training'} failed after base load" in _plain(
        result.output,
    )
    rows = sweep_cli.runs()
    assert len(rows) == 2 and all(row["status"] == "failed" for row in rows)
    if phase == "train":
        assert [(row["base_revision"], row["base_revision_source"]) for row in rows] == [
            (FIRST, "hub_commit"), (SECOND, "hub_commit"),
        ]
    assert sweep_cli.state.lookups == [BASE, BASE]
    assert not list(sweep_cli.root.rglob("soup_run.json"))
    assert provenance.base_revision_kwargs(BASE) == {}


def test_dry_run_never_resolves_or_creates_run_identity(sweep_cli):
    result = sweep_cli.invoke(extra=("--dry-run",))

    assert result.exit_code == 0, result.output
    assert sweep_cli.state.lookups == [] and sweep_cli.state.calls == []
    assert sweep_cli.runs() == []
    assert not list(sweep_cli.root.rglob("soup_run.json"))


@pytest.fixture
def cpu_training(monkeypatch):
    torch = pytest.importorskip("torch")
    from transformers import TrainingArguments

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    original_post_init = TrainingArguments.__post_init__

    def cpu_post_init(args):
        args.use_cpu = True
        original_post_init(args)

    monkeypatch.setattr(TrainingArguments, "__post_init__", cpu_post_init)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def test_real_cpu_sft_arm_updates_saves_and_reloads_the_recorded_base(
    sweep_cli, monkeypatch, cpu_training,
):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    pytest.importorskip("trl")
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from soup_cli.trainer import dispatch
    from tests._tiny_hf_models import tiny_model_dir

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    local = tiny_model_dir("llama")
    original_config = AutoConfig.from_pretrained
    original_model = AutoModelForCausalLM.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    receivers = []

    def redirect(factory, kind):
        def load(model_id, *args, **kwargs):
            if model_id == BASE:
                receivers.append((kind, kwargs.pop("revision", None)))
                model_id = local
            return factory(model_id, *args, **kwargs)
        return load

    monkeypatch.setattr(AutoConfig, "from_pretrained", redirect(original_config, "config"))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", redirect(original_model, "model"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", redirect(original_tokenizer, "tokenizer"))
    monkeypatch.setattr(dispatch, "build_trainer", sweep_cli.state.real_build_trainer)
    cfg = sweep_cli.state.config
    cfg["data"]["max_length"] = 128
    cfg["data"]["chat_template"] = "chatml"
    cfg["training"].update({
        "gradient_accumulation_steps": 1, "gradient_checkpointing": False,
        "warmup_ratio": 0,
        "logging_steps": 1, "save_steps": 100, "rope_scaling_type": "linear",
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    })

    result = sweep_cli.invoke(param="lr=0.001")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert {kind for kind, _ in receivers} >= {"model", "tokenizer", "config"}
    assert all(revision == FIRST for _, revision in receivers)
    assert sweep_cli.state.lookups == [BASE]
    row, = sweep_cli.runs()
    assert row["status"] == "completed" and row["total_steps"] == 1
    assert row["base_revision"] == FIRST and row["training_data_sha256"] is None
    output = sweep_cli.root / row["output_dir"]
    assert provenance.recorded_base_kwargs(str(output), BASE) == {"revision": FIRST}
    saved = load_file(str(output / "adapter_model.safetensors"))
    assert saved and all(torch.isfinite(tensor).all() for tensor in saved.values())
    # PEFT initializes LoRA B to zero; saved nonzero B requires a real update.
    assert any(torch.count_nonzero(tensor) > 0 for key, tensor in saved.items() if "lora_B" in key)
    reloaded = peft.PeftModel.from_pretrained(original_model(local), str(output))
    restored = peft.get_peft_model_state_dict(reloaded)
    assert restored.keys() == saved.keys()
    assert all(torch.equal(restored[key], tensor) for key, tensor in saved.items())
    tokenizer = original_tokenizer(str(output))
    inputs = tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        logits = reloaded(**inputs).logits
        with reloaded.disable_adapter():
            base_logits = reloaded(**inputs).logits
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits, base_logits, atol=1e-8, rtol=0)
