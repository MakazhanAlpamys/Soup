"""Native training pins, nullable provenance and imported-provider boundaries.

Portable cases replace native computation/download boundaries, not the train
command, dispatch scopes, data loading, SQLite or provenance implementation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console

from soup_cli.utils import mlx as mlx_utils
from soup_cli.utils import run_provenance as provenance
from tests.conftest import make_cli_runner, strip_ansi

BASE = "owner/base"
FIRST = "a" * 40
SECOND = "b" * 40


def _plain(text):
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def native_boundary(monkeypatch):
    import huggingface_hub

    state = SimpleNamespace(loads=[], lookups=[], failure=None)

    def model_info(base, **kwargs):
        state.lookups.append(base)
        if state.failure == "lookup":
            raise OSError("private provider exception must not be printed")
        return SimpleNamespace(sha="main" if state.failure == "sha" else (
            FIRST if len(state.lookups) == 1 else SECOND
        ))

    def load(base, **kwargs):
        state.loads.append((base, kwargs))
        if state.failure == "load":
            raise OSError("pinned snapshot unavailable")
        return ("model", "tokenizer", kwargs.get("model_config")) if kwargs.get(
            "return_config"
        ) else ("model", "tokenizer")

    upstream = SimpleNamespace(load=load, snapshot_download=huggingface_hub.snapshot_download)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load, utils=upstream))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", upstream)
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=model_info))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MLXLM_USE_MODELSCOPE",
                "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "MLX_RANK",
                "MLX_HOSTFILE", "MLX_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE",
                "OMPI_COMM_WORLD_RANK", "PMI_SIZE", "PMI_RANK", "PMIX_RANK",
                "ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_DEEPSPEED",
                "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(key, raising=False)
    state.upstream = upstream
    return state


@pytest.fixture
def train_cli(tmp_path, monkeypatch, native_boundary):
    import accelerate

    from soup_cli.cli import app
    from soup_cli.commands import train
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.trainer import dispatch
    from soup_cli.utils.mlx_rope import resolve_mlx_rope_scaling

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(train, "detect_device", lambda backend=None: ("cpu", "CPU"))
    monkeypatch.setattr(train, "get_gpu_info", lambda backend=None: {"memory_total": "N/A"})

    def forbidden_group(*args, **kwargs):
        pytest.fail("MLX provenance must not initialize an Accelerate group")

    monkeypatch.setattr(accelerate, "PartialState", forbidden_group)
    state = native_boundary
    state.config = {
        "base": BASE, "backend": "mlx", "output": "final",
        "data": {"train": "rows.jsonl", "format": "alpaca", "val_split": 0},
        "training": {"epochs": 1, "batch_size": 1, "quantization": "none"},
    }
    state.rope = None
    state.stages = []
    source = tmp_path / "rows.jsonl"
    source.write_text('{"instruction":"hello", "output":"world"}\n', encoding="utf-8")

    class Trainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def setup(self, dataset):
            assert dataset["train"]
            override = None
            if state.rope:
                _, override = resolve_mlx_rope_scaling(self.cfg.base, 128, "linear")
            state.stages.append(("setup", mlx_utils.load_mlx_model(
                self.cfg.base, model_config=override, return_config=True,
            )))

        def train(self, **kwargs):
            state.stages.append(("train", mlx_utils.load_mlx_model(self.cfg.base)))
            mlx_utils.load_mlx_model("other/teacher")
            if state.failure == "train":
                raise RuntimeError("training failed after setup")
            Path(self.cfg.output).mkdir(exist_ok=True)
            return {"output_dir": self.cfg.output, "initial_loss": 1.0, "final_loss": 0.5,
                    "total_steps": 1, "duration_secs": 1.0}

    monkeypatch.setattr(dispatch, "build_trainer", lambda cfg, **kwargs: Trainer(cfg))

    def invoke(*extra, colour=False):
        monkeypatch.setattr(train, "console", Console(width=500, force_terminal=colour))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(state.config), encoding="utf-8")
        return make_cli_runner().invoke(app, [
            "--no-telemetry", "--no-audit-log", "train", "--config", "soup.yaml",
            "--yes", *extra,
        ])

    def runs():
        tracker = ExperimentTracker(tmp_path / "runs.db")
        try:
            return tracker.list_runs()
        finally:
            tracker.close()

    return SimpleNamespace(invoke=invoke, runs=runs, state=state, root=tmp_path, source=source)


@pytest.mark.parametrize("colour", [False, True])
def test_train_pins_actual_receivers_and_records_one_identity(train_cli, colour):
    result = train_cli.invoke(colour=colour)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    state = train_cli.state
    assert state.lookups == [BASE]
    assert state.loads == [
        (BASE, {"revision": FIRST, "return_config": True}),
        (BASE, {"revision": FIRST}), ("other/teacher", {}),
    ]
    row, = train_cli.runs()
    assert (row["status"], row["base_model"], row["base_revision"],
            row["base_revision_source"]) == (
        "completed", BASE, FIRST, "hub_commit",
    )
    assert json.loads(row["config_json"])["base"] == BASE
    metadata = provenance.read_run_provenance(train_cli.root / "final")
    assert metadata == {
        "schema_version": 1, "run_id": row["run_id"], "base_model": BASE,
        "base_revision": FIRST, "base_revision_source": "hub_commit",
        "training_data_sha256": hashlib.sha256(train_cli.source.read_bytes()).hexdigest(),
    }
    assert row["training_data_sha256"] == metadata["training_data_sha256"]
    assert _plain(result.output).count(f"Base revision: {FIRST} (bound Hub commit)") == 1
    assert provenance.base_revision_kwargs(BASE) == {}


@pytest.mark.parametrize("cache_hit", [True, False])
def test_rope_first_read_and_native_download_share_the_same_commit(
    train_cli, monkeypatch, cache_hit,
):
    import huggingface_hub

    downloads = []
    state = train_cli.state
    state.rope = True
    old = train_cli.root / "old.json"
    new = train_cli.root / "new.json"
    old.write_text('{"max_position_embeddings": 64}', encoding="utf-8")
    new.write_text('{"max_position_embeddings": 32}', encoding="utf-8")

    def download(*, repo_id, filename, **kwargs):
        assert repo_id == BASE and filename == "config.json"
        downloads.append(kwargs)
        if kwargs.get("local_files_only") and not cache_hit:
            raise FileNotFoundError("cold cache")
        return str(old if kwargs.get("revision") == FIRST else new)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = train_cli.invoke()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert downloads == [{"local_files_only": True, "revision": FIRST}] + (
        [] if cache_hit else [{"revision": FIRST}]
    )
    assert state.lookups == [BASE]
    assert state.loads[0] == (BASE, {
        "revision": FIRST, "return_config": True,
        "model_config": {"rope_scaling": {"type": "linear", "factor": 2.0},
                         "max_position_embeddings": 128},
    })


@pytest.mark.parametrize("mode,source", [
    ("offline", "offline"), ("lookup", "resolution_failed"), ("sha", "resolution_failed"),
    ("local", "local"), ("cache", "cache_path"), ("modelscope", "unsupported_backend"),
    ("wrapped-loader", "unsupported_backend"), ("world", "unsupported_backend"),
    ("malformed-world", "unsupported_backend"), ("native-launch", "unsupported_backend"),
])
def test_unqualified_sources_keep_loading_and_record_nullable_identity(
    train_cli, monkeypatch, mode, source,
):
    state = train_cli.state
    if mode in {"lookup", "sha"}:
        state.failure = mode
    elif mode == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    elif mode in {"local", "cache"}:
        path = train_cli.root / (f"snapshots/{FIRST}" if mode == "cache" else "local")
        path.mkdir(parents=True)
        state.config["base"] = str(path)
    elif mode == "modelscope":
        # Provider was chosen at import time; clearing the current env cannot
        # turn this already-imported receiver into an HF download function.
        state.upstream.snapshot_download = lambda *a, **kw: None
        monkeypatch.delenv("MLXLM_USE_MODELSCOPE", raising=False)
    elif mode == "wrapped-loader":
        state.upstream.load = lambda *a, **kw: None
    elif mode in {"world", "malformed-world"}:
        monkeypatch.setenv("WORLD_SIZE", "2" if mode == "world" else "bad")
    elif mode == "native-launch":
        monkeypatch.setenv("MLX_HOSTFILE", "hosts.json")
        monkeypatch.setenv("MLX_RANK", "0")
    result = train_cli.invoke(colour=True)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert state.loads and all("revision" not in kwargs for _, kwargs in state.loads)
    assert state.lookups == ([BASE] if mode in {"lookup", "sha"} else [])
    row, = train_cli.runs()
    revision = FIRST if mode == "cache" else None
    assert (row["status"], row["base_revision"], row["base_revision_source"]) == (
        "completed", revision, source,
    )
    metadata = provenance.read_run_provenance(train_cli.root / "final")
    assert metadata["base_revision"] == revision
    assert metadata["base_revision_source"] == source
    if mode in {"modelscope", "wrapped-loader"}:
        assert "MLX loader provider is not qualified as Hugging Face Hub" in _plain(result.output)
    assert "private provider exception" not in result.output


def test_imported_hf_provider_stays_qualified_after_environment_change(train_cli, monkeypatch):
    monkeypatch.setenv("MLXLM_USE_MODELSCOPE", "1")
    result = train_cli.invoke()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert train_cli.runs()[0]["base_revision"] == FIRST
    assert train_cli.state.lookups == [BASE]


def test_two_runs_resolve_independently_without_scope_leaks(train_cli):
    for output, revision in (("first", FIRST), ("second", SECOND)):
        train_cli.state.config["output"] = output
        result = train_cli.invoke()
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert provenance.read_run_provenance(train_cli.root / output)["base_revision"] == revision
        assert provenance.base_revision_kwargs(BASE) == {}
    assert train_cli.state.lookups == [BASE, BASE]
    assert [kwargs["revision"] for base, kwargs in train_cli.state.loads if base == BASE] == [
        FIRST, FIRST, SECOND, SECOND,
    ]


@pytest.mark.parametrize("failure", ["load", "train"])
def test_failure_does_not_retry_head_or_write_completed_artifact(train_cli, failure):
    train_cli.state.failure = failure
    result = train_cli.invoke()
    assert result.exit_code != 0
    row, = train_cli.runs()
    assert row["status"] == "failed"
    # Existing lifecycle records after successful setup, never fabricating a
    # completed load receipt when setup itself failed.
    assert row["base_revision"] == (FIRST if failure == "train" else None)
    assert not (train_cli.root / "final" / "soup_run.json").exists()
    assert train_cli.state.lookups == [BASE]
    assert all(kwargs["revision"] == FIRST
               for base, kwargs in train_cli.state.loads if base == BASE)
    assert provenance.base_revision_kwargs(BASE) == {}


def test_dry_run_never_imports_native_provider_or_resolves_identity(train_cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    result = train_cli.invoke("--dry-run")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert train_cli.state.loads == train_cli.state.lookups == []
    assert train_cli.runs() == []


@pytest.mark.parametrize("task,loss", [
    ("sft", None), ("dpo", None), ("orpo", None), ("simpo", None), ("kto", None),
    ("grpo", None), ("preference", "dpo"), ("preference", "orpo"), ("preference", "simpo"),
])
def test_real_task_setup_receives_active_revision(native_boundary, monkeypatch, task, loss):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.dispatch import build_trainer

    cfg = SoupConfig(base=BASE, backend="mlx", task=task,
                     data={"train": "unused.jsonl", "val_split": 0},
                     training={"batch_size": 2, "quantization": "none",
                               **({"preference_loss": loss} if loss else {})})
    wrapper = build_trainer(cfg, device="cpu")
    monkeypatch.setattr(wrapper, "_require_mlx", lambda: None)
    monkeypatch.setattr(wrapper, "_check_unsupported", lambda: None)
    if task == "grpo":
        monkeypatch.setattr(wrapper, "_prepare", lambda dataset: None)

    class LoadedError(Exception):
        pass

    def stop(base, **kwargs):
        assert base == BASE
        assert kwargs.get("revision") == FIRST
        raise LoadedError

    monkeypatch.setattr(sys.modules["mlx_lm"], "load", stop)
    with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
        with pytest.raises(LoadedError):
            wrapper.setup({"train": [{"text": "hello world"}] * 2, "val": []})
    assert native_boundary.lookups == [BASE]


def test_unscoped_and_unrelated_loads_do_not_resolve_or_pin(native_boundary):
    mlx_utils.load_mlx_model(BASE)
    with provenance.base_revision_scope(provenance.BaseRevision("other/base")):
        mlx_utils.load_mlx_model(BASE)
    assert native_boundary.loads == [(BASE, {}), (BASE, {})]
    assert native_boundary.lookups == []


def test_rope_failed_pinned_download_never_falls_back_to_head(train_cli, monkeypatch):
    import huggingface_hub

    calls = []

    def fail(*, repo_id, filename, **kwargs):
        calls.append(kwargs)
        raise OSError("missing pinned config")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    train_cli.state.rope = True
    result = train_cli.invoke()
    assert result.exit_code != 0
    assert calls == [{"local_files_only": True, "revision": FIRST}, {"revision": FIRST}]
    assert train_cli.state.lookups == [BASE]
    assert train_cli.state.loads == []
    assert train_cli.runs()[0]["status"] == "failed"


@pytest.mark.parametrize("key,value", [
    ("WORLD_SIZE", ""), ("WORLD_SIZE", "0"), ("LOCAL_WORLD_SIZE", "2"),
    ("MLX_WORLD_SIZE", "2"), ("OMPI_COMM_WORLD_SIZE", "2"), ("PMI_SIZE", "bad"),
    ("LOCAL_RANK", "0"), ("MLX_RANK", "0"), ("OMPI_COMM_WORLD_RANK", "0"),
    ("PMI_RANK", "0"), ("PMIX_RANK", "0"), ("ACCELERATE_MIXED_PRECISION", "no"),
])
def test_launcher_qualification_never_initializes_a_process_group(
    native_boundary, monkeypatch, key, value,
):
    import accelerate

    def forbidden(*args, **kwargs):
        pytest.fail("MLX metadata must not initialize an Accelerate group")

    monkeypatch.setattr(accelerate, "PartialState", forbidden)
    monkeypatch.setenv(key, value)
    assert mlx_utils.mlx_revision_binding_reason() is not None
    assert native_boundary.lookups == []


def test_single_process_world_size_one_keeps_binding(train_cli, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    result = train_cli.invoke()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert train_cli.runs()[0]["base_revision"] == FIRST


def test_alternate_hub_config_keeps_its_existing_refusal(train_cli):
    train_cli.state.config["training"]["hub"] = "modelscope"
    result = train_cli.invoke()
    assert result.exit_code == 1
    assert "hub='modelscope' is not supported on backend=mlx" in _plain(result.output)
    assert train_cli.state.lookups == train_cli.state.loads == []
    assert train_cli.runs() == []


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
@pytest.mark.parametrize("cached", [False, True], ids=["local", "cache-hint"])
@pytest.mark.parametrize("provider", ["hf", "modelscope", "launcher"])
def test_local_base_diagnostic_does_not_blame_an_unused_provider(
    train_cli, monkeypatch, colour, cached, provider,
):
    state = train_cli.state
    if provider == "modelscope":
        state.upstream.snapshot_download = lambda *args, **kwargs: None
    elif provider == "launcher":
        monkeypatch.setenv("WORLD_SIZE", "bad")
    base = train_cli.root / (f"snapshots/{FIRST}" if cached else "local-base")
    base.mkdir(parents=True)
    state.config["base"] = str(base)

    result = train_cli.invoke(colour=colour)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    plain = _plain(result.output)
    assert "not qualified" not in plain
    expected = (
        "Base revision: cache-path metadata; local bytes are unverified" if cached else
        "Base revision: unknown (existing loading behavior preserved)"
    )
    assert plain.count(expected) == 1
    assert state.lookups == []
    assert state.loads == [
        (str(base), {"return_config": True}), (str(base), {}), ("other/teacher", {}),
    ]
    row, = train_cli.runs()
    source, revision = ("cache_path", FIRST) if cached else ("local", None)
    assert (row["base_model"], row["base_revision_source"], row["base_revision"]) == (
        str(base), source, revision,
    )
    metadata = provenance.read_run_provenance(train_cli.root / "final")
    assert metadata["base_revision_source"] == source
    assert metadata["base_revision"] == revision
    assert provenance.base_revision_kwargs(str(base)) == {}
