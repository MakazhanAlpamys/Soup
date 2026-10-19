"""Exercise real loader/CLI boundaries without Hub downloads or a GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.utils import run_provenance as provenance

BASE = "owner/base"
REVISION = "a" * 40


@pytest.fixture
def moving_hub(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    probe = Mock(side_effect=[SimpleNamespace(sha=REVISION), SimpleNamespace(sha="b" * 40)])
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=probe))
    return probe


def test_actual_sft_loads_pin_tokenizer_model_and_config(tmp_path, monkeypatch, moving_hub):
    pytest.importorskip("torch")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests._tiny_hf_models import tiny_model_dir

    local = tiny_model_dir("llama")
    original_config = AutoConfig.from_pretrained
    original_model = AutoModelForCausalLM.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    calls = []

    def tokenizer(model_id, *, trust_remote_code, revision):
        assert model_id == BASE
        assert revision == REVISION
        calls.append("tokenizer")
        return original_tokenizer(local, trust_remote_code=trust_remote_code)

    def config(model_id, **kwargs):
        if model_id == BASE:
            assert kwargs.pop("revision") == REVISION
            calls.append("config")
            model_id = local
        # AutoTokenizer also reads the redirected local model's config internally.
        return original_config(model_id, **kwargs)

    def model(model_id, *, trust_remote_code, device_map, torch_dtype, config, revision):
        assert model_id == BASE
        assert revision == REVISION
        calls.append("model")
        return original_model(
            local, trust_remote_code=trust_remote_code, device_map=device_map,
            torch_dtype=torch_dtype, config=config,
        )

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(AutoConfig, "from_pretrained", config)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model)
    cfg = SoupConfig(
        base=BASE, data={"train": "unused.jsonl", "max_length": 256},
        training={"quantization": "none", "rope_scaling_type": "linear",
                  "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]}},
        output=str(tmp_path / "out"),
    )
    identity = provenance.BaseRevision(BASE)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    with provenance.base_revision_scope(identity):
        wrapper._setup_transformers(cfg, cfg.training)
    assert calls == ["tokenizer", "config", "model"]
    assert moving_hub.call_count == 1
    assert cfg.base == BASE
    assert wrapper.model is not None


@pytest.mark.parametrize("mode", ["recorded", "implicit", "override", "legacy"])
def test_copied_adapter_pins_live_eval_and_export_loaders(tmp_path, monkeypatch, mode):
    pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.commands.export import _merge_adapter
    from soup_cli.utils.live_eval import load_model_and_tokenizer
    from tests._tiny_hf_models import tiny_model_dir

    local = tiny_model_dir("llama")
    original_model = AutoModelForCausalLM.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    base = original_model(local)
    adapter = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    adapter.peft_config["default"].base_model_name_or_path = BASE
    source = tmp_path / "original-adapter"
    adapter.save_pretrained(source)
    original_tokenizer(local).save_pretrained(source)
    identity = provenance.BaseRevision(BASE)
    identity.revision, identity.source = REVISION, "hub_commit"
    if mode != "legacy":
        provenance.write_run_provenance(source, identity.metadata("deleted-original-run"))
    copied = tmp_path / "copied-adapter"
    shutil.copytree(source, copied)
    shutil.rmtree(source)
    selected_base = "different/base" if mode == "override" else BASE
    expected_revision = REVISION if mode in {"recorded", "implicit"} else None
    calls = []

    def model(model_id, *, revision=None, **kwargs):
        assert model_id == selected_base
        assert revision == expected_revision
        calls.append(("model", revision))
        return original_model(local, **kwargs)

    def tokenizer(model_id, **kwargs):
        if model_id == selected_base:
            assert kwargs.pop("revision", None) == expected_revision
            calls.append(("tokenizer", expected_revision))
            model_id = local
        else:
            assert "revision" not in kwargs
        return original_tokenizer(model_id, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(provenance, "get_hf_api", Mock(side_effect=AssertionError("no lookup")))
    if mode == "implicit":
        loaded, _, device = load_model_and_tokenizer(str(copied), device="cpu")
    else:
        loaded, _, device = load_model_and_tokenizer(
            selected_base, adapter=str(copied), device="cpu",
        )
    assert loaded is not None and str(device) == "cpu"
    merged = tmp_path / "merged"
    _merge_adapter(str(copied), selected_base, str(merged))
    assert (merged / "config.json").is_file()
    expected_calls = [] if mode == "implicit" else [("tokenizer", expected_revision)]
    assert calls == expected_calls + [("model", expected_revision), ("model", expected_revision)]


@pytest.mark.parametrize("mode", ["bound", "offline", "distributed", "non-hf"])
@pytest.mark.parametrize("stored_hash", [None, "c" * 64], ids=["unknown-data", "stored-data"])
def test_train_cli_records_identity_and_writes_final_artifact(
    tmp_path, monkeypatch, moving_hub, mode, stored_hash,
):
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.experiment.tracker import ExperimentTracker

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    if mode == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("soup_cli.utils.launcher.is_in_distributed", lambda: mode == "distributed")
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", "CPU"))
    monkeypatch.setattr("soup_cli.utils.gpu.get_gpu_info", lambda: {})
    alternate_snapshot = None

    def alternate_hub(hub, model_id, *, local_dir):
        nonlocal alternate_snapshot
        assert hub == "modelscope" and model_id == BASE
        alternate_snapshot = local_dir
        directory = Path(local_dir)
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    if mode == "non-hf":
        monkeypatch.setattr("soup_cli.utils.hubs.download_repo", alternate_hub)
    output = tmp_path / "final"
    (tmp_path / "data.jsonl").write_text(
        '{"instruction":"hi", "output":"hello"}\n', encoding="utf-8",
    )
    (tmp_path / "soup.yaml").write_text(
        f"base: {BASE}\ndata: {{train: data.jsonl, format: alpaca, val_split: 0}}\n"
        "training: {epochs: 1, quantization: none, batch_size: 1, "
        f"hub: {'modelscope' if mode == 'non-hf' else 'hf'}}}\noutput: final\n",
        encoding="utf-8",
    )
    expected = {"revision": REVISION} if mode == "bound" else {}

    class Trainer:
        def setup(self, dataset):
            assert dataset["train"]
            assert provenance.base_revision_kwargs(BASE) == expected

        def train(self, **kwargs):
            assert provenance.base_revision_kwargs(BASE) == expected
            # Emulate a separate producer of the reserved historical hash field.
            conn = kwargs["tracker"]._get_conn()
            conn.execute(
                "UPDATE runs SET training_data_sha256=? WHERE run_id=?",
                (stored_hash, kwargs["run_id"]),
            )
            conn.commit()
            output.mkdir()
            return {"output_dir": str(output), "initial_loss": 1.0, "final_loss": 0.5,
                    "total_steps": 1, "duration_secs": 1.0}

    def build(cfg, **kwargs):
        assert cfg.base == (alternate_snapshot if mode == "non-hf" else BASE)
        return Trainer()

    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", build)
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])
    assert result.exit_code == 0, (result.output, result.exception)
    tracker = ExperimentTracker(tmp_path / "runs.db")
    row = tracker.list_runs()[0]
    assert row["status"] == "completed"
    assert row["base_model"] == BASE
    assert json.loads(row["config_json"])["base"] == BASE
    assert row["base_revision"] == (REVISION if mode == "bound" else None)
    assert row["base_revision_source"] == {
        "bound": "hub_commit", "offline": "offline", "distributed": "unsupported_backend",
        "non-hf": "unsupported_backend",
    }[mode]
    metadata = provenance.read_run_provenance(output)
    assert metadata["run_id"] == row["run_id"]
    assert metadata["base_revision"] == row["base_revision"]
    assert metadata["training_data_sha256"] == stored_hash
    assert moving_hub.call_count == (1 if mode == "bound" else 0)
    assert provenance.base_revision_kwargs(BASE) == {}
    tracker.close()
