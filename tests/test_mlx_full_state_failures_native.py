"""Real bundle corruption and interrupted publication without fabricated optimizer state."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.trainer.mlx_checkpoint import (  # noqa: E402
    FullStateSession,
    _hash,
    discover_bundle,
    read_bundle,
)
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402
from tests.test_mlx_dora_native import _checkpoint, _config, _rows  # noqa: E402


@pytest.fixture
def saved(tmp_path, monkeypatch):
    cfg = _config(_checkpoint(tmp_path / "base"), tmp_path / "out", dora=False)
    cfg.training.mlx_full_state = True
    cfg.training.rewind_log = False
    cfg.training.save_steps = 1
    captured = []
    original = FullStateSession.save

    def stop(session, micro):
        original(session, micro)
        if micro == 2:
            captured.append(session)
            raise KeyboardInterrupt("test interruption")

    with monkeypatch.context() as patch:
        patch.setattr(FullStateSession, "save", stop)
        wrapper = MLXSFTTrainerWrapper(cfg)
        wrapper.setup(_rows())
        with pytest.raises(KeyboardInterrupt, match="test interruption"):
            wrapper.train()
    return cfg, captured[0], Path(cfg.output) / "mlx-state-2"


def prepare_session(cfg, rows, bundle, monkeypatch):
    class CaptureError(Exception):
        pass

    captured = []

    def capture(session):
        captured.append(session)
        raise CaptureError()

    with monkeypatch.context() as patch:
        patch.setattr(FullStateSession, "prepare", capture)
        wrapper = MLXSFTTrainerWrapper(cfg)
        wrapper.setup(rows)
        with pytest.raises(CaptureError):
            wrapper.train(resume_from_checkpoint=str(bundle))
    return captured[0]


def rewrite_member(bundle, name):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["members"][name] = {
        "size": (bundle / name).stat().st_size,
        "sha256": _hash(bundle / name),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("collision", ["same-contract", "different-contract", "older-in-place"])
def test_resume_refuses_populated_output_collisions(saved, monkeypatch, tmp_path, collision):
    import shutil

    cfg, _, bundle = saved
    # Both an in-place continuation and an external continuation into fresh
    # output restore the actual saved optimizer before testing the refusal.
    control = prepare_session(cfg, _rows(), bundle, monkeypatch)
    control.prepare()
    assert control.optimizer.step.item() == 2
    fresh = cfg.model_copy(deep=True)
    fresh.output = str(tmp_path / "fresh")
    control = prepare_session(fresh, _rows(), bundle, monkeypatch)
    control.prepare()
    assert control.optimizer.step.item() == 2

    changed = cfg.model_copy(deep=True)
    if collision == "older-in-place":
        source = bundle.parent / "mlx-state-1"
    else:
        source = bundle
        changed.output = str(tmp_path / "populated")
        shutil.copytree(cfg.output, changed.output)
        if collision == "different-contract":
            path = Path(changed.output) / "mlx-state-2" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["contract"]["recipe"]["training"]["lr"] *= 2
            path.write_text(json.dumps(manifest))
    before = {
        str(p.relative_to(changed.output)): _hash(p)
        for p in Path(changed.output).rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="populated output.*latest checkpoint"):
        prepare_session(changed, _rows(), source, monkeypatch)
    after = {
        str(p.relative_to(changed.output)): _hash(p)
        for p in Path(changed.output).rglob("*")
        if p.is_file()
    }
    assert before == after


def test_save_refuses_replaced_published_checkpoint(saved):
    _, session, bundle = saved
    # Repeated publication of the same completed boundary is an ordinary final-save path.
    session.save(2)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["contract"]["recipe"]["training"]["lr"] *= 2
    path.write_text(json.dumps(manifest))
    before = {p.name: _hash(p) for p in bundle.iterdir()}
    with pytest.raises(ValueError, match="output collision"):
        session.save(2)
    assert before == {p.name: _hash(p) for p in bundle.iterdir()}


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("data", "train"),
        ("rank", "recipe"),
        ("lr", "recipe"),
        ("budget", "recipe"),
        ("scheduler", "recipe"),
        ("optimizer", "recipe"),
        ("tokenizer", "base_files"),
        ("base", "base_files"),
        ("runtime", "runtime"),
        ("cursor", "sampler/progress"),
        ("numpy", "NumPy random"),
        ("python", "Python random"),
        ("state-type", "random/sampler"),
        ("adapter-metadata", "adapter metadata"),
        ("adapter-keys", "adapter tensor header"),
        ("adapter-shape", "adapter tensor header"),
        ("adapter-dtype", "adapter tensor header"),
        ("adapter-nan", "Nonfinite"),
        ("optimizer-counter", "counter/progress"),
        ("random-keys", "random tensor header"),
    ],
)
def test_refusal_precedes_checkpoint_application(saved, monkeypatch, change, diagnostic):
    cfg, _, bundle = saved
    rows = _rows()
    # Adjacent valid state really restores before introducing exactly one mismatch.
    control = prepare_session(cfg, rows, bundle, monkeypatch)
    control.prepare()
    assert control.optimizer.step.item() == 2
    changed = SoupConfig(**cfg.model_dump())
    if change == "data":
        rows["train"][0] = {"text": "gamma beta alpha gamma beta alpha"}
    elif change in {"rank", "lr", "budget", "scheduler", "optimizer"}:
        if change == "rank":
            changed.training.lora.r = 2
        elif change == "budget":
            changed.training.epochs = 2
        elif change == "scheduler":
            changed.training.scheduler = "linear"
        elif change == "optimizer":
            changed.training.optimizer = "adamw_hf"
        else:
            changed.training.lr *= 2
    elif change in {"tokenizer", "base"}:
        path = Path(cfg.base) / (
            "tokenizer_config.json" if change == "tokenizer" else "config.json"
        )
        # Harmless JSON whitespace still identifies a different pinned snapshot.
        path.write_text(path.read_text() + "\n")
    elif change == "runtime":
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["contract"]["runtime"]["mlx"] = "incompatible"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
    elif change in {"cursor", "numpy", "python", "state-type"}:
        path = bundle / "state.json"
        state = json.loads(path.read_text())
        if change == "cursor":
            state["sampler"]["position"] -= 1
        elif change == "numpy":
            state["numpy"][2] = True
        elif change == "python":
            state["python"] = [99, [], None]
        else:
            state = []
        path.write_text(json.dumps(state))
        rewrite_member(bundle, path.name)
    elif change == "adapter-metadata":
        metadata = json.loads((bundle / "adapter_config.json").read_text())
        metadata["lora_parameters"]["rank"] += 1
        (bundle / "adapter_config.json").write_text(json.dumps(metadata))
        rewrite_member(bundle, "adapter_config.json")
    else:
        name = (
            "optimizer"
            if change.startswith("optimizer")
            else "random"
            if change.startswith("random")
            else "adapters"
        ) + ".safetensors"
        tensors = mx.load(str(bundle / name))
        key = next(iter(tensors))
        if change.endswith("keys"):
            tensors["unexpected"] = tensors.pop(key)
        elif change.endswith("shape"):
            tensors[key] = mx.zeros((1,), dtype=tensors[key].dtype)
        elif change.endswith("dtype"):
            tensors[key] = tensors[key].astype(mx.float16)
        elif change.endswith("nan"):
            tensors[key] = mx.full(tensors[key].shape, float("nan"), dtype=tensors[key].dtype)
        else:
            tensors["step"] = mx.array(0, dtype=tensors["step"].dtype)
        # MLX loads can map the original file. Never overwrite their backing
        # inode while serialization still reads the remaining tensor payloads.
        replacement = bundle / "replacement.safetensors"
        mx.save_safetensors(str(replacement), tensors)
        replacement.replace(bundle / name)
        rewrite_member(bundle, name)
    session = prepare_session(changed, rows, bundle, monkeypatch)
    before = {k: mx.array(v) for k, v in tree_flatten(session.model.trainable_parameters())}
    mx.eval(before)
    with pytest.raises(ValueError, match=diagnostic):
        session.prepare()
    actual = dict(tree_flatten(session.model.trainable_parameters()))
    assert all(mx.array_equal(v, actual[k]).item() for k, v in before.items())
    assert session.optimizer.step.item() == 0


@pytest.mark.parametrize(
    "stage",
    [
        "adapters.safetensors",
        "optimizer.safetensors",
        "random.safetensors",
        "state.json",
        "adapter_config.json",
        "manifest.json",
        "publish",
    ],
)
def test_interrupted_write_keeps_last_committed_bundle(saved, monkeypatch, stage):
    import shutil

    import soup_cli.trainer.mlx_checkpoint as checkpoint

    cfg, session, bundle = saved
    previous = Path(cfg.output) / "mlx-state-1"
    hashes = {p.name: _hash(p) for p in previous.iterdir()}
    shutil.rmtree(bundle)  # Only the test-owned later snapshot is removed.
    original_tensor = mx.save_safetensors
    original_text = Path.write_text
    original_rename = checkpoint.os.rename

    def tensor(path, *args, **kwargs):
        if Path(path).name == stage:
            raise OSError("injected disk write failure")
        return original_tensor(path, *args, **kwargs)

    def text(path, *args, **kwargs):
        if path.name == stage:
            raise OSError("injected disk write failure")
        return original_text(path, *args, **kwargs)

    def rename(*args):
        if stage == "publish":
            raise OSError("injected disk write failure")
        return original_rename(*args)

    with monkeypatch.context() as patch:
        patch.setattr(mx, "save_safetensors", tensor)
        patch.setattr(Path, "write_text", text)
        patch.setattr(checkpoint.os, "rename", rename)
        with pytest.raises(OSError, match="injected disk write"):
            session.save(2)
    assert discover_bundle(cfg.output) == previous
    assert read_bundle(previous)["update"] == 1
    assert hashes == {p.name: _hash(p) for p in previous.iterdir()}
    assert not list(Path(cfg.output).glob(".mlx-state-*"))
    session.save(2)
    assert read_bundle(bundle)["update"] == 2
