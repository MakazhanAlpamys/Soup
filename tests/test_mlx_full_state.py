"""Full-state MLX checkpoint discovery and config boundaries (#1977)."""

import json

import pytest

from soup_cli.commands.train import _resolve_checkpoint
from soup_cli.config.schema import SoupConfig


def config(**training):
    return SoupConfig(
        base="local-base",
        backend="mlx",
        data={"train": "data.jsonl"},
        training={"quantization": "none", "mlx_full_state": True, **training},
    )


def test_full_state_is_explicitly_enabled():
    assert config().training.mlx_full_state is True
    assert SoupConfig(base="b", data={"train": "d"}).training.mlx_full_state is False


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ({"optimizer": "sgd"}, "AdamW"),
        ({"lora": {"use_dora": True}}, "ordinary linear LoRA"),
        ({"lora": {"use_rslora": True}}, "ordinary linear LoRA"),
        ({"loss_watchdog": True}, "mlx_full_state.*loss_watchdog history"),
    ],
)
def test_exact_full_state_scope_and_adjacent_fresh_controls(change, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        config(**change)
    assert config(mlx_full_state=False, **change).training.mlx_full_state is False


def test_full_state_requires_native_sft():
    with pytest.raises(ValueError, match="MLX text SFT"):
        SoupConfig(base="b", data={"train": "d"}, training={"mlx_full_state": True})


def test_implicit_full_state_resume_checks_watchdog_scope():
    from soup_cli.trainer.mlx_checkpoint import validate_config

    cfg = config(mlx_full_state=False, loss_watchdog=True)
    with pytest.raises(ValueError, match="mlx_full_state.*loss_watchdog history"):
        validate_config(cfg)


def test_full_state_auto_prefers_numeric_committed_bundle(tmp_path):
    for name in ["mlx-state-99", "mlx-state-100"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text("{}")
    (tmp_path / ".mlx-state-101-staging").mkdir()
    (tmp_path / "9999999_adapters.safetensors").write_bytes(b"legacy")
    assert _resolve_checkpoint("auto", str(tmp_path), backend="mlx") == str(
        tmp_path / "mlx-state-100"
    )
    assert _resolve_checkpoint(str(tmp_path / "mlx-state-99"), "", backend="mlx") == str(
        tmp_path / "mlx-state-99"
    )


def test_corrupt_committed_bundle_is_not_legacy_fallback(tmp_path):
    (tmp_path / "mlx-state-100").mkdir()
    (tmp_path / "adapters.safetensors").write_bytes(b"legacy")
    assert _resolve_checkpoint("auto", str(tmp_path), backend="mlx") == str(
        tmp_path / "mlx-state-100"
    )
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    with pytest.raises(ValueError, match="manifest"):
        read_bundle(tmp_path / "mlx-state-100")


def test_bundle_manifest_version_and_progress_are_strict(tmp_path):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    bundle = tmp_path / "mlx-state-1"
    bundle.mkdir()
    manifest = {"format": "soup-mlx-state", "version": True}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        read_bundle(bundle)


def _bundle(tmp_path, update=1):
    from soup_cli.trainer.mlx_checkpoint import _hash

    path = tmp_path / f"mlx-state-{update}"
    path.mkdir()
    names = [
        "adapters.safetensors",
        "optimizer.safetensors",
        "random.safetensors",
        "state.json",
        "adapter_config.json",
    ]
    for name in names:
        (path / name).write_bytes(b"validation-fixture")
    manifest = {
        "format": "soup-mlx-state",
        "version": 1,
        "update": update,
        "microbatch": update * 2,
        "accumulation": 2,
        "total_updates": 100,
        "contract": {},
        "members": {
            n: {"size": (path / n).stat().st_size, "sha256": _hash(path / n)} for n in names
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path, manifest


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", False),
        ("update", True),
        ("update", 0),
        ("microbatch", -1),
        ("microbatch", 3),
        ("total_updates", 0),
        ("accumulation", "2"),
        ("contract", []),
    ],
)
def test_manifest_refusals_have_valid_adjacent_control(tmp_path, field, value):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    path, manifest = _bundle(tmp_path)
    assert read_bundle(path)["update"] == 1
    manifest[field] = value
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="MLX checkpoint"):
        read_bundle(path)


@pytest.mark.parametrize("kind", ["missing", "truncated", "checksum", "traversal"])
def test_bundle_integrity_and_containment(tmp_path, kind):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    path, manifest = _bundle(tmp_path)
    assert read_bundle(path)["update"] == 1
    member = path / "optimizer.safetensors"
    if kind == "missing":
        member.unlink()
    elif kind == "truncated":
        member.write_bytes(b"bad")
    elif kind == "checksum":
        member.write_bytes(b"VALIDATION-fixture")
    else:
        manifest["members"]["../external"] = manifest["members"].pop("optimizer.safetensors")
        (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint"):
        read_bundle(path)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("inside", [False, True])
def test_member_symlinks_are_refused_inside_and_outside_bundle(tmp_path, inside):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    path, _ = _bundle(tmp_path)
    assert read_bundle(path)["update"] == 1
    member = path / "optimizer.safetensors"
    external = (path if inside else tmp_path) / "external"
    external.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(external)
    with pytest.raises(ValueError, match="member path"):
        read_bundle(path)


@pytest.mark.requires_symlink
def test_staged_and_symlinked_bundle_is_never_resumable(tmp_path):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    path, _ = _bundle(tmp_path)
    staged = tmp_path / ".mlx-state-staging-uncommitted"
    path.rename(staged)
    with pytest.raises(ValueError, match="Uncommitted"):
        read_bundle(staged)
    assert _resolve_checkpoint("auto", str(tmp_path), backend="mlx") is None
    link = tmp_path / "mlx-state-1"
    link.symlink_to(staged, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        read_bundle(link)


def test_sampler_roundtrip_after_multiple_epochs_preserves_next_batches():
    from soup_cli.trainer.mlx_checkpoint import BatchCursor

    dataset = [([1] * n, 0) for n in [9, 1, 7, 4, 3, 8]]
    original = BatchCursor(dataset, 2, 0)
    seen = [original.next() for _ in range(8)]
    restored = BatchCursor(dataset, 2, 0, original.state())
    assert len(seen) == 8 and original.epoch == 3
    assert [original.next() for _ in range(12)] == [restored.next() for _ in range(12)]


@pytest.mark.parametrize(
    "field,value", [("position", True), ("position", 99), ("order", [0, 0]), ("epoch", 0)]
)
def test_sampler_state_validation(tmp_path, field, value):
    from soup_cli.trainer.mlx_checkpoint import BatchCursor

    data = [([1, 2], 0)] * 4
    cursor = BatchCursor(data, 2, 7)
    cursor.next()
    state = cursor.state()
    state[field] = value
    with pytest.raises(ValueError, match="sampler"):
        BatchCursor(data, 2, 7, state)


def test_dataset_fingerprint_preserves_content_and_order():
    from soup_cli.trainer.mlx_checkpoint import _dataset_digest

    rows = [([1, 2, 3], [0, 1, 1]), ([4, 5], [0, 1])]
    assert _dataset_digest(rows) == _dataset_digest(list(rows))
    assert _dataset_digest(rows) != _dataset_digest(rows[::-1])
    assert _dataset_digest(rows) != _dataset_digest([rows[0], ([4, 6], [0, 1])])
    assert _dataset_digest(None) == _dataset_digest([])


def test_checkpoint_cannot_exceed_original_budget(tmp_path):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    path, manifest = _bundle(tmp_path, update=2)
    assert read_bundle(path)["update"] == 2
    manifest["total_updates"] = 1
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="progress/budget"):
        read_bundle(path)


@pytest.mark.parametrize("corruption", ["version", "word", "negative", "cursor", "nan", "inf"])
def test_python_random_state_refusals_and_advanced_control(corruption):
    import random

    from soup_cli.trainer.mlx_checkpoint import _python_state

    rng = random.Random(7)
    rng.gauss(0, 1)  # Retain an advanced Gaussian cache as well as MT state.
    state = json.loads(json.dumps(rng.getstate()))
    restored = random.Random()
    restored.setstate(_python_state(state))
    assert restored.gauss(0, 1) == rng.gauss(0, 1)
    assert restored.random() == rng.random()
    if corruption == "version":
        state[0] = True
    elif corruption == "word":
        state[1][0] = True
    elif corruption == "negative":
        state[1][0] = -1
    elif corruption == "cursor":
        state[1][-1] = 625
    else:
        state[2] = float(corruption)
    with pytest.raises(ValueError, match="Python random state"):
        _python_state(state)


def test_full_state_hub_base_refused_before_dependency_or_model_loading(tmp_path, monkeypatch):
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    seen = []
    monkeypatch.setattr(MLXSFTTrainerWrapper, '_require_mlx', lambda self: seen.append('backend'))
    from soup_cli.utils import mlx

    def load(*args, **kwargs):
        seen.append('model')
        return object(), object()

    monkeypatch.setattr(mlx, 'load_mlx_model', load)
    cfg = config()
    cfg.base = 'owner/base'
    with pytest.raises(ValueError, match='require a local pinned base snapshot'):
        MLXSFTTrainerWrapper(cfg).setup({'train': [{'text': 'alpha beta'}]})
    assert seen == []
    local = tmp_path / 'pinned-base'
    local.mkdir()
    cfg.base = str(local)
    MLXSFTTrainerWrapper(cfg).setup({'train': [{'text': 'alpha beta'}]})
    assert seen == ['backend', 'model']
    seen.clear()
    cfg.base = 'owner/base'
    cfg.training.mlx_full_state = False
    MLXSFTTrainerWrapper(cfg).setup({'train': [{'text': 'alpha beta'}]})
    assert seen == ['backend', 'model']



def test_missing_full_state_bundle_diagnostic(tmp_path):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    with pytest.raises(ValueError, match='not found'):
        read_bundle(tmp_path / 'mlx-state-7')


@pytest.mark.requires_symlink
def test_dangling_full_state_bundle_is_still_a_refused_symlink(tmp_path):
    from soup_cli.trainer.mlx_checkpoint import read_bundle

    path = tmp_path / 'mlx-state-7'
    path.symlink_to(tmp_path / 'missing', target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks are refused'):
        read_bundle(path)
