"""Both merge paths load the commit recorded by the adapter's training run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner, Result

from tests.conftest import strip_ansi

BASE = "org/base"
RECORDED = "a" * 40
MOVING_HEAD = "b" * 40
UNKNOWN_LINE = "Base revision: unknown (legacy, local source or base override)"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture()
def merge_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from soup_cli.cli import app
    from soup_cli.utils import hubs, lowmem_merge, run_provenance

    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": BASE,
    }), encoding="utf-8")
    metadata = {
        "schema_version": 1, "run_id": "training-run", "base_model": BASE,
        "base_revision": RECORDED, "base_revision_source": "hub_commit",
    }
    provenance = adapter / "soup_run.json"
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    calls: list[tuple[str, str, dict]] = []

    def save_model(directory: str | Path, revision: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "loaded-revision.json").write_text(json.dumps(revision), encoding="utf-8")

    def load_model(base: str, **kwargs: object) -> SimpleNamespace:
        calls.append(("model", base, kwargs))
        revision = kwargs.get("revision", MOVING_HEAD)
        return SimpleNamespace(save_pretrained=lambda path: save_model(path, revision))

    def snapshot(base: str, **kwargs: object) -> str:
        calls.append(("snapshot", base, kwargs))
        path = tmp_path / str(kwargs.get("revision", MOVING_HEAD))
        save_model(path, str(kwargs.get("revision", MOVING_HEAD)))
        return str(path)

    def save_tokenizer(directory: str | Path) -> None:
        (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")

    def merge_low_memory(base: Path, adapter: Path, output: Path, dtype: str,
                         **kwargs: object) -> None:
        revision = json.loads((base / "loaded-revision.json").read_text())
        save_model(output, revision)
        kwargs["finalize"](output)

    def no_hub_resolution() -> None:
        pytest.fail("merge must use recorded identity, not query today's Hub head")

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", SimpleNamespace(
        from_pretrained=load_model,
    ))
    monkeypatch.setattr(transformers, "AutoTokenizer", SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(save_pretrained=save_tokenizer),
    ))
    monkeypatch.setattr(peft, "PeftModel", SimpleNamespace(
        from_pretrained=lambda model, adapter: SimpleNamespace(merge_and_unload=lambda: model),
    ))
    monkeypatch.setattr(hubs, "snapshot_download", snapshot)
    monkeypatch.setattr(lowmem_merge, "merge_low_memory", merge_low_memory)
    monkeypatch.setattr(run_provenance, "get_hf_api", no_hub_resolution)

    def invoke(*extra: str) -> Result:
        return CliRunner().invoke(app, [
            "--no-audit-log", "--no-telemetry", "merge", "--adapter", str(adapter),
            "--output", "merged", *extra,
        ])

    return SimpleNamespace(invoke=invoke, calls=calls, adapter=adapter,
                           metadata=metadata, provenance=provenance, root=tmp_path)


@pytest.mark.parametrize("explicit", [False, True], ids=["detected", "matching-override"])
def test_regular_merge_uses_the_recorded_commit(merge_cli: SimpleNamespace, explicit: bool) -> None:
    args = ["--base", BASE] if explicit else []
    result = merge_cli.invoke(*args)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(merge_cli.calls) == 1
    kind, base, kwargs = merge_cli.calls[0]
    assert (kind, base) == ("model", BASE)
    assert kwargs.get("revision") == RECORDED
    assert json.loads((merge_cli.root / "merged/loaded-revision.json").read_text()) == RECORDED
    assert f"Base revision: {RECORDED} (recorded run)" in _plain(result.output)


@pytest.mark.parametrize("explicit", [False, True], ids=["detected", "matching-override"])
def test_low_memory_merge_downloads_the_recorded_commit(
    merge_cli: SimpleNamespace, explicit: bool,
) -> None:
    args = ["--base", BASE] if explicit else []
    result = merge_cli.invoke("--low-memory", *args)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(merge_cli.calls) == 1
    kind, base, kwargs = merge_cli.calls[0]
    assert (kind, base) == ("snapshot", BASE)
    assert kwargs.get("revision") == RECORDED
    assert json.loads((merge_cli.root / "merged/loaded-revision.json").read_text()) == RECORDED
    assert f"Base revision: {RECORDED} (recorded run)" in _plain(result.output)


@pytest.mark.parametrize("low_memory", [False, True], ids=["regular", "low-memory"])
@pytest.mark.parametrize("case", [
    "different-base", "legacy", "unknown", "cache-hint", "invalid-commit",
    "malformed", "local", "non-hf",
])
def test_merge_preserves_unknown_local_and_override_loads(
    merge_cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, low_memory: bool, case: str,
) -> None:
    args = ["--low-memory"] if low_memory else []
    if case == "different-base":
        args.extend(["--base", "org/other"])
    elif case == "legacy":
        merge_cli.provenance.unlink()
    elif case == "unknown":
        merge_cli.metadata.update(base_revision=None, base_revision_source="unknown")
    elif case == "cache-hint":
        merge_cli.metadata["base_revision_source"] = "cache_path"
    elif case == "invalid-commit":
        merge_cli.metadata["base_revision"] = "not-a-commit"
    elif case in {"local", "non-hf"}:
        local = merge_cli.root / "local-base"
        local.mkdir()
        (local / "loaded-revision.json").write_text(json.dumps("local-bytes"))
        if case == "local":
            merge_cli.metadata["base_model"] = str(local)
            args.extend(["--base", str(local)])
        else:
            monkeypatch.setattr("soup_cli.utils.hubs.apply_hub_to_cli_model",
                                lambda *args, **kwargs: (str(merge_cli.adapter), str(local)))
            args.extend(["--hub", "modelscope"])

    if case != "legacy":
        merge_cli.provenance.write_text(
            "{broken" if case == "malformed" else json.dumps(merge_cli.metadata),
        )
    result = merge_cli.invoke(*args)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert all(kwargs.get("revision") is None for _, _, kwargs in merge_cli.calls)
    assert UNKNOWN_LINE in _plain(result.output)
    assert "(recorded run)" not in _plain(result.output)
    assert (merge_cli.root / "merged/loaded-revision.json").is_file()


@pytest.mark.parametrize("low_memory", [False, True], ids=["regular", "low-memory"])
def test_recorded_commit_selects_the_real_offline_merge_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, low_memory: bool,
) -> None:
    import torch
    import transformers
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from tests.test_lowmem_merge import _read_all, _regular_merge, _tiny_adapter, _tiny_base

    monkeypatch.chdir(tmp_path)
    recorded = _tiny_base(tmp_path / "recorded")
    moved = tmp_path / "moved"
    loader = transformers.AutoModelForCausalLM.from_pretrained
    changed = loader(recorded, dtype=torch.float32)
    with torch.no_grad():
        changed.model.layers[0].self_attn.q_proj.weight.add_(0.5)
    changed.save_pretrained(moved)
    adapter = _tiny_adapter(recorded, tmp_path / "adapter")
    expected = _regular_merge(recorded, adapter, torch.float32)
    wrong = _regular_merge(moved, adapter, torch.float32)
    assert any(not torch.equal(expected[key], wrong[key]) for key in expected)
    config = adapter / "adapter_config.json"
    data = json.loads(config.read_text())
    data["base_model_name_or_path"] = BASE
    config.write_text(json.dumps(data))
    (adapter / "soup_run.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "trained", "base_model": BASE,
        "base_revision": RECORDED, "base_revision_source": "hub_commit",
    }))

    def select_model(base: str, **kwargs: object) -> object:
        assert base == BASE
        selected = recorded if kwargs.pop("revision", None) == RECORDED else moved
        return loader(selected, **kwargs)

    def select_snapshot(base: str, **kwargs: object) -> str:
        assert base == BASE
        return str(recorded if kwargs.get("revision") == RECORDED else moved)

    def save_tokenizer(path: str | Path) -> None:
        (Path(path) / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", SimpleNamespace(
        from_pretrained=select_model,
    ))
    monkeypatch.setattr(transformers, "AutoTokenizer", SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(save_pretrained=save_tokenizer),
    ))
    monkeypatch.setattr("soup_cli.utils.hubs.snapshot_download", select_snapshot)
    args = ["--low-memory"] if low_memory else []
    result = CliRunner().invoke(app, [
        "--no-audit-log", "--no-telemetry", "merge", "--adapter", str(adapter),
        "--output", "merged", "--dtype", "float32", *args,
    ])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    actual = _read_all(tmp_path / "merged")
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert f"Base revision: {RECORDED} (recorded run)" in _plain(result.output)
