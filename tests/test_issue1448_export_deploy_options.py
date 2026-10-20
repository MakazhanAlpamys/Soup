"""Tests for issue #1448: soup export --deploy and --registry-id on non-GGUF formats.

Acceptance criteria:
- Formats that Ollama cannot load (onnx, tensorrt, awq, gptq, torchao) refuse --deploy.
- Advanced GGUF formats (gguf-ud, bitnet, tq1_0) wire --deploy to _auto_deploy_ollama.
- All export formats wire --registry-id (and auto-attach) to _maybe_attach_export.
- _VALID_KINDS in registry store includes 'torchao'.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import _VALID_KINDS
from tests.conftest import strip_ansi

runner = CliRunner()


def _clean(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def test_torchao_in_valid_kinds():
    """torchao must be recognized as a valid registry artifact kind."""
    assert "torchao" in _VALID_KINDS


@pytest.mark.parametrize("fmt", ["onnx", "tensorrt", "awq", "gptq", "torchao"])
def test_export_refuses_deploy_for_unsupported_formats(tmp_path: Path, fmt: str):
    """Formats Ollama cannot load must refuse --deploy with an informative error."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    result = runner.invoke(
        app,
        ["export", "--model", str(model_dir), "--format", fmt, "--deploy", "ollama"],
    )
    assert result.exit_code == 1
    output = _clean(result.output)
    assert f"--deploy is not supported for format '{fmt}'" in output
    assert "Supported formats for deploy: gguf, gguf-ud, bitnet, tq1_0" in output


def test_export_refuses_unsupported_deploy_target(tmp_path: Path):
    """Unsupported deploy targets must exit 1 and report supported targets."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    result = runner.invoke(
        app,
        ["export", "--model", str(model_dir), "--format", "gguf", "--deploy", "invalid_target"],
    )
    assert result.exit_code == 1
    output = _clean(result.output)
    assert "Unsupported deploy target: invalid_target" in output
    assert "Supported: ollama" in output


@pytest.mark.parametrize("fmt", ["gguf-ud", "bitnet", "tq1_0"])
def test_export_wires_deploy_for_deployable_formats(tmp_path: Path, fmt: str):
    """Formats producing deployable GGUF files must forward --deploy to _auto_deploy_ollama."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake_gguf = tmp_path / "exported.gguf"
    fake_gguf.write_bytes(b"GGUF_TEST")

    with patch("soup_cli.commands.export._auto_deploy_ollama") as mock_deploy, \
         patch("soup_cli.commands.export._maybe_attach_export") as mock_attach, \
         patch("soup_cli.commands.export._export_gguf_advanced", return_value=fake_gguf), \
         patch("soup_cli.commands.export._export_bitnet_gguf", return_value=fake_gguf):

        extra_args = ["--gguf-flavour", "Q4_0_4_4"] if fmt == "gguf-ud" else []
        result = runner.invoke(
            app,
            [
                "export",
                "--model", str(model_dir),
                "--format", fmt,
                "--deploy", "ollama",
                "--deploy-name", "my-deployed-model",
                *extra_args,
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        mock_deploy.assert_called_once()
        call_args = mock_deploy.call_args[0]
        assert call_args[0] == fake_gguf
        assert call_args[2] == "ollama"
        assert call_args[3] == "my-deployed-model"
        mock_attach.assert_called_once()


@pytest.mark.parametrize(
    ("fmt", "patch_target", "expected_kind"),
    [
        ("onnx", "soup_cli.commands.export._export_onnx", "onnx"),
        ("tensorrt", "soup_cli.commands.export._export_tensorrt", "tensorrt"),
        ("awq", "soup_cli.commands.export._export_awq", "awq"),
        ("gptq", "soup_cli.commands.export._export_gptq", "gptq"),
        ("torchao", "soup_cli.commands.export._export_torchao_cli", "torchao"),
        ("gguf-ud", "soup_cli.commands.export._export_gguf_advanced", "gguf"),
        ("bitnet", "soup_cli.commands.export._export_bitnet_gguf", "gguf"),
        ("tq1_0", "soup_cli.commands.export._export_bitnet_gguf", "gguf"),
    ],
)
def test_export_wires_registry_id_across_all_formats(
    tmp_path: Path, fmt: str, patch_target: str, expected_kind: str
):
    """Every export format must attach its artifact via _maybe_attach_export with --registry-id."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    exported_file = tmp_path / f"export_{fmt}.bin"
    exported_file.write_bytes(b"DATA")

    with patch(patch_target, return_value=exported_file), \
         patch("soup_cli.commands.export._maybe_attach_export") as mock_attach:

        extra_args = []
        if fmt == "gguf-ud":
            extra_args = ["--gguf-flavour", "Q4_0_4_4"]
        elif fmt == "torchao":
            q_file = tmp_path / "q.yaml"
            q_file.write_text("scheme: int8wo\n", encoding="utf-8")
            extra_args = ["--quant-config", str(q_file)]

        result = runner.invoke(
            app,
            [
                "export",
                "--model", str(model_dir),
                "--format", fmt,
                "--registry-id", "reg-12345",
                *extra_args,
            ],
        )

        assert result.exit_code == 0, f"Export for {fmt} failed: {result.output}"
        mock_attach.assert_called_once()
        kwargs = mock_attach.call_args[1]
        assert kwargs["explicit_id"] == "reg-12345"
        assert kwargs["kind"] == expected_kind
        assert kwargs["artifact_path"] == str(exported_file)


def test_the_real_helper_hands_its_output_to_the_registry(tmp_path: Path, monkeypatch):
    """Integration test: CLI command dispatches to real _maybe_attach_export and registers in DB."""
    from soup_cli.registry.store import RegistryStore

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "reg.db"
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))

    store = RegistryStore(db_path=db_path)
    entry_id = store.push(
        name="my-onnx-model", tag="v1", base_model="org/base", task="sft", run_id=None, config={},
    )
    store.close()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    exported_file = tmp_path / "output.onnx"
    exported_file.write_bytes(b"MODEL_DATA")

    with patch("soup_cli.commands.export._export_onnx", return_value=exported_file):
        result = runner.invoke(
            app,
            [
                "export",
                "--model", str(model_dir),
                "--format", "onnx",
                "--registry-id", entry_id,
            ],
        )

    assert result.exit_code == 0, result.output
    output = _clean(result.output)
    expected_msg = (
        f"Attached export to registry entry '{entry_id}' as onnx ({exported_file.name})"
    )
    assert expected_msg in output

    store = RegistryStore(db_path=db_path)
    artifacts = store.get_artifacts(entry_id)
    store.close()

    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "onnx"
    assert os.path.realpath(artifacts[0]["path"]) == os.path.realpath(str(exported_file))


def test_maybe_attach_export_resolves_directory_to_primary_file(tmp_path: Path, monkeypatch):
    """When artifact_path is a directory, _maybe_attach_export finds the primary model file."""
    from soup_cli.commands.export import _maybe_attach_export
    from soup_cli.registry.store import RegistryStore

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "reg.db"
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))

    store = RegistryStore(db_path=db_path)
    entry_id = store.push(
        name="test-model", tag="v1", base_model="org/base", task="sft", run_id=None, config={},
    )
    store.close()

    export_dir = tmp_path / "onnx_out"
    export_dir.mkdir()
    onnx_file = export_dir / "model.onnx"
    onnx_file.write_bytes(b"FAKE_ONNX_WEIGHTS")

    _maybe_attach_export(
        artifact_path=str(export_dir),
        kind="onnx",
        explicit_id=entry_id,
        source_model=str(tmp_path / "source"),
    )

    store = RegistryStore(db_path=db_path)
    artifacts = store.get_artifacts(entry_id)
    store.close()

    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "onnx"
    assert os.path.realpath(artifacts[0]["path"]) == os.path.realpath(str(onnx_file))


def test_a_tensorrt_export_is_attached(tmp_path: Path, monkeypatch):
    """TensorRT engine layout (<target>/engine/*.engine) is discovered and attached."""
    from soup_cli.commands.export import _maybe_attach_export
    from soup_cli.registry.store import RegistryStore

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "reg.db"
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))

    store = RegistryStore(db_path=db_path)
    entry_id = store.push(
        name="trt-model", tag="v1", base_model="org/base", task="sft", run_id=None, config={},
    )
    store.close()

    trt_dir = tmp_path / "trt_out"
    (trt_dir / "checkpoint").mkdir(parents=True)
    (trt_dir / "checkpoint" / "config.json").write_text("{}", encoding="utf-8")
    (trt_dir / "engine").mkdir(parents=True)
    engine_file = trt_dir / "engine" / "rank0.engine"
    engine_file.write_bytes(b"TRT_ENGINE_BYTES")

    _maybe_attach_export(
        artifact_path=str(trt_dir),
        kind="tensorrt",
        explicit_id=entry_id,
        source_model=str(tmp_path / "source"),
    )

    store = RegistryStore(db_path=db_path)
    artifacts = store.get_artifacts(entry_id)
    store.close()

    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "tensorrt"
    assert os.path.realpath(artifacts[0]["path"]) == os.path.realpath(str(engine_file))


def test_directory_with_multiple_shards_attaches_all_shards(tmp_path: Path, monkeypatch):
    """Directory export containing multiple shards attaches all shard files."""
    from soup_cli.commands.export import _maybe_attach_export
    from soup_cli.registry.store import RegistryStore

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "reg.db"
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))

    store = RegistryStore(db_path=db_path)
    entry_id = store.push(
        name="multi-shard", tag="v1", base_model="org/base", task="sft", run_id=None, config={},
    )
    store.close()

    export_dir = tmp_path / "sharded_out"
    export_dir.mkdir()
    shard1 = export_dir / "model-00001-of-00002.safetensors"
    shard2 = export_dir / "model-00002-of-00002.safetensors"
    shard1.write_bytes(b"SHARD1")
    shard2.write_bytes(b"SHARD2")
    (export_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    _maybe_attach_export(
        artifact_path=str(export_dir),
        kind="torchao",
        explicit_id=entry_id,
        source_model=str(tmp_path / "source"),
    )

    store = RegistryStore(db_path=db_path)
    artifacts = store.get_artifacts(entry_id)
    store.close()

    assert len(artifacts) == 2
    attached_paths = {os.path.realpath(a["path"]) for a in artifacts}
    assert attached_paths == {os.path.realpath(str(shard1)), os.path.realpath(str(shard2))}
    assert all(a["kind"] == "torchao" for a in artifacts)


def test_onnx_external_data_attaches_both_files(tmp_path: Path, monkeypatch):
    """ONNX export with external data file (.onnx + .onnx_data) attaches both."""
    from soup_cli.commands.export import _maybe_attach_export
    from soup_cli.registry.store import RegistryStore

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "reg.db"
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))

    store = RegistryStore(db_path=db_path)
    entry_id = store.push(
        name="onnx-large", tag="v1", base_model="org/base", task="sft", run_id=None, config={},
    )
    store.close()

    export_dir = tmp_path / "onnx_dir"
    export_dir.mkdir()
    onnx_file = export_dir / "model.onnx"
    data_file = export_dir / "model.onnx_data"
    onnx_file.write_bytes(b"ONNX_GRAPH")
    data_file.write_bytes(b"ONNX_WEIGHTS")

    _maybe_attach_export(
        artifact_path=str(export_dir),
        kind="onnx",
        explicit_id=entry_id,
        source_model=str(tmp_path / "source"),
    )

    store = RegistryStore(db_path=db_path)
    artifacts = store.get_artifacts(entry_id)
    store.close()

    assert len(artifacts) == 2
    attached_paths = {os.path.realpath(a["path"]) for a in artifacts}
    assert attached_paths == {os.path.realpath(str(onnx_file)), os.path.realpath(str(data_file))}
    assert all(a["kind"] == "onnx" for a in artifacts)


def test_torchao_dense_export_artifact_beats_lora_config():
    """A torchao export proves the artifact is a standalone model even if trained via LoRA."""
    from soup_cli.commands.card import _DENSE_KINDS, _is_adapter

    assert "torchao" in _DENSE_KINDS
    artifacts = [{"kind": "torchao", "path": "model.torchao", "sha256": "x"}]
    config = {"training": {"lora": {"r": 8}}}
    assert _is_adapter(artifacts, config) is False

