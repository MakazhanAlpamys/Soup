"""#1912: no-MTP conversion follows saved tensors and the selected tool's flag."""

import json
import struct
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import export as export_mod
from soup_cli.utils.gguf_mtp import needs_no_mtp
from tests.conftest import strip_ansi


def _weights(path: Path, names: tuple[str, ...]) -> None:
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [index * 4, index * 4 + 4]}
        for index, name in enumerate(names)
    }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(names)))


@pytest.fixture
def export_mtp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "merged"
    model.mkdir()
    tool = tmp_path / "llama.cpp"
    tool.mkdir()
    calls = []
    real_run = subprocess.run
    monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda path=None: tool)
    monkeypatch.setattr(export_mod, "_install_convert_deps", lambda **kwargs: None)
    monkeypatch.setattr(export_mod, "_maybe_attach_export", lambda **kwargs: None)

    def run(command, **kwargs):
        if "--print-supported-models" in command:
            # #1911's independent diagnostic may probe the same selected tool.
            return real_run(command, **kwargs)
        if "--help" in command:
            calls.append("help")
            return real_run(command, **kwargs)
        # Exercise the real _run_convert argument construction, not a fake helper.
        calls.append(command)
        Path(command[command.index("--outfile") + 1]).write_bytes(b"conversion fixture")
        return subprocess.CompletedProcess(command, 0, "", "")

    def quantize(llama, source, output, quant, *, verbose=False):
        # Match _run_quantize's contract: conversion-only options must fail here.
        calls.append("quantize")
        output.write_bytes(b"quantization fixture")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(export_mod, "_run_quantize", quantize)

    def invoke(
        *,
        tensors=("model.layers.0.weight",),
        quant="f16",
        flag=True,
        failed=False,
        sharded=False,
        config=None,
        corrupt=False,
    ):
        metadata = {"architectures": ["Qwen3_5ForCausalLM"], "mtp_num_hidden_layers": 1}
        (model / "config.json").write_text(json.dumps(metadata if config is None else config))
        config_before = (model / "config.json").read_bytes()
        if sharded:
            for index, name in enumerate(tensors):
                _weights(model / f"model-{index}.safetensors", (name,))
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            name: f"model-{index}.safetensors" for index, name in enumerate(tensors)
                        },
                    }
                )
            )
        elif corrupt:
            (model / "model.safetensors").write_bytes(b"bad")
        else:
            _weights(model / "model.safetensors", tensors)
        (tool / "convert_hf_to_gguf.py").write_text(
            "import sys\n"
            "if '--print-supported-models' in sys.argv:\n"
            "    print('Supported models:\\nTEXT:\\n - Qwen3_5ForCausalLM"
            "\\n - LlamaForCausalLM\\n - OuterModel')\n"
            "    sys.exit(0)\n"
            f"print({'options: --no-mtp' if flag else 'options: --outtype'!r})\n"
            f"sys.exit({1 if failed else 0})\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            [
                "export",
                "--model",
                str(model),
                "--llama-cpp",
                str(tool),
                "--quant",
                quant,
                "--output",
                str(tmp_path / "out.gguf"),
            ],
        )
        assert (model / "config.json").read_bytes() == config_before
        return result, calls

    return invoke


@pytest.mark.parametrize(
    "config,message",
    [
        ({"mtp_num_hidden_layers": -1}, "nonnegative integer"),
        ({"mtp_num_hidden_layers": True}, "nonnegative integer"),
        ({"mtp_num_hidden_layers": None}, "nonnegative integer"),
        ({"text_config": []}, "text_config must contain an object"),
        ([], "config.json must contain an object"),
    ],
)
def test_invalid_configuration_is_refused_for_its_own_reason(tmp_path, config, message):
    # A readable checkpoint WITH saved MTP is present, so only the config check can raise.
    _weights(tmp_path / "model.safetensors", ("mtp.layers.0.weight",))
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        needs_no_mtp(tmp_path, tmp_path / "unused.py")


def test_index_shard_outside_the_model_dir_is_refused_even_when_it_exists(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "outside.safetensors", ("model.layers.0.weight",))
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="escapes the model directory"):
        needs_no_mtp(model, tmp_path / "unused.py")


@pytest.mark.requires_symlink
def test_index_shard_symlinked_outside_the_model_dir_is_refused(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "outside.safetensors", ("model.layers.0.weight",))
    (model / "link.safetensors").symlink_to(tmp_path / "outside.safetensors")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "link.safetensors"}})
    )
    with pytest.raises(ValueError, match="escapes the model directory"):
        needs_no_mtp(model, tmp_path / "unused.py")


def test_index_shard_without_safetensors_suffix_is_refused(tmp_path):
    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "model.bin", ("model.layers.0.weight",))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model.bin"}})
    )
    with pytest.raises(ValueError, match="invalid shard filename"):
        needs_no_mtp(tmp_path, tmp_path / "unused.py")


def test_oversized_config_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"mtp_num_hidden_layers":1,"pad":"' + "x" * (1024 * 1024) + '"}'
    )
    _weights(tmp_path / "model.safetensors", ("mtp.layers.0.weight",))
    with pytest.raises(ValueError, match="metadata limit"):
        needs_no_mtp(tmp_path, tmp_path / "unused.py")


def test_unindexed_shards_are_all_inspected(tmp_path):
    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "a.safetensors", ("model.layers.0.weight",))
    _weights(tmp_path / "b.safetensors", ("mtp.layers.0.weight",))
    assert needs_no_mtp(tmp_path, tmp_path / "unused.py") is False


def test_every_shard_is_validated_even_after_mtp_is_found(tmp_path):
    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "a.safetensors", ("mtp.layers.0.weight",))
    (tmp_path / "b.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated"):
        needs_no_mtp(tmp_path, tmp_path / "unused.py")


def test_probe_is_time_bounded(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "model.safetensors", ("model.layers.0.weight",))
    seen = {}

    def probe(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "options: --no-mtp", "")

    monkeypatch.setattr(subprocess, "run", probe)
    assert needs_no_mtp(tmp_path, tmp_path / "converter.py") is True
    assert seen["timeout"] == 15


@pytest.mark.parametrize("mode", ["prefixed-flag", "undecodable"])
def test_more_probe_outcomes_do_not_establish_capability(tmp_path, monkeypatch, mode):
    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "model.safetensors", ("model.layers.0.weight",))

    def probe(command, **kwargs):
        if mode == "undecodable":
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return subprocess.CompletedProcess(command, 0, "usage: --x--no-mtp", "")

    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(ValueError, match="--llama-cpp"):
        needs_no_mtp(tmp_path, tmp_path / "converter.py")


def test_refusal_text_is_terminal_safe(export_mtp, monkeypatch):
    def boom(model_dir, converter):
        raise ValueError("[bold]shard[/] \x9bevil")

    monkeypatch.setattr("soup_cli.utils.gguf_mtp.needs_no_mtp", boom)
    result, calls = export_mtp()
    output = strip_ansi(result.output)
    assert result.exit_code == 1, output
    assert "[bold]shard[/]" in output
    assert "\x9b" not in result.output
    assert calls == []


@pytest.mark.parametrize("quant", ["f16", "f32", "q4_k_m"])
@pytest.mark.parametrize("sharded", [False, True])
def test_absent_mtp_adds_flag_on_both_conversion_paths(export_mtp, quant, sharded):
    result, calls = export_mtp(quant=quant, sharded=sharded)
    assert result.exit_code == 0, (result.output, result.exception)
    conversions = [call for call in calls if isinstance(call, list)]
    assert len(conversions) == 1
    assert "--no-mtp" in conversions[0]
    assert calls[0] == "help"
    assert ("quantize" in calls) == (quant == "q4_k_m")


@pytest.mark.parametrize("name", ["mtp.layers.0.weight", "model.mtp.layers.0.weight"])
@pytest.mark.parametrize("sharded", [False, True])
def test_saved_mtp_is_preserved_without_flag_or_probe(export_mtp, name, sharded):
    result, calls = export_mtp(tensors=("model.layers.0.weight", name), sharded=sharded)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "help" not in calls
    assert all("--no-mtp" not in command for command in calls if isinstance(command, list))


@pytest.mark.parametrize("flag,failed", [(False, False), (True, True)])
def test_unavailable_converter_flag_refuses_before_conversion(export_mtp, flag, failed):
    result, calls = export_mtp(flag=flag, failed=failed)
    output = strip_ansi(result.output)
    assert result.exit_code == 1, output
    assert "--no-mtp" in output
    assert "--llama-cpp" in output
    assert calls == ["help"]


def test_model_without_mtp_configuration_keeps_existing_path(export_mtp):
    result, calls = export_mtp(config={"architectures": ["LlamaForCausalLM"]}, flag=False)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "help" not in calls
    assert all("--no-mtp" not in command for command in calls if isinstance(command, list))


def test_nested_mtp_configuration_is_honored(export_mtp):
    result, calls = export_mtp(
        config={"architectures": ["OuterModel"], "text_config": {"mtp_num_hidden_layers": 1}}
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "--no-mtp" in next(call for call in calls if isinstance(call, list))


def test_zero_layer_workaround_still_uses_no_mtp_when_weights_are_absent(export_mtp):
    result, calls = export_mtp(
        config={"architectures": ["Qwen3_5ForCausalLM"], "mtp_num_hidden_layers": 0}
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "--no-mtp" in next(call for call in calls if isinstance(call, list))


def test_malformed_header_does_not_become_absence(export_mtp):
    result, calls = export_mtp(corrupt=True)
    output = strip_ansi(result.output)
    assert result.exit_code == 1, output
    assert "MTP" in output
    assert "safetensors" in output
    assert calls == []


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"weight_map": {}},
        {"weight_map": {"x": "../outside.safetensors"}},
        {"weight_map": {"x": "missing.safetensors"}},
        {"weight_map": {"x": None}},
    ],
)
def test_invalid_index_is_refused(tmp_path, payload):
    from soup_cli.utils.gguf_mtp import needs_no_mtp

    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="MTP safetensors"):
        needs_no_mtp(tmp_path, tmp_path / "converter.py")


def test_headers_are_read_without_tensor_payloads(tmp_path, monkeypatch):
    import builtins

    from soup_cli.utils.gguf_mtp import needs_no_mtp

    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    weights = tmp_path / "model.safetensors"
    _weights(weights, ("mtp.layers.0.weight",))
    original = builtins.open
    reads = []

    class HeaderOnly:
        def __init__(self):
            self.handle = original(weights, "rb")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            assert size >= 0
            reads.append(size)
            return self.handle.read(size)

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: HeaderOnly())
    assert needs_no_mtp(tmp_path, tmp_path / "unused.py") is False
    assert reads == [8, weights.stat().st_size - 8 - 4]


@pytest.mark.parametrize("mode", ["timeout", "oserror", "different-flag"])
def test_probe_failure_or_similar_flag_does_not_establish_capability(tmp_path, monkeypatch, mode):
    from soup_cli.utils.gguf_mtp import needs_no_mtp

    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    _weights(tmp_path / "model.safetensors", ("model.layers.0.weight",))

    def probe(command, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if mode == "oserror":
            raise OSError("unavailable")
        return subprocess.CompletedProcess(command, 0, "--no-mtp-extra", "")

    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(ValueError, match="--llama-cpp"):
        needs_no_mtp(tmp_path, tmp_path / "converter.py")


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"text_config": []},
        {"mtp_num_hidden_layers": -1},
        {"mtp_num_hidden_layers": True},
        {"mtp_num_hidden_layers": None},
    ],
)
def test_invalid_configuration_is_refused(tmp_path, config):
    from soup_cli.utils.gguf_mtp import needs_no_mtp

    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="MTP safetensors metadata"):
        needs_no_mtp(tmp_path, tmp_path / "unused.py")


def test_missing_checkpoint_is_not_absent_mtp(tmp_path):
    from soup_cli.utils.gguf_mtp import needs_no_mtp

    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    with pytest.raises(ValueError, match="no safetensors checkpoint"):
        needs_no_mtp(tmp_path, tmp_path / "unused.py")


def test_index_selects_nested_shards_and_ignores_unreferenced_weights(tmp_path, monkeypatch):
    from soup_cli.utils.gguf_mtp import needs_no_mtp

    (tmp_path / "config.json").write_text('{"mtp_num_hidden_layers":1}')
    nested = tmp_path / "shards"
    nested.mkdir()
    _weights(nested / "part.safetensors", ("model.layers.0.weight",))
    _weights(tmp_path / "unused.safetensors", ("mtp.layers.0.weight",))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"model.layers.0.weight": "shards/part.safetensors"},
            }
        )
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            "",
            "options: --no-mtp",
        ),
    )
    assert needs_no_mtp(tmp_path, tmp_path / "converter.py") is True
