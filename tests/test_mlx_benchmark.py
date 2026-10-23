"""Public native benchmark dispatch, refusal and persistent evidence contracts."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model").mkdir()
    return tmp_path


def invoke(*options):
    return CliRunner().invoke(
        app,
        [
            "eval",
            "benchmark",
            "--backend",
            "mlx",
            "--model",
            "model",
            "--benchmarks",
            "hellaswag",
            *options,
        ],
    )


@pytest.mark.parametrize(
    "options,diagnostic",
    [
        (["--benchmarks", "wikitext"], "supports hellaswag and gsm8k"),
        (["--benchmarks", "mmlu"], "supports hellaswag and gsm8k"),
        (["--device", "cuda"], "device"),
        (["--batch-size", "0"], "batch-size"),
        (["--limit", "0"], "limit"),
        (["--fewshot", "-1"], "fewshot"),
        (["--max-length", "1"], "max-length"),
        (["--max-tokens", "0"], "max-tokens"),
        (["--seed", "-1"], "seed"),
    ],
)
def test_native_benchmark_controls_fail_before_loading(workspace, options, diagnostic):
    result = invoke(*options)
    assert result.exit_code != 0
    assert diagnostic in result.output
    assert "Results saved" not in result.output


def test_native_benchmark_missing_harness_has_install_diagnostic(workspace, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "lm_eval", None)
    result = invoke()
    assert result.exit_code == 1
    assert "eval" in result.output and "install" in result.output.lower()
    assert "Results saved" not in result.output


def test_native_benchmark_refuses_peft_before_loading(workspace):
    (workspace / "model" / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "owner/base",
                "r": 4,
            }
        )
    )
    result = invoke()
    assert result.exit_code == 1
    assert "native MLX metadata" in result.output
    assert "Results saved" not in result.output


def test_native_benchmark_refuses_base_identity_mismatch(workspace):
    adapter = workspace / "model"
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "model": "owner/base",
                "fine_tune_type": "lora",
                "num_layers": 1,
                "lora_parameters": {"rank": 4, "scale": 2, "dropout": 0, "keys": ["lm_head"]},
            }
        )
    )
    (adapter / "adapters.safetensors").write_bytes(b"not loaded for identity refusal")
    result = invoke("--base", "owner/wrong")
    assert result.exit_code == 1
    assert "does not match" in result.output
    assert "Results saved" not in result.output


@pytest.mark.parametrize("source", ["path", "metadata"])
def test_native_benchmark_keeps_delimiter_injection_refusal(workspace, source):
    model = workspace / "model"
    if source == "metadata":
        (model / "adapter_config.json").write_text(
            json.dumps(
                {
                    "model": "owner/base,trust_remote_code=True",
                    "fine_tune_type": "lora",
                    "num_layers": 1,
                    "lora_parameters": {"rank": 4, "scale": 2, "dropout": 0, "keys": ["lm_head"]},
                }
            )
        )
        (model / "adapters.safetensors").write_bytes(b"not loaded for injection refusal")
        result = invoke()
    else:
        injected = workspace / "model,trust_remote_code=True"
        injected.mkdir()
        result = invoke("--model", str(injected))
    assert result.exit_code == 1
    assert "delimiters" in result.output
    assert "Results saved" not in result.output
