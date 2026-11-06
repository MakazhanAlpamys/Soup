"""Public native benchmark dispatch, refusal and persistent evidence contracts."""

import json
from copy import deepcopy

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


@pytest.mark.parametrize(
    "corruption",
    [None, "minimal", "disjoint", "metric", "null-arguments", "null-resps", "null-filtered_resps"],
)
def test_sample_evidence_guard_without_optional_backends(corruption):
    from soup_cli.eval.mlx_benchmark import _validate_sample_evidence

    samples = [
        {
            "doc_id": doc, "filter": name, "doc": {"question": "A"}, "target": 0,
            "arguments": [["A", "B"]], "resps": [[[-0.1, True]]],
            "filtered_resps": [[-0.1, True]], "metrics": ["acc"], "acc": 1.0,
            "doc_hash": f"doc-{doc}", "prompt_hash": f"prompt-{doc}", "target_hash": "target",
        }
        for name in ("none", "second") for doc in (0, 1)
    ]
    if corruption == "minimal":
        samples = [{"doc_id": row["doc_id"], "filter": row["filter"]} for row in samples]
    elif corruption == "disjoint":
        for row in samples:
            if row["filter"] == "second":
                row["doc_id"] += 10
    elif corruption == "metric":
        del samples[0]["acc"]
    elif corruption and corruption.startswith("null-"):
        samples[0][corruption.removeprefix("null-")] = [None]
    if corruption is None:
        original = deepcopy(samples)
        _validate_sample_evidence("hellaswag", samples, {"none", "second"}, 2, "acc")
        assert samples == original
    else:
        with pytest.raises(ValueError, match="sample.*evidence"):
            _validate_sample_evidence("hellaswag", samples, {"none", "second"}, 2, "acc")


@pytest.mark.parametrize("field", [None, "arguments", "resps", "filtered_resps"])
def test_generation_evidence_accepts_empty_text_but_refuses_null_payload(field):
    from soup_cli.eval.mlx_benchmark import _validate_sample_evidence

    samples = [{
        "doc_id": 0, "filter": "none", "doc": {"question": "A"}, "target": "B",
        "arguments": [["A", {}]], "resps": [[""]], "filtered_resps": [""],
        "metrics": ["exact_match"], "exact_match": 0.0,
        "doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target",
    }]
    if field is None:
        original = deepcopy(samples)
        _validate_sample_evidence("gsm8k", samples, {"none"}, 1, "exact_match")
        assert samples == original
    else:
        samples[0][field] = [None]
        with pytest.raises(ValueError, match="sample.*evidence"):
            _validate_sample_evidence("gsm8k", samples, {"none"}, 1, "exact_match")


@pytest.mark.parametrize("value", [[float("nan"), True], [0.0, 1], [True, False]])
def test_likelihood_payload_refuses_nonfinite_or_wrong_typed_components(value):
    from soup_cli.eval.mlx_benchmark import _validate_sample_evidence

    samples = [{
        "doc_id": 0, "filter": "none", "doc": {"question": "A"}, "target": 0,
        "arguments": [["A", "B"]], "resps": [[value]], "filtered_resps": [value],
        "metrics": ["acc"], "acc": 1.0,
        "doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target",
    }]
    with pytest.raises(ValueError, match="sample.*evidence"):
        _validate_sample_evidence("hellaswag", samples, {"none"}, 1, "acc")


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
