"""Negative controls for interpreting a real pilot journal."""

import copy

import pytest


def _fixture():
    index = {
        "total_params": 671026419200,
        "arch": "deepseek_v3",
        "n_layers": 61,
        "expert_count": 256,
        "expert_layers": list(range(3, 61)),
        "source_files": [[f"model-{n}.safetensors", 1, 1] for n in range(163)],
        "quant": "nf4",
        "double_quant": False,
    }
    events = [
        {"event": "environment", "cache_budget_bytes": 100, "vram_bytes": 8000},
        {"event": "sharded", "source_parameters": index["total_params"], "cache_bytes": 90},
        {"event": "heldout_before", "loss": 3.0},
    ]
    for step in range(2):
        events.append(
            {
                "event": "step",
                "step": step,
                "loss": 2.0,
                "gradient_norm": 0.1,
                "gradient_tensors": 610,
                "adapter_sha256_before": str(step) * 64,
                "adapter_sha256_after": str(step + 1) * 64,
                "seconds": 10.0,
                "tokens": 64,
                "tokens_per_second": 6.4,
                "peak_allocated": 1000,
                "peak_reserved": 2000,
            }
        )
    events += [
        {"event": "heldout_after", "loss": 3.1},
        {
            "event": "reload",
            "equal": True,
            "max_abs_error": 0,
            "parameter_digest_matches": True,
            "parameter_sha256": "2" * 64,
        },
        {"event": "complete"},
    ]
    return events, index


def test_quality_degradation_is_reported_without_changing_acceptance_threshold():
    from benchmarks.harness.giant_moe_verdict import verdict

    events, index = _fixture()
    result = verdict(events, index, 8151)
    assert result["pilot_checks_passed"]
    assert result["heldout_probe"]["heldout_after"] == 3.1
    assert result["ram_32gb_hardware_validated"] is False


@pytest.mark.parametrize(
    ("defect", "failures"),
    [
        ("nan", ["non-finite training loss"]),
        (
            "no_update",
            ["optimizer did not demonstrably change the adapter", "adapter digest chain is broken"],
        ),
        ("reload", ["reload logits are not exactly equal"]),
        ("reload_parameters", ["reload adapter parameters differ from trained parameters"]),
        ("toy", ["source is not a real giant", "journal/index parameter counts differ"]),
        ("incomplete", ["pilot did not complete"]),
        ("disk", ["cache exceeds planned budget"]),
        ("throughput", ["invalid measured throughput"]),
        ("gradients", ["missing finite nonzero gradients"]),
    ],
)
def test_invalid_run_cannot_be_accepted(defect, failures):
    from benchmarks.harness.giant_moe_verdict import verdict

    events, index = copy.deepcopy(_fixture())
    if defect == "nan":
        events[3]["loss"] = float("nan")
    elif defect == "no_update":
        events[3]["adapter_sha256_after"] = events[3]["adapter_sha256_before"]
    elif defect == "reload":
        events[-2]["equal"] = False
    elif defect == "reload_parameters":
        events[-2]["parameter_sha256"] = "0" * 64
    elif defect == "toy":
        index["total_params"] = 1000000
    elif defect == "disk":
        events[1]["cache_bytes"] = 101
    elif defect == "throughput":
        events[3]["tokens_per_second"] = 1
    elif defect == "gradients":
        events[3]["gradient_norm"] = 0
    else:
        events.pop()
    result = verdict(events, index, 8151)
    assert not result["pilot_checks_passed"]
    assert result["failures"] == failures


def test_shorter_unpadded_input_uses_its_actual_token_count():
    from benchmarks.harness.giant_moe_verdict import verdict

    events, index = _fixture()
    events[4]["tokens"] = 62
    events[4]["tokens_per_second"] = 6.2
    assert verdict(events, index, 8151)["pilot_checks_passed"]
