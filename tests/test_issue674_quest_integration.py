"""Issue #674: explicit mixed-precision QuEST engineering integration.

The quality result behind this route is evaluation-only.  These tests protect
the engineering contract: exact route accounting, trainable fake quantisation,
metadata round-trips, resume fidelity and fail-closed configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.config.schema import SoupConfig

_TEST_CALIBRATION_SHA256 = "ab" * 32


def _config(**training):
    values = {
        "quantization_aware": "quest",
        "quantization": "none",
        "batch_size": 2,
        "lora": {"r": 0},
        **training,
    }
    return SoupConfig(
        base="ahxt/LiteLlama-460M-1T",
        task="sft",
        modality="text",
        backend="transformers",
        data={"train": "data.jsonl"},
        training=values,
    )


def _tiny_litellama(width: int = 128):
    import torch

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(width, width, bias=False)
            self.k_proj = torch.nn.Linear(width, width, bias=False)
            self.v_proj = torch.nn.Linear(width, width, bias=False)
            self.o_proj = torch.nn.Linear(width, width, bias=False)

    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(width, width, bias=False)
            self.up_proj = torch.nn.Linear(width, width, bias=False)
            self.down_proj = torch.nn.Linear(width, width, bias=False)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(Block() for _ in range(24))
            self.lm_head = torch.nn.Linear(width, 17, bias=False)
            self.config = SimpleNamespace(
                model_type="llama",
                num_hidden_layers=24,
                hidden_size=width,
                intermediate_size=width,
            )

    return Model()


def _scales(model) -> dict[str, float]:
    import torch

    return {
        name: 3.0
        for name, module in model.named_modules()
        if type(module) is torch.nn.Linear and name != "lm_head"
    }


def _calibration_model():
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(128, 128, bias=False)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            values = torch.nn.functional.one_hot(input_ids.remainder(128), num_classes=128).float()
            return self.linear(values)

    return Model()


def _calibration_rows(count: int = 32) -> list[dict[str, list[int]]]:
    return [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 3],
        }
        for _ in range(count)
    ]


def test_schema_accepts_only_the_explicit_first_slice():
    cfg = _config()
    assert cfg.training.quantization_aware == "quest"
    assert cfg.training.quantization == "none"
    assert cfg.training.lora.r == 0
    assert cfg.training.batch_size == 2


@pytest.mark.parametrize(
    "root,training,match",
    [
        ({"task": "dpo"}, {}, "task='sft'"),
        ({"backend": "unsloth"}, {}, "backend='transformers'"),
        ({"modality": "vision"}, {}, "modality='text'"),
        ({}, {"quantization": "4bit"}, "quantization='none'"),
        ({}, {"lora": {"r": 8}}, "lora.r=0"),
        ({}, {"batch_size": "auto"}, "explicit training.batch_size"),
        ({}, {"stream_layers": True}, "stream_layers=false"),
        ({}, {"nvfp4": True}, "nvfp4=false"),
        ({}, {"activation_offloading": "cpu"}, "activation_offloading"),
        ({}, {"activation_offloading": "disk"}, "activation_offloading"),
    ],
)
def test_schema_rejects_unwired_quest_combinations(root, training, match):
    payload = {
        "base": "ahxt/LiteLlama-460M-1T",
        "task": "sft",
        "modality": "text",
        "backend": "transformers",
        "data": {"train": "data.jsonl"},
        "training": {
            "quantization_aware": "quest",
            "quantization": "none",
            "batch_size": 2,
            "lora": {"r": 0},
        },
    }
    payload.update(root)
    payload["training"].update(training)
    with pytest.raises(ValueError, match=match):
        SoupConfig(**payload)


@pytest.mark.parametrize(
    "training",
    [
        {"freeze_layers": 12},
        {"freeze_ratio": 0.5},
        {"unfrozen_parameters": [".*q_proj.*"]},
        {"lisa_enabled": True},
        {"expand_layers": 1, "freeze_trainable_layers": 1},
    ],
)
def test_schema_rejects_partial_training_routes(training):
    with pytest.raises(ValueError, match="unmodified full fine-tuning"):
        _config(**training)


def test_schema_keeps_preload_cut_ce_available_for_quest():
    cfg = _config(use_cut_ce=True)
    assert cfg.training.use_cut_ce is True


def test_non_quest_defaults_are_unchanged():
    cfg = SoupConfig(base="org/model", data={"train": "data.jsonl"})
    assert cfg.training.quantization_aware is False
    assert cfg.training.quantization == "4bit"


def test_cli_dry_run_reports_quest_without_qat_or_torchao(tmp_path, monkeypatch):
    import builtins

    from typer.testing import CliRunner

    import soup_cli.commands.train as train_mod
    import soup_cli.utils.qat as qat
    from soup_cli.cli import app

    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "hello"}\n', encoding="utf-8")
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        "base: ahxt/LiteLlama-460M-1T\n"
        "task: sft\n"
        "backend: transformers\n"
        "modality: text\n"
        f"output: {tmp_path / 'out'}\n"
        "data:\n"
        f"  train: {data_path}\n"
        "training:\n"
        "  quantization_aware: quest\n"
        "  quantization: none\n"
        "  batch_size: 2\n"
        "  lora:\n"
        "    r: 0\n",
        encoding="utf-8",
    )
    qat_calls = []
    monkeypatch.setattr(train_mod, "detect_device", lambda backend=None: ("cpu", "CPU"))
    monkeypatch.setattr(
        train_mod, "get_gpu_info", lambda backend=None: {"memory_total": "N/A"}
    )
    monkeypatch.setattr(
        train_mod,
        "load_dataset",
        lambda *args, **kwargs: {"train": [{"text": "hello"}]},
    )
    monkeypatch.setattr(
        qat,
        "validate_qat_config",
        lambda *args, **kwargs: qat_calls.append((args, kwargs)) or ["must not run"],
    )
    real_import = builtins.__import__

    def reject_torchao(name, *args, **kwargs):
        if name == "torchao" or name.startswith("torchao."):
            raise AssertionError("QuEST dry-run must not require torchao")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torchao)
    result = CliRunner().invoke(
        app, ["train", "--config", str(config_path), "--dry-run", "--yes"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "mixed W4/A4+A16 (QuEST fake quant)" in result.output
    assert qat_calls == []


def test_install_accounts_for_every_weight_and_activation_route():
    from soup_cli.utils.quest import install_mixed_quest

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    assert metadata["recipe"] == "w4a4-group128-block23-a16"
    assert metadata["weight_routes"] == {"w4": 168}
    assert metadata["activation_routes"] == {"a4": 161, "a16": 7}
    assert metadata["pure_w4a4"] is False
    assert metadata["training_quality_validated"] is False
    assert metadata["packed_int4"] is False
    assert metadata["transform"] == "full-width-normalized-hadamard"
    assert metadata["weight_scale"] == 2.513930578568423
    assert metadata["surrogate"] == "quest-trust-gradient"
    assert metadata["calibration"]["rows_sha256"] == _TEST_CALIBRATION_SHA256
    assert metadata["a16_modules"] == [
        "model.layers.23.self_attn.q_proj",
        "model.layers.23.self_attn.k_proj",
        "model.layers.23.self_attn.v_proj",
        "model.layers.23.self_attn.o_proj",
        "model.layers.23.mlp.gate_proj",
        "model.layers.23.mlp.up_proj",
        "model.layers.23.mlp.down_proj",
    ]


def test_route_provenance_is_topology_only_and_bound_to_the_public_record():
    from soup_cli.utils.quest import ROUTE_PROVENANCE

    assert ROUTE_PROVENANCE == {
        "schema_version": 1,
        "scope": "topology_selection_only",
        "measured_on": {
            "mode": "evaluation_only",
            "model": "ahxt/LiteLlama-460M-1T",
            "examples": 704,
            "targets": 25017,
            "gap_nat": 0.0863441881,
            "ci95": [0.0802412531, 0.0931898109],
            "result_sha256": (
                "94fee10542da29281f7753cbf221a3421ad5acf67f2b290e52d65acece359cdf"
            ),
        },
        "claims": {
            "artifact_training_quality": False,
            "cross_model_quality": False,
        },
    }
    measured = ROUTE_PROVENANCE["measured_on"]
    record = (
        Path(__file__).parents[1] / "benchmarks" / "gate-674-quest-mixed-route.md"
    ).read_text(encoding="utf-8")
    for binding in (
        measured["result_sha256"],
        "0.086344",
        "[0.080241, 0.093190]",
        "704 examples",
        "25,017 targets",
    ):
        assert str(binding) in record


def test_metadata_accepts_a_schema_valid_provenance_correction():
    from soup_cli.utils.quest import install_mixed_quest, validate_metadata

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="other/model-with-the-same-topology",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    measured = metadata["route_provenance"]["measured_on"]
    measured.update(
        model="corrected/model-id",
        examples=705,
        targets=25018,
        gap_nat=0.08,
        ci95=[0.07, 0.09],
        result_sha256="cd" * 32,
    )
    assert validate_metadata(metadata) is metadata


def test_provenance_correction_does_not_invalidate_resume_or_restore(tmp_path):
    from soup_cli.utils.quest import (
        install_mixed_quest,
        restore_mixed_quest,
        validate_resume_metadata,
        write_metadata,
    )

    source = _tiny_litellama()
    current = install_mixed_quest(
        source,
        activation_scales=_scales(source),
        base_model="same/model",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    historical = json.loads(json.dumps(current))
    historical["route_provenance"]["measured_on"].update(
        model="corrected/model-id",
        examples=705,
        targets=25018,
        gap_nat=0.08,
        ci95=[0.07, 0.09],
        result_sha256="cd" * 32,
    )
    write_metadata(tmp_path, historical)
    validate_resume_metadata(tmp_path, current)

    restored = _tiny_litellama()
    assert restore_mixed_quest(restored, historical) is restored
    assert int(restored.model.layers[0].self_attn.q_proj.quest_activation_bits) == 4


def test_v1_local_sidecar_restores_without_the_original_base_directory():
    from soup_cli.utils.quest import install_mixed_quest, restore_mixed_quest

    source = _tiny_litellama()
    historical = install_mixed_quest(
        source,
        activation_scales=_scales(source),
        base_model="C:/Users/old-account/models/base",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
        format_version=1,
    )
    target = _tiny_litellama()
    assert restore_mixed_quest(target, historical) is target
    assert int(target.model.layers[0].self_attn.q_proj.quest_activation_bits) == 4


def test_quantizer_matches_the_retained_grid_and_trust_gradient():
    import torch

    from soup_cli.utils.quest import quantize_rotated

    torch.manual_seed(674)
    value = torch.randn(2, 3, 256, dtype=torch.float32, requires_grad=True)
    scale = 3.0
    actual = quantize_rotated(value, scale=scale, group=128)
    grouped = value.detach().reshape(2, 3, 2, 128)
    rms = grouped.square().mean(-1, keepdim=True).sqrt()
    bound = rms * scale + 1e-8
    spacing = 2 * bound / 15
    expected = (grouped.clamp(-bound, bound) / spacing + 0.5).round() * spacing - spacing / 2
    trusted = ((expected - grouped).abs() <= rms * (scale / 15)).float()
    # The trust-gradient expression reconstructs the same grid value through
    # one detached subtraction/addition, so the forward may differ by one FP32
    # rounding ulp from the directly written reference expression.
    torch.testing.assert_close(
        actual.detach(),
        expected.reshape_as(value),
        rtol=0,
        atol=torch.finfo(torch.float32).eps,
    )
    actual.sum().backward()
    assert torch.equal(value.grad, trusted.reshape_as(value))


def test_all_weights_stay_w4_while_only_block23_activations_bypass_a4(monkeypatch):
    import torch

    import soup_cli.utils.quest as quest

    model = _tiny_litellama()
    quest.install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    calls: list[tuple[str, int]] = []
    original = quest.quantize_rotated

    def record(value, *, scale, group):
        calls.append(("weight" if value.ndim == 2 else "activation", group))
        return original(value, scale=scale, group=group)

    monkeypatch.setattr(quest, "quantize_rotated", record)
    value = torch.randn(2, 3, 128)
    model.model.layers[22].self_attn.q_proj(value)
    assert calls == [("activation", 128), ("weight", 128)]
    calls.clear()
    model.model.layers[23].self_attn.q_proj(value)
    assert calls == [("weight", 128)]


def test_mixed_layer_backward_updates_the_original_master_weight():
    import torch

    from soup_cli.utils.quest import install_mixed_quest

    model = _tiny_litellama()
    parameter = model.model.layers[23].mlp.down_proj.weight
    install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    value = torch.randn(2, 128, requires_grad=True)
    model.model.layers[23].mlp.down_proj(value).square().mean().backward()
    assert model.model.layers[23].mlp.down_proj.weight is parameter
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad)


def test_state_dict_roundtrip_preserves_the_exact_route_and_output():
    import torch

    from soup_cli.utils.quest import install_mixed_quest

    torch.manual_seed(17)
    source = _tiny_litellama()
    scales = _scales(source)
    install_mixed_quest(
        source,
        activation_scales=scales,
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    target = _tiny_litellama()
    install_mixed_quest(
        target,
        activation_scales=scales,
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    target.load_state_dict(source.state_dict(), strict=True)
    value = torch.randn(2, 128)
    assert torch.equal(
        source.model.layers[22].mlp.down_proj(value),
        target.model.layers[22].mlp.down_proj(value),
    )
    assert torch.equal(
        source.model.layers[23].mlp.down_proj(value),
        target.model.layers[23].mlp.down_proj(value),
    )


def test_route_buffers_do_not_pollute_the_transformers_checkpoint():
    from soup_cli.utils.quest import install_mixed_quest

    model = _tiny_litellama()
    install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    assert not any("quest_" in name for name in model.state_dict())


def test_build_metadata_rejects_an_incomplete_activation_scale_table():
    from soup_cli.utils.quest import build_metadata

    scales = _scales(_tiny_litellama())
    scales.pop("model.layers.23.mlp.down_proj")
    with pytest.raises(ValueError, match="all 168"):
        build_metadata(
            activation_scales=scales,
            base_model="ahxt/LiteLlama-460M-1T",
            calibration_sha256=_TEST_CALIBRATION_SHA256,
        )


def test_metadata_roundtrip_and_resume_fidelity(tmp_path: Path):
    from soup_cli.utils.quest import (
        install_mixed_quest,
        load_metadata,
        validate_resume_metadata,
        write_metadata,
    )

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    path = write_metadata(tmp_path, metadata)
    assert path.name == "quest_mixed_precision.json"
    loaded = load_metadata(tmp_path)
    assert loaded == metadata
    validate_resume_metadata(tmp_path, metadata)


def test_resume_rejects_missing_or_different_route(tmp_path: Path):
    from soup_cli.utils.quest import (
        install_mixed_quest,
        validate_resume_metadata,
        write_metadata,
    )

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    with pytest.raises(ValueError, match="missing QuEST metadata"):
        validate_resume_metadata(tmp_path, metadata)
    changed = json.loads(json.dumps(metadata))
    changed["activation_scales"]["model.layers.0.self_attn.q_proj"] = 4.0
    write_metadata(tmp_path, changed)
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_metadata(tmp_path, metadata)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(format_version=True), "format_version"),
        (lambda value: value.update(recipe="unknown"), "recipe"),
        (lambda value: value.update(pure_w4a4=True), "pure_w4a4"),
        (lambda value: value["a16_modules"].pop(), "seven"),
        (lambda value: value["activation_routes"].update(a4=162), "route"),
        (
            lambda value: value["activation_scales"].update(
                {"model.layers.0.self_attn.q_proj": float("nan")}
            ),
            "scale",
        ),
        (
            lambda value: value["calibration"].update(rows_sha256="not-a-digest"),
            "row binding",
        ),
        (
            lambda value: value["route_provenance"]["measured_on"]["ci95"].pop(),
            "confidence interval",
        ),
        (
            lambda value: value["route_provenance"]["claims"].update(
                artifact_training_quality=True
            ),
            "must not claim artifact quality",
        ),
    ],
)
def test_metadata_validation_rejects_unknown_or_malformed_routes(tmp_path, mutation, match):
    from soup_cli.utils.quest import install_mixed_quest, load_metadata, write_metadata

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    mutation(metadata)
    write_metadata(tmp_path, metadata, validate=False)
    with pytest.raises(ValueError, match=match):
        load_metadata(tmp_path)


@pytest.mark.parametrize("count", [23, 25])
def test_install_refuses_any_topology_other_than_the_measured_24_blocks(count):
    import copy

    import torch

    from soup_cli.utils.quest import install_mixed_quest

    model = _tiny_litellama()
    layers = list(model.model.layers)
    if count == 23:
        layers = layers[:23]
    else:
        layers.append(copy.deepcopy(layers[-1]))
    model.model.layers = torch.nn.ModuleList(layers)
    with pytest.raises(ValueError, match="24 blocks"):
        install_mixed_quest(
            model,
            activation_scales=_scales(model),
            base_model="other/model",
            calibration_sha256=_TEST_CALIBRATION_SHA256,
        )


def test_install_refuses_partial_calibration_before_replacing_any_layer():
    import torch

    from soup_cli.utils.quest import install_mixed_quest

    model = _tiny_litellama()
    scales = _scales(model)
    scales.pop("model.layers.23.mlp.down_proj")
    with pytest.raises(ValueError, match="all 168"):
        install_mixed_quest(
            model,
            activation_scales=scales,
            base_model="ahxt/LiteLlama-460M-1T",
            calibration_sha256=_TEST_CALIBRATION_SHA256,
        )
    assert all(
        type(module) is torch.nn.Linear for name, module in model.named_modules() if name in scales
    )


def test_install_refuses_non_fp32_masters_before_replacing_any_layer():
    import torch

    from soup_cli.utils.quest import install_mixed_quest

    model = _tiny_litellama().to(dtype=torch.float64)
    with pytest.raises(ValueError, match="FP32 master weights"):
        install_mixed_quest(
            model,
            activation_scales=_scales(model),
            base_model="ahxt/LiteLlama-460M-1T",
            calibration_sha256=_TEST_CALIBRATION_SHA256,
        )
    assert type(model.model.layers[0].self_attn.q_proj) is torch.nn.Linear


def test_calibration_selects_local_error_minimum_and_removes_hooks(monkeypatch):
    import soup_cli.utils.quest as quest

    model = _calibration_model()
    model.train()
    monkeypatch.setattr(quest, "EXPECTED_MODULES", ("linear",))
    monkeypatch.setattr(
        quest, "_validate_raw_topology", lambda candidate: {"linear": candidate.linear}
    )
    monkeypatch.setattr(quest, "rotate", lambda value: value)

    def controlled_quantizer(value, *, scale, group):
        del group
        if value.ndim == 2:
            return value
        return value if scale == 3.0 else value * 0

    monkeypatch.setattr(quest, "quantize_rotated", controlled_quantizer)
    rows = _calibration_rows(33)
    rows[-1] = {}
    selected = quest.calibrate_activation_scales(model, rows)
    assert selected == {"linear": 3.0}
    assert model.training is True
    assert not model.linear._forward_pre_hooks


def test_calibration_failure_removes_hooks_and_restores_mode(monkeypatch):
    import soup_cli.utils.quest as quest

    model = _calibration_model().eval()
    monkeypatch.setattr(quest, "EXPECTED_MODULES", ("linear",))
    monkeypatch.setattr(
        quest, "_validate_raw_topology", lambda candidate: {"linear": candidate.linear}
    )
    rows = _calibration_rows()
    rows[0].pop("labels")
    with pytest.raises(ValueError, match="missing"):
        quest.calibrate_activation_scales(model, rows)
    assert model.training is False
    assert not model.linear._forward_pre_hooks


def test_calibration_rejects_a_zero_energy_proxy(monkeypatch):
    import soup_cli.utils.quest as quest

    model = _calibration_model()
    model.linear.weight.data.zero_()
    monkeypatch.setattr(quest, "EXPECTED_MODULES", ("linear",))
    monkeypatch.setattr(
        quest, "_validate_raw_topology", lambda candidate: {"linear": candidate.linear}
    )
    with pytest.raises(ValueError, match="local error"):
        quest.calibrate_activation_scales(model, _calibration_rows())


def test_calibration_refuses_an_unserialized_position_limit():
    from soup_cli.utils.quest import calibrate_activation_scales

    with pytest.raises(ValueError, match="calibration_position_limit at 16"):
        calibrate_activation_scales(
            _calibration_model(),
            _calibration_rows(),
            position_limit=8,
        )


def test_calibration_row_binding_is_deterministic_and_sensitive():
    from soup_cli.utils.quest import calibration_rows_sha256

    first = _calibration_rows(33)
    second = _calibration_rows(33)
    second[-1]["input_ids"][0] = 99
    assert calibration_rows_sha256(first) == calibration_rows_sha256(second)
    second[0]["input_ids"][0] = 99
    assert calibration_rows_sha256(first) != calibration_rows_sha256(second)


@pytest.mark.parametrize("mutation", ["precision", "missing"])
def test_runtime_route_assertion_kills_route_mutations(mutation):
    import torch

    from soup_cli.utils.quest import assert_route, install_mixed_quest

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    target = model.model.layers[0].self_attn.q_proj
    if mutation == "precision":
        target.quest_activation_bits.fill_(16)
    else:
        model.model.layers[0].self_attn.q_proj = torch.nn.Linear(128, 128)
    with pytest.raises(ValueError, match="route"):
        assert_route(model, metadata)


def test_loader_requires_metadata_and_reconstructs_the_route(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.quest import (
        install_mixed_quest,
        load_mixed_quest_artifact,
        write_metadata,
    )

    source = _tiny_litellama()
    metadata = install_mixed_quest(
        source,
        activation_scales=_scales(source),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    write_metadata(tmp_path, metadata)
    raw = _tiny_litellama()
    raw.config.soup_quest = metadata
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return raw

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_load)
    loaded = load_mixed_quest_artifact(tmp_path, local_files_only=True)
    assert loaded is raw
    assert calls == [(tmp_path, {"local_files_only": True})]
    assert int(raw.model.layers[0].self_attn.q_proj.quest_activation_bits) == 4
    assert int(raw.model.layers[23].mlp.down_proj.quest_activation_bits) == 16


def test_loader_refuses_config_sidecar_drift(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.quest import (
        install_mixed_quest,
        load_mixed_quest_artifact,
        write_metadata,
    )

    source = _tiny_litellama()
    metadata = install_mixed_quest(
        source,
        activation_scales=_scales(source),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    write_metadata(tmp_path, metadata)
    raw = _tiny_litellama()
    raw.config.soup_quest = {**metadata, "recipe": "different"}
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="config declaration"):
        load_mixed_quest_artifact(tmp_path)


def test_loader_refuses_a_missing_config_declaration(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.quest import (
        install_mixed_quest,
        load_mixed_quest_artifact,
        write_metadata,
    )

    source = _tiny_litellama()
    metadata = install_mixed_quest(
        source,
        activation_scales=_scales(source),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    write_metadata(tmp_path, metadata)
    raw = _tiny_litellama()
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="config declaration"):
        load_mixed_quest_artifact(tmp_path)


def test_hardware_gate_requires_exactly_one_ampere_or_newer_gpu(monkeypatch):
    import torch

    from soup_cli.utils.quest import validate_cuda_hardware

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "RTX 3090")
    assert validate_cuda_hardware() == ("RTX 3090", (8, 6))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    with pytest.raises(RuntimeError, match="exactly one visible"):
        validate_cuda_hardware()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 5))
    with pytest.raises(RuntimeError, match="Ampere or newer"):
        validate_cuda_hardware()


def test_hardware_gate_rejects_cpu(monkeypatch):
    import torch

    from soup_cli.utils.quest import validate_cuda_hardware

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        validate_cuda_hardware()


def test_setup_quest_rejects_every_distributed_route(monkeypatch):
    from soup_cli.trainer import stream_setup
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.deepspeed_config = None
    wrapper.fsdp_config = None
    monkeypatch.setattr(stream_setup, "_distributed_launch", lambda: True)
    with pytest.raises(ValueError, match="single-GPU"):
        wrapper._setup_quest(_calibration_rows())


def test_setup_quest_calibrates_then_installs_and_stores_metadata(monkeypatch):
    import soup_cli.utils.quest as quest
    from soup_cli.trainer import stream_setup
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.deepspeed_config = None
    wrapper.fsdp_config = None
    wrapper.model = SimpleNamespace(config=SimpleNamespace())
    wrapper.config = SimpleNamespace(base="ahxt/LiteLlama-460M-1T")
    rows = _calibration_rows()
    events = []
    metadata = {"route": "sentinel"}
    monkeypatch.setattr(stream_setup, "_distributed_launch", lambda: False)
    monkeypatch.setattr(quest, "validate_cuda_hardware", lambda: ("RTX 3090", (8, 6)))
    monkeypatch.setattr(
        quest,
        "calibrate_activation_scales",
        lambda model, dataset: events.append(("calibrate", model, dataset)) or {"x": 3.0},
    )
    monkeypatch.setattr(
        quest,
        "calibration_rows_sha256",
        lambda dataset: events.append(("fingerprint", dataset)) or _TEST_CALIBRATION_SHA256,
    )
    monkeypatch.setattr(
        quest,
        "install_mixed_quest",
        lambda model, *, activation_scales, base_model, calibration_sha256: (
            events.append(("install", model, activation_scales, base_model, calibration_sha256))
            or metadata
        ),
    )
    wrapper._setup_quest(rows)
    assert [event[0] for event in events] == [
        "fingerprint",
        "calibrate",
        "install",
    ]
    assert events[0][1] == rows
    assert events[0][1] is events[1][2]
    assert events[2][2:] == (
        {"x": 3.0},
        "ahxt/LiteLlama-460M-1T",
        _TEST_CALIBRATION_SHA256,
    )
    assert wrapper._quest_metadata is metadata
    assert wrapper.model.config.soup_quest is metadata


def test_transformers_setup_checks_hardware_before_loading(monkeypatch):
    import soup_cli.utils.quest as quest
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = object.__new__(SFTTrainerWrapper)
    monkeypatch.setattr(
        quest,
        "validate_cuda_hardware",
        lambda: (_ for _ in ()).throw(RuntimeError("hardware sentinel")),
    )
    with pytest.raises(RuntimeError, match="hardware sentinel"):
        wrapper._setup_transformers(
            SimpleNamespace(base="unused"),
            SimpleNamespace(quantization_aware="quest"),
        )


def test_quest_never_enters_normal_qat_or_v028_paths(monkeypatch):
    import soup_cli.utils.qat as qat
    import soup_cli.utils.v028_features as v028_features
    from soup_cli.trainer.sft import SFTTrainerWrapper

    def unexpected(*args, **kwargs):
        raise AssertionError("QuEST must bypass the normal QAT and v0.28 paths")

    monkeypatch.setattr(qat, "prepare_model_for_qat", unexpected)
    monkeypatch.setattr(v028_features, "apply_v028_speed_memory", unexpected)
    wrapper = object.__new__(SFTTrainerWrapper)
    original_model = object()
    wrapper.model = original_model
    wrapper._apply_quantization_aware(SimpleNamespace(quantization_aware="quest"))
    assert wrapper.model is original_model


def test_train_validates_resume_and_rewrites_final_metadata(monkeypatch, tmp_path):
    import contextlib

    import soup_cli.trainer.sft as sft
    import soup_cli.utils.quest as quest
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import ebft_gdpo, peft_wiring

    events = []
    metadata = {"route": "sentinel"}

    class FakeModel:
        def named_parameters(self):
            return []

    class FakeTrainer:
        def __init__(self):
            self.model = FakeModel()
            self.args = SimpleNamespace(fp16=False, bf16=False, should_save=True)
            self.state = SimpleNamespace(log_history=[], global_step=9)

        def train(self, *, resume_from_checkpoint):
            events.append(("train", resume_from_checkpoint))

        def save_model(self, output):
            events.append(("save", output))

    class FakeTokenizer:
        def save_pretrained(self, output):
            events.append(("tokenizer", output))

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.config = _config()
    wrapper._quest_metadata = metadata
    wrapper._output_dir = str(tmp_path)
    wrapper.trainer = FakeTrainer()
    wrapper.tokenizer = FakeTokenizer()
    wrapper._training_context = lambda context: contextlib.ExitStack()
    wrapper._report_rewind = lambda: None

    for name in (
        "attach_loraplus_optimizer",
        "attach_relora_callback",
        "attach_lisa_callback",
        "attach_curriculum_callback",
        "attach_plugin_callback",
    ):
        monkeypatch.setattr(peft_wiring, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(ebft_gdpo, "attach_ebft_compute_loss", lambda *args: None)
    monkeypatch.setattr(sft, "align_trainable_dtype_for_fp16", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        quest,
        "validate_resume_metadata",
        lambda checkpoint, value, **kwargs: events.append(("resume", checkpoint, value, kwargs)),
    )
    monkeypatch.setattr(
        quest,
        "write_metadata",
        lambda output, value: events.append(("metadata", output, value)),
    )

    result = wrapper.train(resume_from_checkpoint="checkpoint-4")
    assert events[:5] == [
        (
            "resume",
            "checkpoint-4",
            metadata,
            {"legacy_base_model": "ahxt/LiteLlama-460M-1T"},
        ),
        ("metadata", str(tmp_path), metadata),
        ("train", "checkpoint-4"),
        ("save", str(tmp_path)),
        ("metadata", str(tmp_path), metadata),
    ]
    assert events[-1] == ("tokenizer", str(tmp_path))
    assert result["quest_mixed_precision"] is metadata
    assert result["total_steps"] == 9

    events.clear()
    monkeypatch.setattr(
        quest,
        "validate_resume_metadata",
        lambda checkpoint, value, **kwargs: (
            _ for _ in ()
        ).throw(ValueError("resume route mismatch")),
    )
    with pytest.raises(ValueError, match="resume route mismatch"):
        wrapper.train(resume_from_checkpoint="checkpoint-5")
    assert events == []


def test_callback_writes_the_same_metadata_into_periodic_checkpoint(tmp_path: Path):
    from soup_cli.utils.quest import QuestMetadataCallback, load_metadata

    model = _tiny_litellama()
    from soup_cli.utils.quest import install_mixed_quest

    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    callback = QuestMetadataCallback(str(tmp_path), metadata)
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path), should_save=True),
        SimpleNamespace(global_step=12),
        None,
    )
    assert load_metadata(checkpoint) == metadata


def test_callback_does_not_write_on_non_saving_rank(tmp_path: Path):
    from soup_cli.utils.quest import QuestMetadataCallback, install_mixed_quest

    model = _tiny_litellama()
    metadata = install_mixed_quest(
        model,
        activation_scales=_scales(model),
        base_model="ahxt/LiteLlama-460M-1T",
        calibration_sha256=_TEST_CALIBRATION_SHA256,
    )
    callback = QuestMetadataCallback(str(tmp_path), metadata)
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path), should_save=False),
        SimpleNamespace(global_step=12),
        None,
    )
    assert not (tmp_path / "checkpoint-12" / "quest_mixed_precision.json").exists()


def test_heavy_dependencies_stay_lazy():
    import ast

    source = Path("src/soup_cli/utils/quest.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] not in {"torch", "transformers"} for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"torch", "transformers"}
