"""Portable pattern precedence and serialized reconstruction validation."""

import copy
import json

import pytest

from soup_cli.config.schema import LoraConfig, SoupConfig
from soup_cli.trainer.mlx_lora import resolve_module_parameters, validate_pattern_manifest
from soup_cli.trainer.mlx_sft import build_mlx_adapter_config
from soup_cli.utils.mlx_generation import read_native_adapter

NAMES = [f"model.layers.{layer}.self_attn.{projection}"
         for layer in (0, 1) for projection in ("q_proj", "v_proj")]


def _recipe(**kwargs):
    return dict(r=4, alpha=8, dropout=0.0, use_rslora=False,
                rank_pattern={}, alpha_pattern={}) | kwargs


def _manifest():
    recipe = _recipe(rank_pattern={"q_proj": 2}, alpha_pattern={"v_proj": 12})
    return {"version": 1, "recipe": recipe,
            "target_keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "modules": resolve_module_parameters(NAMES, recipe)}


def _config(*, backend="mlx", task="sft", **lora):
    return SoupConfig(
        base="test/model", backend=backend, task=task, data={"train": "train.jsonl"},
        training={"quantization": "none", "lora": lora,
                  **({"preference_loss": "dpo"} if task == "preference" else {})},
    )


@pytest.mark.parametrize("patterns", [
    {"rank_pattern": {"q_proj": 2}},
    {"alpha_pattern": {"v_proj": 12}},
    {"rank_pattern": {"q_proj": 2}, "alpha_pattern": {"v_proj": 12}},
])
@pytest.mark.parametrize("rslora", [False, True])
def test_supported_patterns_survive_public_config_roundtrip(patterns, rslora):
    from soup_cli.config.loader import load_config_from_string

    cfg = _config(r=4, alpha=8, use_rslora=rslora, **patterns)
    restored = load_config_from_string(cfg.model_dump_json())
    for actual in (cfg, restored):
        assert actual.backend == "mlx" and actual.task == "sft"
        assert (actual.training.lora.r, actual.training.lora.alpha) == (4, 8)
        assert actual.training.lora.use_rslora is rslora
        for field in ("rank_pattern", "alpha_pattern"):
            assert getattr(actual.training.lora, field) == patterns.get(field)


def test_patterns_survive_mcp_and_sweep_config_entry_points(tmp_path, monkeypatch):
    from soup_cli.commands.sweep import _set_nested_param
    from soup_cli.mcp_server.registry import _load_config_under_cwd

    monkeypatch.chdir(tmp_path)
    cfg = _config(backend="transformers", rank_pattern={"q_proj": 2},
                  alpha_pattern={"v_proj": 12})
    arm = cfg.model_dump()
    _set_nested_param(arm, "backend", "mlx")
    mlx_cfg = SoupConfig(**arm)
    (tmp_path / "soup.yaml").write_text(mlx_cfg.model_dump_json(), encoding="utf-8")
    loaded = _load_config_under_cwd("soup.yaml")
    for actual in (mlx_cfg, loaded):
        assert actual.backend == "mlx"
        assert actual.training.lora.rank_pattern == {"q_proj": 2}
        assert actual.training.lora.alpha_pattern == {"v_proj": 12}


def test_public_pattern_config_passes_train_dry_run(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from tests.conftest import strip_ansi

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"alpha beta gamma"}\n' * 2, encoding="utf-8")
    cfg = _config(rank_pattern={"q_proj": 2}, alpha_pattern={"v_proj": 12})
    cfg.data.train = str(source)
    cfg.data.format = "plaintext"
    cfg.data.val_split = 0
    config = tmp_path / "soup.yaml"
    config.write_text(cfg.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Data OK: 2 train" in " ".join(strip_ansi(result.output).split())
    assert not (tmp_path / "output" / "adapters.safetensors").exists()


@pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto", "grpo"])
@pytest.mark.parametrize("empty", [None, {}])
def test_uniform_mlx_recipes_remain_valid(task, empty):
    cfg = _config(task=task, r=8, alpha=16, use_rslora=True,
                  rank_pattern=empty, alpha_pattern=empty)
    assert cfg.training.lora.rank_pattern == empty
    assert cfg.training.lora.alpha_pattern == empty
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (8, 16)


@pytest.mark.parametrize("task", ["sft", "dpo", "orpo"])
def test_transformers_keeps_patterns(task):
    cfg = _config(backend="transformers", task=task, rank_pattern={"q_proj": 2},
                  alpha_pattern={"v_proj": 12})
    assert cfg.training.lora.rank_pattern == {"q_proj": 2}
    assert cfg.training.lora.alpha_pattern == {"v_proj": 12}


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_mlx_dora_still_refuses_patterns_with_its_own_diagnostic(field):
    assert _config(use_dora=True).training.lora.use_dora
    with pytest.raises(ValueError, match="MLX DoRA does not support"):
        _config(use_dora=True, **{field: {"q_proj": 2}})


@pytest.mark.parametrize("task", ["ipo", "pretrain"])
def test_patterns_do_not_enable_unsupported_mlx_tasks(task):
    assert _config(backend="transformers", task=task).task == task
    with pytest.raises(ValueError, match="MLX backend only ships SFT"):
        _config(task=task, rank_pattern={"q_proj": 2})


@pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto", "grpo", "preference"])
@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_non_sft_patterns_refused_at_load_with_uniform_controls(task, field):
    assert _config(task=task).task == task
    assert getattr(_config(task=task, **{field: {}}).training.lora, field) == {}
    with pytest.raises(ValueError, match=rf"training.lora.{field}.*text SFT only"):
        _config(task=task, **{field: {"q_proj": 2}})


def test_patterns_do_not_enable_mlx_full_fine_tuning():
    assert _config(backend="transformers", r=0).training.lora.r == 0
    with pytest.raises(ValueError, match="full fine-tuning.*requires backend='transformers'"):
        _config(r=0, rank_pattern={"q_proj": 2})


def test_independent_rank_alpha_defaults_and_ordered_regex_precedence():
    """Reference: PEFT 0.21.2 ``peft.utils.other.get_pattern_key`` matching rules."""
    recipe = _recipe(
        rank_pattern={r"^model.layers.0.self_attn.q_proj": 2, "q_proj": 3},
        alpha_pattern={"v_proj": 20},
    )
    result = resolve_module_parameters(NAMES, recipe)
    assert result == {
        NAMES[0]: {"rank": 2, "alpha": 8, "scale": 4, "dropout": 0},
        NAMES[1]: {"rank": 4, "alpha": 20, "scale": 5, "dropout": 0},
        NAMES[2]: {"rank": 3, "alpha": 8, "scale": 8 / 3, "dropout": 0},
        NAMES[3]: {"rank": 4, "alpha": 20, "scale": 5, "dropout": 0},
    }
    recipe["rank_pattern"] = {"q_proj": 3, r"^model.layers.0.self_attn.q_proj": 2}
    assert resolve_module_parameters(NAMES, recipe)[NAMES[0]]["rank"] == 3
    # The narrower pattern still matches a selected module, but is shadowed.
    recipe["rank_pattern"] = {r"layers.[01].self_attn.q_proj": 2}
    assert [p["rank"] for p in resolve_module_parameters(NAMES, recipe).values()] == [2, 4, 2, 4]


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_nonexistent_pattern_refuses_but_real_suffix_is_valid(field):
    assert resolve_module_parameters(NAMES, _recipe(**{field: {"q_proj": 2}}))
    with pytest.raises(ValueError, match=rf"{field} matches no selected module"):
        resolve_module_parameters(NAMES, _recipe(**{field: {"nonexistent": 2}}))


def test_local_rslora_rank_and_alpha_only_override():
    result = resolve_module_parameters(NAMES, _recipe(
        use_rslora=True, rank_pattern={"q_proj": 9}, alpha_pattern={"v_proj": 12},
    ))
    assert result[NAMES[0]]["scale"] == 8 / 3
    assert result[NAMES[1]]["scale"] == 6
    alpha_only = resolve_module_parameters(NAMES, _recipe(alpha_pattern={"q_proj": 12}))
    assert alpha_only[NAMES[0]]["rank"] == 4
    assert alpha_only[NAMES[0]]["scale"] == 3
    assert alpha_only[NAMES[1]]["scale"] == 2


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
@pytest.mark.parametrize("value", [True, 0, -1, 1025, 1.5, "2"])
def test_schema_pattern_values_remain_strict(field, value):
    with pytest.raises(ValueError, match="values must"):
        LoraConfig.model_validate({field: {"q_proj": value}})


@pytest.mark.parametrize("patterns", [{"(": 2}, {"q" * 513: 2},
                                       {f"module{i}": 2 for i in range(257)}])
def test_pattern_shape_limits_remain_bounded(patterns):
    with pytest.raises(ValueError, match="invalid regex|character cap|caps at"):
        resolve_module_parameters(NAMES, _recipe(rank_pattern=patterns))


def test_artifact_roundtrip_and_legacy_uniform_control(tmp_path):
    manifest = _manifest()
    assert validate_pattern_manifest(json.loads(json.dumps(manifest))) == manifest
    cfg = LoraConfig(r=4, alpha=8, dropout=0, rank_pattern={"q_proj": 2})
    uniform = build_mlx_adapter_config(cfg, adapter_path=str(tmp_path))
    assert uniform["lora_parameters"]["rank"] == 4
    config = build_mlx_adapter_config(cfg, adapter_path=str(tmp_path), soup_lora=manifest,
                                      model="owner/base", num_layers=2)
    assert "lora_parameters" not in config
    (tmp_path / "adapters.safetensors").write_bytes(b"not loaded at this boundary")
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    assert read_native_adapter(tmp_path)["soup_lora"] == manifest
    config["lora_parameters"] = uniform["lora_parameters"]
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Ambiguous"):
        read_native_adapter(tmp_path)


@pytest.mark.parametrize("change, diagnostic", [
    ("version", "version"), ("recipe", "targets/recipe"), ("bool_rank", "metadata drift"),
    ("rank", "metadata drift"), ("alpha", "metadata drift"), ("scale", "metadata drift"),
    ("dropout", "metadata drift"), ("extra", "metadata drift"),
])
def test_manifest_drift_refusals_have_valid_control(change, diagnostic):
    manifest = _manifest()
    assert validate_pattern_manifest(manifest) == manifest
    altered = copy.deepcopy(manifest)
    if change == "version":
        altered["version"] = True
    elif change == "recipe":
        altered["recipe"]["unknown"] = 1
    else:
        params = altered["modules"][NAMES[0]]
        if change == "bool_rank":
            params["rank"] = True
        elif change == "extra":
            params["unknown"] = 1
        else:
            params[change] = float("nan") if change == "dropout" else params[change] + 1
    with pytest.raises(ValueError, match=diagnostic):
        validate_pattern_manifest(altered)


def test_incompatible_output_guard_and_atomic_metadata_failure(tmp_path, monkeypatch):
    from soup_cli.trainer.mlx_lora import check_pattern_output, write_pattern_metadata
    from soup_cli.utils import paths

    monkeypatch.chdir(tmp_path)
    metadata = {"model": "owner/base", "soup_lora": _manifest()}
    write_pattern_metadata(tmp_path, metadata)
    (tmp_path / "adapters.safetensors").write_bytes(b"retained weights")
    before = (tmp_path / "adapter_config.json").read_bytes()
    check_pattern_output(tmp_path, metadata)
    changed = copy.deepcopy(metadata)
    changed["model"] = "different/base"
    with pytest.raises(ValueError, match="fresh output directory"):
        check_pattern_output(tmp_path, changed)
    assert (tmp_path / "adapter_config.json").read_bytes() == before

    def unavailable_replace(source, destination):
        raise OSError("injected atomic publish failure")

    monkeypatch.setattr(paths.os, "replace", unavailable_replace)
    with pytest.raises(OSError, match="injected atomic publish failure"):
        write_pattern_metadata(tmp_path, changed)
    assert (tmp_path / "adapter_config.json").read_bytes() == before
    assert (tmp_path / "adapters.safetensors").read_bytes() == b"retained weights"
    assert not list(tmp_path.glob(".soup.*"))


def test_metadata_accepts_absolute_output_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_lora import write_pattern_metadata

    working = tmp_path / "working"
    working.mkdir()
    output = tmp_path / "elsewhere" / "run"
    monkeypatch.chdir(working)
    cfg = SoupConfig.model_validate({
        "base": "owner/base", "data": {"train": "unused.jsonl"}, "output": str(output),
    })
    metadata = {"model": cfg.base, "soup_lora": _manifest()}
    write_pattern_metadata(output, metadata)
    assert json.loads((output / "adapter_config.json").read_text()) == metadata
    assert not list(output.glob(".soup.*"))


def test_uniform_output_reuse_refuses_only_existing_patterned_weights(tmp_path):
    from soup_cli.trainer.mlx_lora import check_pattern_output

    uniform = {"model": "owner/base", "soup_lora": None}
    check_pattern_output(tmp_path, uniform)
    (tmp_path / "adapters.safetensors").write_bytes(b"existing weights")
    check_pattern_output(tmp_path, uniform)  # Legacy weights without metadata.
    metadata = tmp_path / "adapter_config.json"
    metadata.write_text(json.dumps({"model": "owner/base", "lora_parameters": {"rank": 4}}))
    check_pattern_output(tmp_path, uniform)
    metadata.write_text(json.dumps({"model": "owner/base", "soup_lora": _manifest()}))
    with pytest.raises(ValueError, match="fresh output directory"):
        check_pattern_output(tmp_path, uniform)


def _ranks(pattern):
    resolved = resolve_module_parameters(NAMES, _recipe(rank_pattern=pattern))
    return [params["rank"] for params in resolved.values()]


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
@pytest.mark.parametrize("key", ["proj", "q_pro", "self_attn.q_pro", "model.layers.0", "Q_PROJ"])
def test_pattern_must_cover_whole_trailing_name_components(field, key):
    with pytest.raises(ValueError, match=f"{field} matches no selected module"):
        resolve_module_parameters(NAMES, _recipe(**{field: {key: 2}}))


def test_alternation_and_dotted_suffix_spellings():
    assert _ranks({"q_proj|v_proj": 2}) == [2, 2, 2, 2]
    assert _ranks({"self_attn.q_proj": 2}) == [2, 4, 2, 4]
    assert _ranks({"layers.1.self_attn.v_proj": 2}) == [4, 4, 4, 2]


def test_resolved_modules_carry_the_recipe_dropout():
    resolved = resolve_module_parameters(NAMES, _recipe(dropout=0.25, rank_pattern={"q_proj": 2}))
    assert {params["dropout"] for params in resolved.values()} == {0.25}


@pytest.mark.parametrize("change", [
    {"r": 0}, {"alpha": 0}, {"alpha": -4}, {"dropout": 1.0}, {"dropout": -0.1},
    {"dropout": float("nan")}, {"use_rslora": 1},
])
def test_recipe_defaults_are_revalidated_when_read_back(change):
    assert resolve_module_parameters(NAMES, _recipe())
    with pytest.raises(ValueError, match="MLX patterned LoRA requires"):
        resolve_module_parameters(NAMES, _recipe(**change))


@pytest.mark.parametrize("field", ["scale", "dropout"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_module_values_are_refused_as_recipe_drift(field, value):
    manifest = _manifest()
    assert validate_pattern_manifest(copy.deepcopy(manifest)) == manifest
    altered = copy.deepcopy(manifest)
    altered["modules"][NAMES[0]][field] = value
    with pytest.raises(ValueError, match="metadata drift"):
        validate_pattern_manifest(altered)


def test_bool_module_values_are_refused_even_when_they_equal_the_resolved_value():
    recipe = _recipe(rank_pattern={"q_proj": 1})
    manifest = {"version": 1, "recipe": recipe,
                "target_keys": ["self_attn.q_proj", "self_attn.v_proj"],
                "modules": resolve_module_parameters(NAMES, recipe)}
    assert validate_pattern_manifest(copy.deepcopy(manifest)) == manifest
    for key, value in (("rank", True), ("dropout", False)):
        altered = copy.deepcopy(manifest)
        altered["modules"][NAMES[0]][key] = value
        with pytest.raises(ValueError, match="metadata drift"):
            validate_pattern_manifest(altered)


@pytest.mark.parametrize("change, diagnostic", [
    ("drift", "metadata drift"), ("keys_auto", "per-module targets"),
    ("keys_duplicate", "per-module targets"),
    ("layers_zero", "num_layers must be a positive integer"),
    ("layers_bool", "num_layers must be a positive integer"),
    ("no_weights", "missing adapters.safetensors"),
])
def test_read_native_adapter_refuses_bad_per_module_metadata(tmp_path, change, diagnostic):
    config = build_mlx_adapter_config(
        LoraConfig(r=4, alpha=8, dropout=0), adapter_path=str(tmp_path),
        soup_lora=_manifest(), model="owner/base", num_layers=2,
    )
    (tmp_path / "adapters.safetensors").write_bytes(b"not loaded at this boundary")
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    assert read_native_adapter(tmp_path)["num_layers"] == 2
    if change == "drift":
        config["soup_lora"]["modules"][NAMES[0]]["scale"] += 1
    elif change == "keys_auto":
        config["soup_lora"]["target_keys"] = ["auto"]
    elif change == "keys_duplicate":
        config["soup_lora"]["target_keys"] = ["self_attn.q_proj"] * 2
    elif change == "layers_zero":
        config["num_layers"] = 0
    elif change == "layers_bool":
        config["num_layers"] = True
    else:
        (tmp_path / "adapters.safetensors").unlink()
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=diagnostic):
        read_native_adapter(tmp_path)


@pytest.mark.parametrize("weights", ["adapters.safetensors", "0000100_adapters.safetensors"])
def test_patterned_run_refuses_existing_weights_it_cannot_identify(tmp_path, weights):
    from soup_cli.trainer.mlx_lora import check_pattern_output

    patterned = {"model": "owner/base", "soup_lora": _manifest()}
    (tmp_path / weights).write_bytes(b"existing weights")
    metadata = tmp_path / "adapter_config.json"
    for text in (None, "{not json",
                 json.dumps({"model": "owner/base", "lora_parameters": {"rank": 4}})):
        if text is not None:
            metadata.write_text(text)
        with pytest.raises(ValueError, match="fresh output directory"):
            check_pattern_output(tmp_path, patterned)
    metadata.write_text(json.dumps(patterned))
    check_pattern_output(tmp_path, patterned)  # same recipe and base: allowed


def test_non_sft_pattern_config_is_refused_by_train_dry_run(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from tests.conftest import strip_ansi

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: test/model\ntask: dpo\nbackend: mlx\ndata:\n  train: train.jsonl\n"
        "training:\n  quantization: none\n  lora:\n    rank_pattern: {q_proj: 2}\n"
        "    alpha_pattern: {v_proj: 12}\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = " ".join(strip_ansi(result.output).split())
    assert "training.lora.rank_pattern and training.lora.alpha_pattern" in text
    assert "text SFT only" in text


def test_mlx_to_hf_conversion_refuses_a_per_module_manifest(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file

    from soup_cli.utils.apple_adapter import build_apple_adapter_plan, convert_apple_adapter

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "patterned"
    source.mkdir()
    save_file({"model.layers.0.self_attn.q_proj.lora_a": np.zeros((16, 2), np.float32),
               "model.layers.0.self_attn.q_proj.lora_b": np.zeros((2, 16), np.float32)},
              str(source / "adapters.safetensors"))
    (source / "adapter_config.json").write_text(
        json.dumps({"model": "tiny/base", "soup_lora": _manifest()}))
    plan = build_apple_adapter_plan(source_dir="patterned", output_dir="out",
                                    direction="mlx-to-hf")
    with pytest.raises(ValueError, match="soup_lora"):
        convert_apple_adapter(plan)
    assert not (tmp_path / "out" / "adapter_model.safetensors").exists()


@pytest.mark.parametrize("change, diagnostic", [
    ("recipe_rank", "rank must be at most"),
    ("module_rank", "rank must be at most"),
    ("layers", "num_layers must be at most"),
    ("target_count", "keys must be at most"),
    ("target_length", "keys must be at most"),
    ("module_count", "keys must be at most"),
    ("module_length", "keys must be at most"),
])
def test_per_module_reader_enforces_uniform_resource_bounds(
    tmp_path, monkeypatch, change, diagnostic,
):
    from soup_cli.utils import mlx_generation as native

    manifest = _manifest()
    layers = 2
    if change == "recipe_rank":
        manifest["recipe"]["r"] = native._MAX_ADAPTER_RANK + 1
        # All effective modules are bounded: isolate the unused default bound.
        manifest["recipe"]["rank_pattern"] = {"q_proj|v_proj": 2}
    elif change == "module_rank":
        # Lower the resource ceiling so a schema-valid pattern exceeds it.
        monkeypatch.setattr(native, "_MAX_ADAPTER_RANK", 3)
        manifest["recipe"]["r"] = 2
        manifest["recipe"]["rank_pattern"] = {"q_proj": 4}
        manifest["modules"] = resolve_module_parameters(NAMES, manifest["recipe"])
    elif change == "layers":
        layers = native._MAX_ADAPTER_LAYERS + 1
    elif change == "target_count":
        manifest["target_keys"] = [f"target{i}" for i in range(native._MAX_ADAPTER_KEYS + 1)]
    elif change == "target_length":
        manifest["target_keys"] = ["q" * (native._MAX_ADAPTER_KEY_LEN + 1)]
    elif change == "module_count":
        names = [f"model.layers.{i}.self_attn.q_proj"
                 for i in range(native._MAX_ADAPTER_KEYS + 1)]
        manifest["recipe"]["alpha_pattern"] = {}
        manifest["modules"] = resolve_module_parameters(names, manifest["recipe"])
    else:
        name = "q" * native._MAX_ADAPTER_KEY_LEN + ".self_attn.q_proj"
        manifest["recipe"]["alpha_pattern"] = {}
        manifest["modules"] = resolve_module_parameters([name], manifest["recipe"])
    if change != "module_rank":
        manifest["modules"] = resolve_module_parameters(list(manifest["modules"]),
                                                       manifest["recipe"])
    config = build_mlx_adapter_config(
        LoraConfig(), adapter_path=str(tmp_path), soup_lora=manifest,
        model="owner/base", num_layers=layers,
    )
    (tmp_path / "adapters.safetensors").write_bytes(b"not read")
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=diagnostic):
        native.read_native_adapter(tmp_path)


def test_invalid_manifest_is_refused_before_optional_model_import(tmp_path):
    from soup_cli.utils.mlx_generation import load_native_model

    config = build_mlx_adapter_config(
        LoraConfig(), adapter_path=str(tmp_path), soup_lora=_manifest(),
        model="not-a-real-model", num_layers=2,
    )
    config["soup_lora"]["version"] = 99
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="manifest version/fields"):
        load_native_model(str(tmp_path))


def test_per_module_reader_accepts_values_at_resource_bounds(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_generation as native

    manifest = _manifest()
    monkeypatch.setattr(native, "_MAX_ADAPTER_RANK", 4)
    monkeypatch.setattr(native, "_MAX_ADAPTER_LAYERS", 2)
    monkeypatch.setattr(native, "_MAX_ADAPTER_KEYS", len(NAMES))
    monkeypatch.setattr(native, "_MAX_ADAPTER_KEY_LEN", max(map(len, NAMES)))
    config = build_mlx_adapter_config(
        LoraConfig(), adapter_path=str(tmp_path), soup_lora=manifest,
        model="owner/base", num_layers=2,
    )
    (tmp_path / "adapters.safetensors").write_bytes(b"not read")
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    assert native.read_native_adapter(tmp_path)["soup_lora"] == manifest
