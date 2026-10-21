"""Portable pattern precedence and serialized reconstruction validation."""

import copy
import json

import pytest

from soup_cli.config.schema import LoraConfig
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


def test_independent_rank_alpha_defaults_and_ordered_regex_precedence():
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
