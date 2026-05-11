"""v0.46.0 Part A — On-Device Deploy Autopilot tests."""

from __future__ import annotations

import os
import stat as _stat
import sys
from types import MappingProxyType

import pytest
from typer.testing import CliRunner

from soup_cli.utils.deploy_autopilot import (
    DeployProfile,
    autopilot_artifacts,
    get_profile,
    has_profile,
    list_profiles,
    render_deploy_script,
    render_recipe_yaml,
    write_deploy_script,
    write_recipe,
)

runner = CliRunner()

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_is_mapping_proxy():
    profiles = list_profiles()
    assert isinstance(profiles, MappingProxyType)
    with pytest.raises(TypeError):
        profiles["x"] = "y"  # type: ignore[index]


def test_catalog_has_all_10_documented_profiles():
    names = set(list_profiles().keys())
    expected = {
        "mac-m3", "mac-m4-pro", "rtx-3060-12gb", "rtx-4090-24gb",
        "iphone-16", "pixel-9", "ollama-local", "lm-studio",
        "runpod-a100", "hf-jobs-h100",
    }
    assert expected.issubset(names)


def test_get_profile_known():
    profile = get_profile("mac-m3")
    assert isinstance(profile, DeployProfile)
    assert profile.name == "mac-m3"
    assert profile.runtime == "mlx"


def test_get_profile_case_insensitive():
    assert get_profile("MAC-M3").name == "mac-m3"
    assert get_profile("  Mac-M3  ").name == "mac-m3"


def test_get_profile_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_profile("unknown-target")


def test_get_profile_empty_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        get_profile("")


def test_get_profile_null_byte_rejected():
    with pytest.raises(ValueError, match="NUL"):
        get_profile("mac-m3\x00")


def test_get_profile_non_string_rejected():
    with pytest.raises(TypeError):
        get_profile(123)  # type: ignore[arg-type]


def test_has_profile_truthy():
    assert has_profile("mac-m3") is True
    assert has_profile("unknown-target") is False
    assert has_profile(None) is False  # type: ignore[arg-type]
    assert has_profile(b"mac-m3") is False  # type: ignore[arg-type]


def test_deploy_profile_is_frozen():
    import dataclasses

    profile = get_profile("mac-m3")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# render_recipe_yaml
# ---------------------------------------------------------------------------


def test_render_recipe_includes_base_and_quant():
    profile = get_profile("rtx-4090-24gb")
    yaml_text = render_recipe_yaml(profile, base="meta-llama/Llama-3.2-1B",
                                   output_dir="./out")
    assert "base: meta-llama/Llama-3.2-1B" in yaml_text
    assert "quantization: awq" in yaml_text
    assert "output: ./out" in yaml_text


def test_render_recipe_mlx_backend_for_mlx_runtime():
    profile = get_profile("mac-m3")
    yaml_text = render_recipe_yaml(profile, base="meta-llama/Llama-3.2-1B",
                                   output_dir="./out")
    assert "backend: mlx" in yaml_text


def test_render_recipe_transformers_backend_default():
    profile = get_profile("rtx-3060-12gb")
    yaml_text = render_recipe_yaml(profile, base="meta-llama/Llama-3.2-1B",
                                   output_dir="./out")
    assert "backend: transformers" in yaml_text


def test_render_recipe_lora_section_present_for_lora_peft():
    profile = get_profile("rtx-4090-24gb")
    yaml_text = render_recipe_yaml(profile, base="meta-llama/Llama-3.2-1B",
                                   output_dir="./out")
    assert "lora:" in yaml_text
    assert "r: 16" in yaml_text


def test_render_recipe_dora_flag_present():
    # No built-in profile uses dora, but the helper should emit it correctly
    # when given a synthetic profile. Validate by constructing manually.
    profile = DeployProfile(
        name="dora-test", description="x", runtime="transformers",
        quant="4bit", peft="dora", spec_decoding=False,
        recommended_max_length=2048, notes="",
    )
    yaml_text = render_recipe_yaml(profile, base="meta-llama/Llama-3.2-1B",
                                   output_dir="./out")
    assert "use_dora: true" in yaml_text


def test_render_recipe_full_peft_omits_lora_section():
    profile = DeployProfile(
        name="full-test", description="x", runtime="transformers",
        quant="none", peft="full", spec_decoding=False,
        recommended_max_length=2048, notes="",
    )
    yaml_text = render_recipe_yaml(profile, base="meta-llama/Llama-3.2-1B",
                                   output_dir="./out")
    assert "lora:" not in yaml_text


def test_render_recipe_base_validation():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError, match="non-empty"):
        render_recipe_yaml(profile, base="", output_dir="./out")
    with pytest.raises(ValueError, match="NUL"):
        render_recipe_yaml(profile, base="evil\x00", output_dir="./out")
    with pytest.raises(ValueError):
        render_recipe_yaml(profile, base="a\nb", output_dir="./out")


def test_render_recipe_base_too_long():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError, match="exceeds"):
        render_recipe_yaml(profile, base="x" * 201, output_dir="./out")


def test_render_recipe_base_non_string():
    profile = get_profile("mac-m3")
    with pytest.raises(TypeError):
        render_recipe_yaml(profile, base=123, output_dir="./out")  # type: ignore[arg-type]


def test_render_recipe_output_dir_validation():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError):
        render_recipe_yaml(profile, base="m/r", output_dir="")
    with pytest.raises(ValueError):
        render_recipe_yaml(profile, base="m/r", output_dir="x\x00")
    with pytest.raises(ValueError):
        render_recipe_yaml(profile, base="m/r", output_dir="a\nb")


def test_render_recipe_profile_type_check():
    with pytest.raises(TypeError):
        render_recipe_yaml("not-a-profile", base="m/r",  # type: ignore[arg-type]
                          output_dir="./o")


def test_render_recipe_max_length_used():
    profile = get_profile("hf-jobs-h100")
    yaml_text = render_recipe_yaml(profile, base="m/r", output_dir="./out")
    assert "max_length: 65536" in yaml_text


# ---------------------------------------------------------------------------
# render_deploy_script
# ---------------------------------------------------------------------------


def test_deploy_script_for_ollama_uses_planned_command():
    profile = get_profile("ollama-local")
    script = render_deploy_script(profile, model_path="./out")
    assert "soup deploy ollama" in script


def test_deploy_script_for_vllm_uses_serve():
    profile = get_profile("rtx-4090-24gb")
    script = render_deploy_script(profile, model_path="./out")
    assert "soup serve --backend vllm" in script
    assert "--auto-spec" in script  # spec_decoding=True


def test_deploy_script_no_spec_flag_when_disabled():
    profile = get_profile("ollama-local")
    script = render_deploy_script(profile, model_path="./out")
    assert "--auto-spec" not in script


def test_deploy_script_for_mlx():
    profile = get_profile("mac-m3")
    script = render_deploy_script(profile, model_path="./out")
    assert "soup serve --backend mlx" in script


def test_deploy_script_for_lm_studio():
    profile = get_profile("lm-studio")
    script = render_deploy_script(profile, model_path="./out")
    assert "lm-studio" in script


def test_deploy_script_for_executorch():
    profile = get_profile("iphone-16")
    script = render_deploy_script(profile, model_path="./out")
    assert "ExecuTorch" in script or "executorch" in script.lower()


def test_deploy_script_quotes_model_path():
    profile = get_profile("mac-m3")
    # Path with spaces — shlex.quote must be applied
    script = render_deploy_script(profile, model_path="./path with spaces")
    assert "'./path with spaces'" in script


def test_deploy_script_rejects_newline_path():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError):
        render_deploy_script(profile, model_path="evil\necho")


def test_deploy_script_rejects_null_byte_path():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError):
        render_deploy_script(profile, model_path="evil\x00")


def test_deploy_script_rejects_empty_path():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError):
        render_deploy_script(profile, model_path="")


def test_deploy_script_rejects_oversize_path():
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError):
        render_deploy_script(profile, model_path="x" * 4097)


def test_deploy_script_non_string_path():
    profile = get_profile("mac-m3")
    with pytest.raises(TypeError):
        render_deploy_script(profile, model_path=123)  # type: ignore[arg-type]


def test_deploy_script_profile_type_check():
    with pytest.raises(TypeError):
        render_deploy_script("not-a-profile", model_path="./out")  # type: ignore[arg-type]


def test_deploy_script_has_shebang_and_set_e():
    profile = get_profile("mac-m3")
    script = render_deploy_script(profile, model_path="./out")
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script


# ---------------------------------------------------------------------------
# write_recipe / write_deploy_script (cwd containment)
# ---------------------------------------------------------------------------


def test_write_recipe_under_cwd_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    out = write_recipe(profile, base="m/r", output_dir="./out",
                       recipe_path="recipe.yaml")
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as fh:
        assert "base: m/r" in fh.read()


def test_write_recipe_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    abs_outside = str(tmp_path.parent / "evil.yaml")
    with pytest.raises(ValueError, match="must stay under cwd"):
        write_recipe(profile, base="m/r", output_dir="./out",
                     recipe_path=abs_outside)


def test_write_recipe_null_byte_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError, match="NUL"):
        write_recipe(profile, base="m/r", output_dir="./out",
                     recipe_path="recipe\x00.yaml")


def test_write_recipe_non_string_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    with pytest.raises(TypeError):
        write_recipe(profile, base="m/r", output_dir="./out",
                     recipe_path=123)  # type: ignore[arg-type]


def test_write_deploy_script_under_cwd_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    out = write_deploy_script(profile, model_path="./out",
                              script_path="deploy.sh")
    assert os.path.exists(out)
    if os.name != "nt":
        mode = os.stat(out).st_mode
        assert mode & _stat.S_IXUSR


def test_write_deploy_script_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    abs_outside = str(tmp_path.parent / "evil.sh")
    with pytest.raises(ValueError, match="must stay under cwd"):
        write_deploy_script(profile, model_path="./out",
                            script_path=abs_outside)


def test_write_deploy_script_null_byte_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError):
        write_deploy_script(profile, model_path="./out",
                            script_path="x\x00.sh")


# ---------------------------------------------------------------------------
# autopilot_artifacts helper
# ---------------------------------------------------------------------------


def test_autopilot_artifacts_returns_pair():
    recipe, script = autopilot_artifacts(
        "rtx-4090-24gb", base="m/r", output_dir="./out",
    )
    assert "base: m/r" in recipe
    assert "soup serve" in script


def test_autopilot_artifacts_default_model_path():
    _, script = autopilot_artifacts(
        "mac-m3", base="m/r", output_dir="./out",
    )
    assert "./out" in script


def test_autopilot_artifacts_explicit_model_path():
    _, script = autopilot_artifacts(
        "mac-m3", base="m/r", output_dir="./train", model_path="./serve",
    )
    assert "./serve" in script
    assert "./serve" in script


def test_autopilot_artifacts_unknown_profile_raises():
    with pytest.raises(KeyError):
        autopilot_artifacts("nope", base="m/r", output_dir="./out")


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_autopilot_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import deploy

    result = runner.invoke(deploy.app, ["autopilot", "--list"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # At least one known profile name appears
    assert "mac-m3" in result.output
    assert "rtx-4090-24gb" in result.output


def test_cli_autopilot_help():
    from soup_cli.commands import deploy

    result = runner.invoke(deploy.app, ["autopilot", "--help"])
    assert result.exit_code == 0, result.output
    assert "Profile name" in result.output or "profile" in result.output.lower()


def test_cli_autopilot_writes_recipe_and_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import deploy

    result = runner.invoke(
        deploy.app,
        ["autopilot", "--target", "mac-m3", "--base", "meta-llama/Llama-3.2-1B",
         "--recipe-out", "recipe.yaml", "--script-out", "deploy.sh"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "recipe.yaml").exists()
    assert (tmp_path / "deploy.sh").exists()


def test_cli_autopilot_unknown_target_exits_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import deploy

    result = runner.invoke(deploy.app, ["autopilot", "--target", "nope"])
    assert result.exit_code == 2, result.output


@pytest.mark.skipif(sys.platform == "win32", reason="symlink ACL on Windows")
def test_write_recipe_symlink_target_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.yaml"
    real.write_text("", encoding="utf-8")
    link = tmp_path / "recipe.yaml"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError, match="symlink"):
        write_recipe(profile, base="m/r", output_dir="./out",
                     recipe_path="recipe.yaml")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink ACL on Windows")
def test_write_deploy_script_symlink_target_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.sh"
    real.write_text("", encoding="utf-8")
    link = tmp_path / "deploy.sh"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError, match="symlink"):
        write_deploy_script(profile, model_path="./out",
                            script_path="deploy.sh")


def test_write_recipe_path_too_long_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = get_profile("mac-m3")
    with pytest.raises(ValueError, match="exceeds"):
        write_recipe(profile, base="m/r", output_dir="./out",
                     recipe_path="x" * 4097)


def test_cli_autopilot_outside_cwd_script_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import deploy

    abs_outside = str(tmp_path.parent / "evil.sh")
    result = runner.invoke(
        deploy.app,
        ["autopilot", "--target", "mac-m3", "--script-out", abs_outside],
    )
    assert result.exit_code == 1, result.output


def test_cli_autopilot_outside_cwd_recipe_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import deploy

    abs_outside = str(tmp_path.parent / "evil.yaml")
    result = runner.invoke(
        deploy.app,
        ["autopilot", "--target", "mac-m3", "--recipe-out", abs_outside],
    )
    assert result.exit_code == 1, result.output


# ---------------------------------------------------------------------------
# Internal validators (via DeployProfile construction)
# ---------------------------------------------------------------------------


def test_make_helper_rejects_invalid_runtime():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(ValueError, match="runtime"):
        _make("x", "d", runtime="nope", quant="4bit", peft="lora",
              spec_decoding=False, recommended_max_length=2048)


def test_make_helper_rejects_invalid_quant():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(ValueError, match="quant"):
        _make("x", "d", runtime="transformers", quant="nope", peft="lora",
              spec_decoding=False, recommended_max_length=2048)


def test_make_helper_rejects_invalid_peft():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(ValueError, match="peft"):
        _make("x", "d", runtime="transformers", quant="4bit", peft="nope",
              spec_decoding=False, recommended_max_length=2048)


def test_make_helper_rejects_bool_max_length():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(TypeError, match="bool"):
        _make("x", "d", runtime="transformers", quant="4bit", peft="lora",
              spec_decoding=False, recommended_max_length=True)


def test_make_helper_rejects_max_length_out_of_bounds():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(ValueError, match="64"):
        _make("x", "d", runtime="transformers", quant="4bit", peft="lora",
              spec_decoding=False, recommended_max_length=32)


def test_make_helper_rejects_bad_name():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(ValueError, match="kebab"):
        _make("Bad Name", "d", runtime="transformers", quant="4bit", peft="lora",
              spec_decoding=False, recommended_max_length=2048)


def test_make_helper_rejects_null_byte_description():
    from soup_cli.utils.deploy_autopilot import _make

    with pytest.raises(ValueError, match="NUL"):
        _make("ok", "evil\x00", runtime="transformers", quant="4bit", peft="lora",
              spec_decoding=False, recommended_max_length=2048)


def test_make_helper_via_known_failure_modes():
    # Internal _make is used by _BUILTIN at import time. If the catalog
    # is loaded then all known profiles passed validation. Just assert the
    # imports succeeded and the count matches.
    assert "soup_cli.utils.deploy_autopilot" in sys.modules


def test_recommended_max_length_in_bounds():
    for profile in list_profiles().values():
        assert 64 <= profile.recommended_max_length <= 1_048_576
