"""The L2L run recipe: the closed codec gate, the effective k and the resume check.

Seam: ``soup_cli.utils.l2l_recipe`` (pure logic and one JSON file in the output directory).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from soup_cli.utils import l2l_recipe
from soup_cli.utils.l2l_recipe import (
    AVAILABLE_CODECS,
    RECIPE_NAME,
    load_recipe,
    resolve_l2l_recipe,
    validate_l2l_options,
    validate_resume,
    write_recipe,
)


def _tcfg(**overrides):
    values = dict(
        l2l=True,
        l2l_activation_codec="none",
        l2l_auto_k=False,
        l2l_spill_dir=None,
        gradient_accumulation_steps=8,
        batch_size=1,
        stream_expert_layout="fused",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_launch_accepts_the_none_codec_only():
    assert AVAILABLE_CODECS == ("none",)


def test_fp8_is_refused_with_the_exact_message():
    with pytest.raises(ValueError) as caught:
        validate_l2l_options(_tcfg(l2l_activation_codec="fp8_e4m3"))
    assert str(caught.value) == (
        "training.l2l_activation_codec='fp8_e4m3' is not available yet: the launch "
        "supports 'none'. Set it to 'none'."
    )


def test_resolving_a_recipe_refuses_fp8_too():
    with pytest.raises(ValueError, match="is not available yet"):
        resolve_l2l_recipe(_tcfg(l2l_activation_codec="fp8_e4m3"))


def test_the_gate_opens_only_through_available_codecs(monkeypatch):
    monkeypatch.setattr(l2l_recipe, "AVAILABLE_CODECS", ("none", "fp8_e4m3"))
    validate_l2l_options(_tcfg(l2l_activation_codec="fp8_e4m3"))


def test_the_effective_k_is_the_configured_accumulation():
    recipe = resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=16))
    assert (recipe.requested_k, recipe.effective_k, recipe.auto_k) == (16, 16, False)
    assert recipe.activation_codec == "none"


def test_auto_k_without_per_expert_streaming_is_refused_with_the_field_named():
    with pytest.raises(ValueError, match=r"training\.l2l_auto_k requires DeepSeek-V3 per_expert"):
        validate_l2l_options(_tcfg(l2l_auto_k=True))


def test_auto_k_with_a_spill_dir_is_refused():
    tcfg = _tcfg(l2l_auto_k=True, stream_expert_layout="per_expert", l2l_spill_dir="spill")
    with pytest.raises(ValueError, match=r"training\.l2l_auto_k uses RAM only"):
        validate_l2l_options(tcfg)


def test_auto_k_is_refused_until_moe_l2l_is_available():
    tcfg = _tcfg(l2l_auto_k=True, stream_expert_layout="per_expert")
    with pytest.raises(ValueError, match=r"l2l_auto_k plans the window of a MoE model"):
        validate_l2l_options(tcfg)
    with pytest.raises(ValueError, match=r"l2l_auto_k plans the window of a MoE model"):
        resolve_l2l_recipe(tcfg, effective_k=2)


def test_the_recipe_round_trips_through_the_output_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    recipe = resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=8))
    path = write_recipe(out, recipe)
    assert path.endswith(RECIPE_NAME)
    assert load_recipe(out) == recipe


def test_a_missing_recipe_loads_as_none(tmp_path):
    assert load_recipe(tmp_path) is None


def test_a_malformed_recipe_is_refused_with_its_path(tmp_path):
    (tmp_path / RECIPE_NAME).write_text(json.dumps({"effective_k": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"is not a valid L2L recipe: recipe is missing"):
        load_recipe(tmp_path)


def test_resume_accepts_a_directory_without_a_recipe(tmp_path):
    validate_resume(tmp_path, resolve_l2l_recipe(_tcfg()))


def test_resume_refuses_another_k(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_recipe(tmp_path, resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=4)))
    with pytest.raises(ValueError, match=r"cannot resume: .*effective_k=4.*effective_k=8"):
        validate_resume(tmp_path, resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=8)))


def test_resume_refuses_another_codec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as opened:
        opened.setattr(l2l_recipe, "AVAILABLE_CODECS", ("none", "fp8_e4m3"))
        write_recipe(tmp_path, resolve_l2l_recipe(_tcfg(l2l_activation_codec="fp8_e4m3")))
    with pytest.raises(ValueError, match=r"cannot resume: .*activation_codec='fp8_e4m3'.*'none'"):
        validate_resume(tmp_path, resolve_l2l_recipe(_tcfg()))


def test_resume_accepts_the_same_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = resolve_l2l_recipe(_tcfg())
    write_recipe(tmp_path, recipe)
    validate_resume(tmp_path, recipe)
