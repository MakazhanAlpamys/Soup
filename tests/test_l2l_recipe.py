"""The L2L run recipe: the closed codec gate, the effective k and the resume check.

Seam: ``soup_cli.utils.l2l_recipe`` (pure logic and one JSON file in the output directory).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from soup_cli.utils import l2l_recipe
from soup_cli.utils.l2l_recipe import (
    AVAILABLE_CODECS,
    RECIPE_NAME,
    load_recipe,
    refuse_distributed_l2l,
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


_AUTO_K = (
    "training.l2l_auto_k requires DeepSeek-V3 per_expert streaming and a fixed batch_size"
)


@pytest.mark.parametrize("layout,batch_size", [("fused", 1), ("per_expert", "auto")])
def test_auto_k_requires_per_expert_streaming_and_a_fixed_batch(layout, batch_size):
    tcfg = _tcfg(l2l_auto_k=True, stream_expert_layout=layout, batch_size=batch_size)
    with pytest.raises(ValueError) as caught:
        validate_l2l_options(tcfg)
    assert str(caught.value) == _AUTO_K
    with pytest.raises(ValueError) as caught:
        resolve_l2l_recipe(tcfg, effective_k=2)
    assert str(caught.value) == _AUTO_K


@pytest.mark.parametrize("effective_k", [None, 0, 9, True])
def test_auto_k_requires_a_resolved_window_within_the_requested_limit(effective_k):
    with pytest.raises(ValueError) as caught:
        resolve_l2l_recipe(
            _tcfg(l2l_auto_k=True, stream_expert_layout="per_expert"), effective_k=effective_k
        )
    assert str(caught.value) == (
        "training.l2l_auto_k requires a resolved, RAM-fitting effective k"
    )


def test_auto_k_refuses_an_explicit_spill_directory():
    with pytest.raises(ValueError) as caught:
        validate_l2l_options(
            _tcfg(l2l_auto_k=True, stream_expert_layout="per_expert", l2l_spill_dir="spill")
        )
    assert str(caught.value) == "training.l2l_auto_k uses RAM only; unset training.l2l_spill_dir"


def test_auto_k_refuses_an_mbr_group_that_exceeds_the_resolved_window():
    with pytest.raises(ValueError) as caught:
        resolve_l2l_recipe(
            _tcfg(l2l_auto_k=True, stream_expert_layout="per_expert", l2l_mbr_group_size=4),
            effective_k=2,
        )
    assert str(caught.value) == (
        "training.l2l_mbr_group_size=4 exceeds the effective L2L window of 2 micro-batches"
    )


def test_a_spill_dir_outside_the_allowed_roots_is_refused(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill
    from soup_cli.utils.terminal import for_terminal

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = str(tmp_path / "outside")
    monkeypatch.setattr(l2l_spill, "_allowed_roots", lambda: (str(allowed),))
    with pytest.raises(ValueError) as caught:
        validate_l2l_options(_tcfg(l2l_spill_dir=outside))
    reason = (
        f"{for_terminal(outside)} resolves to {for_terminal(os.path.realpath(outside))}, "
        "which is outside $HOME, $CWD and $TMPDIR, the only places the spill may be written"
    )
    assert str(caught.value) == (
        f"training.l2l_spill_dir is refused: {reason}. Set it to a directory under "
        "$HOME, the working directory or the temp directory."
    )
    assert not os.path.exists(outside)
    validate_l2l_options(_tcfg(l2l_spill_dir=str(allowed / "spill")))


def test_the_distributed_refusal_names_l2l_and_the_distributed_stream():
    refuse_distributed_l2l(False)
    with pytest.raises(ValueError) as caught:
        refuse_distributed_l2l(True)
    assert str(caught.value) == (
        "training.l2l cannot run on a distributed layer stream: the L2L step runs its "
        "own backward, so the adapter-gradient average between ranks would be skipped. "
        "Run one process, or set training.l2l to false."
    )


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
