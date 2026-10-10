"""Continuing a QuEST artifact must preserve its measured clipping table."""

import copy
from types import SimpleNamespace

import pytest

from soup_cli.trainer.sft import SFTTrainerWrapper
from soup_cli.utils import quest


def wrapper_for(path, monkeypatch, declaration):
    from soup_cli.trainer import stream_setup

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.deepspeed_config = None
    wrapper.fsdp_config = None
    wrapper.model = SimpleNamespace(config=SimpleNamespace(soup_quest=declaration))
    wrapper.config = SimpleNamespace(base=str(path))
    wrapper._quest_base_identity_before = "local-sha256:" + "a" * 64
    monkeypatch.setattr(stream_setup, "_distributed_launch", lambda: False)
    monkeypatch.setattr(quest, "validate_cuda_hardware", lambda: ("RTX 3090", (8, 6)))
    monkeypatch.setattr(quest, "resolve_base_model_identity", lambda _: "local-sha256:" + "a" * 64)
    monkeypatch.setattr(
        quest,
        "calibrate_activation_scales",
        lambda *_: pytest.fail("existing artifact was recalibrated"),
    )
    return wrapper


def metadata():
    return quest.build_metadata(
        activation_scales=dict.fromkeys(quest.EXPECTED_MODULES, 3.0),
        base_model="local-sha256:" + "b" * 64,
        calibration_sha256="c" * 64,
    )


def test_warm_start_restores_the_declared_route_without_new_calibration(tmp_path, monkeypatch):
    original = metadata()
    quest.write_metadata(tmp_path, original)
    wrapper = wrapper_for(tmp_path, monkeypatch, copy.deepcopy(original))
    restored = []
    monkeypatch.setattr(quest, "restore_mixed_quest", lambda model, value: restored.append(value))
    wrapper._setup_quest([])
    assert restored == [original]
    assert wrapper._quest_metadata == original
    assert wrapper.model.config.soup_quest == original


def test_warm_start_rejects_a_declaration_without_a_local_artifact(tmp_path, monkeypatch):
    original = metadata()
    wrapper = wrapper_for(tmp_path / "org" / "hub-model", monkeypatch, copy.deepcopy(original))
    monkeypatch.setattr(
        quest, "restore_mixed_quest", lambda *_: pytest.fail("non-local artifact restored")
    )
    with pytest.raises(ValueError, match="requires a local artifact"):
        wrapper._setup_quest([])


@pytest.mark.parametrize("mutation", ["missing-sidecar", "missing-config", "different-config"])
def test_warm_start_rejects_incomplete_or_different_declarations(tmp_path, monkeypatch, mutation):
    original = metadata()
    if mutation != "missing-sidecar":
        quest.write_metadata(tmp_path, original)
    declaration = None if mutation == "missing-config" else copy.deepcopy(original)
    if mutation == "different-config":
        declaration["activation_scales"][quest.EXPECTED_MODULES[0]] = 4.0
    wrapper = wrapper_for(tmp_path, monkeypatch, declaration)
    monkeypatch.setattr(
        quest, "restore_mixed_quest", lambda *_: pytest.fail("bad artifact restored")
    )
    with pytest.raises((ValueError, FileNotFoundError), match="(?i)(sidecar|metadata|declaration)"):
        wrapper._setup_quest([])


def test_warm_start_rejects_base_changed_during_loading(tmp_path, monkeypatch):
    original = metadata()
    quest.write_metadata(tmp_path, original)
    wrapper = wrapper_for(tmp_path, monkeypatch, original)
    wrapper._quest_base_identity_before = "local-sha256:" + "d" * 64
    monkeypatch.setattr(
        quest, "restore_mixed_quest", lambda *_: pytest.fail("changed base restored")
    )
    with pytest.raises(ValueError, match="base changed"):
        wrapper._setup_quest([])
