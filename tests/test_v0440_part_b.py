"""v0.44.0 Part B — UX fix tests.

Covers: graceful_save (Ctrl+C SIGINT), checkpoint_trigger, shortcuts,
onboarding wizard.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from soup_cli.utils.checkpoint_trigger import (
    TRIGGER_FILENAME,
    consume_trigger,
    should_save_now,
    trigger_path,
    write_trigger,
)
from soup_cli.utils.graceful_save import GracefulSaveHandler
from soup_cli.utils.onboarding import (
    VALID_QUANT,
    VALID_TASKS,
    render_onboarding_yaml,
)
from soup_cli.utils.shortcuts import (
    build_for_current_platform,
    build_linux_desktop_entry,
    build_macos_command_file,
    build_windows_cmd,
    detect_platform,
)

# --- graceful_save (SIGINT handler) -----------------------------------------

class _FakeState:
    should_save = False
    should_training_stop = False


def test_graceful_save_first_sigint_triggers_save():
    handler = GracefulSaveHandler()
    state = _FakeState()
    handler.attach_state(state)
    handler._handle_sigint(2, None)
    assert state.should_save is True
    assert state.should_training_stop is False


def test_graceful_save_second_sigint_stops_training():
    handler = GracefulSaveHandler()
    state = _FakeState()
    handler.attach_state(state)
    handler._handle_sigint(2, None)
    handler._handle_sigint(2, None)
    assert state.should_training_stop is True


def test_graceful_save_no_state_first_signal_keyboardinterrupt():
    handler = GracefulSaveHandler()
    # No state attached → 1st signal raises since count==1 and state is None.
    with pytest.raises(KeyboardInterrupt):
        handler._handle_sigint(2, None)


def test_graceful_save_install_idempotent(monkeypatch):
    import signal as _signal

    calls = []

    def fake_signal(signum, handler):
        calls.append(signum)
        return _signal.SIG_DFL

    monkeypatch.setattr(_signal, "signal", fake_signal)
    handler = GracefulSaveHandler()
    handler.install()
    handler.install()
    assert len(calls) == 1


def test_graceful_save_state_attribute_missing_does_not_crash():
    handler = GracefulSaveHandler()

    class _Empty:
        pass

    handler.attach_state(_Empty())
    # First SIGINT: should_save assignment AttributeError swallowed.
    handler._handle_sigint(2, None)


# --- checkpoint_trigger -----------------------------------------------------

def test_trigger_path_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    path = trigger_path(str(out))
    assert path.endswith(TRIGGER_FILENAME)


def test_trigger_path_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path.parent / "definitely-not-under-cwd"
    with pytest.raises(ValueError, match="outside cwd"):
        trigger_path(str(other))


def test_trigger_path_rejects_null_byte(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        trigger_path("./out\x00bad")


def test_trigger_path_rejects_empty():
    with pytest.raises(ValueError):
        trigger_path("")


def test_trigger_path_rejects_non_string():
    with pytest.raises(TypeError):
        trigger_path(123)  # type: ignore[arg-type]


def test_should_save_now_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert should_save_now(str(out)) is False


def test_write_and_consume_trigger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    write_trigger(str(out))
    assert should_save_now(str(out)) is True
    assert consume_trigger(str(out)) is True
    assert should_save_now(str(out)) is False
    # Consuming a non-existent trigger returns False, not an error.
    assert consume_trigger(str(out)) is False


def test_write_trigger_rejects_null_byte_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError):
        write_trigger(str(out), contents="bad\x00byte")


def test_write_trigger_rejects_non_string_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(TypeError):
        write_trigger(str(out), contents=123)  # type: ignore[arg-type]


def test_should_save_now_false_for_invalid_input():
    assert should_save_now("") is False
    assert should_save_now(123) is False  # type: ignore[arg-type]


# --- shortcuts --------------------------------------------------------------

def test_detect_platform_known():
    assert detect_platform() in {"linux", "darwin", "windows", "unknown"}


def test_build_linux_desktop_entry():
    spec = build_linux_desktop_entry(name="Soup CLI", command="soup ui")
    assert spec.platform == "linux"
    assert spec.filename == "soup-cli.desktop"
    assert "Exec=soup ui" in spec.content
    assert "Categories=Development;" in spec.content


def test_build_macos_command_file():
    spec = build_macos_command_file(name="Soup", command="soup ui")
    assert spec.platform == "darwin"
    assert spec.filename.endswith(".command")
    assert spec.content.startswith("#!/usr/bin/env bash")


def test_build_windows_cmd():
    spec = build_windows_cmd(name="Soup", command="soup ui")
    assert spec.platform == "windows"
    assert spec.filename.endswith(".cmd")
    assert spec.content.startswith("@echo off")


def test_shortcut_rejects_disallowed_chars_in_name():
    with pytest.raises(ValueError):
        build_linux_desktop_entry(name="bad`name", command="x")


def test_shortcut_rejects_control_in_command():
    with pytest.raises(ValueError):
        build_linux_desktop_entry(name="ok", command="x\nrm -rf /")


def test_shortcut_rejects_empty_command():
    with pytest.raises(ValueError):
        build_linux_desktop_entry(name="ok", command="")


def test_shortcut_rejects_oversize_name():
    with pytest.raises(ValueError):
        build_linux_desktop_entry(name="x" * 100, command="echo")


def test_shortcut_rejects_non_string_name():
    with pytest.raises(TypeError):
        build_linux_desktop_entry(name=123, command="x")  # type: ignore[arg-type]


def test_build_for_current_platform_returns_a_spec():
    plat = detect_platform()
    if plat == "unknown":
        with pytest.raises(NotImplementedError):
            build_for_current_platform(name="Soup", command="soup ui")
    else:
        spec = build_for_current_platform(name="Soup", command="soup ui")
        assert spec.platform == plat


# --- onboarding wizard ------------------------------------------------------

def test_render_onboarding_yaml_happy_path():
    text = render_onboarding_yaml(
        {
            "base": "meta-llama/Llama-3.2-1B",
            "dataset": "./train.jsonl",
            "task": "sft",
            "quantization": "4bit",
            "epochs": 3,
        }
    )
    cfg = yaml.safe_load(text)
    assert cfg["base"] == "meta-llama/Llama-3.2-1B"
    assert cfg["task"] == "sft"
    assert cfg["training"]["epochs"] == 3
    assert cfg["training"]["quantization"] == "4bit"


def test_render_onboarding_yaml_default_quant_and_output():
    text = render_onboarding_yaml(
        {
            "base": "x/y",
            "dataset": "./d.jsonl",
            "task": "dpo",
            "epochs": 1,
        }
    )
    cfg = yaml.safe_load(text)
    assert cfg["training"]["quantization"] == "4bit"
    assert cfg["output"] == "./out"
    assert cfg["training"]["batch_size"] == "auto"


def test_render_onboarding_yaml_rejects_unknown_task():
    with pytest.raises(ValueError, match="task must be"):
        render_onboarding_yaml(
            {"base": "x/y", "dataset": "d", "task": "bogus", "epochs": 1}
        )


def test_render_onboarding_yaml_rejects_unknown_quant():
    with pytest.raises(ValueError, match="quantization"):
        render_onboarding_yaml(
            {
                "base": "x/y",
                "dataset": "d",
                "task": "sft",
                "quantization": "16bit",
                "epochs": 1,
            }
        )


def test_render_onboarding_yaml_rejects_bad_epochs():
    with pytest.raises(ValueError):
        render_onboarding_yaml(
            {"base": "x/y", "dataset": "d", "task": "sft", "epochs": 0}
        )
    with pytest.raises(ValueError):
        render_onboarding_yaml(
            {"base": "x/y", "dataset": "d", "task": "sft", "epochs": 99}
        )
    with pytest.raises(TypeError):
        render_onboarding_yaml(
            {"base": "x/y", "dataset": "d", "task": "sft", "epochs": True}
        )


def test_render_onboarding_yaml_rejects_null_byte():
    with pytest.raises(ValueError):
        render_onboarding_yaml(
            {
                "base": "x\x00/y",
                "dataset": "d",
                "task": "sft",
                "epochs": 1,
            }
        )


def test_render_onboarding_yaml_rejects_oversize_base():
    with pytest.raises(ValueError):
        render_onboarding_yaml(
            {
                "base": "a" * 1000,
                "dataset": "d",
                "task": "sft",
                "epochs": 1,
            }
        )


def test_render_onboarding_yaml_explicit_batch_size():
    text = render_onboarding_yaml(
        {
            "base": "x/y",
            "dataset": "d",
            "task": "sft",
            "epochs": 1,
            "batch_size": 8,
        }
    )
    cfg = yaml.safe_load(text)
    assert cfg["training"]["batch_size"] == 8


def test_render_onboarding_yaml_rejects_bool_batch_size():
    with pytest.raises(TypeError):
        render_onboarding_yaml(
            {
                "base": "x/y",
                "dataset": "d",
                "task": "sft",
                "epochs": 1,
                "batch_size": True,
            }
        )


def test_render_onboarding_yaml_rejects_non_dict():
    with pytest.raises(TypeError):
        render_onboarding_yaml("not a dict")  # type: ignore[arg-type]


def test_onboarding_constants_align():
    # Sanity: constants exported don't drift.
    assert "sft" in VALID_TASKS
    assert "preference" in VALID_TASKS
    assert "4bit" in VALID_QUANT
    assert "none" in VALID_QUANT


def test_path_exists(tmp_path, monkeypatch):
    """Sanity: the test temp dir is real and matches Path semantics."""
    monkeypatch.chdir(tmp_path)
    assert os.path.isdir(Path.cwd())
