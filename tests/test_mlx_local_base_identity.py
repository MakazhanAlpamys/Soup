"""Native MLX metadata production -> reading across working directories (#2989).

Portable tests execute real metadata writers/readers. Only optional native
computation is replaced for SFT; no MLX or Torch installation is required.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from soup_cli.config.schema import LoraConfig, SoupConfig
from soup_cli.trainer.mlx_sft import build_mlx_adapter_config
from soup_cli.utils.mlx_generation import read_native_adapter
from tests.conftest import make_cli_runner, strip_ansi
from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx, _mlx_wrapper


def _check_sft_local_base(tmp_path, monkeypatch, base_kind):
    work = tmp_path / "work"
    work.mkdir()
    base = work / "base"
    base.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(work)
    _install_fake_mlx(monkeypatch)

    def save_adapter(**kwargs):
        Path(kwargs["args"].adapter_file).write_bytes(b"native boundary")

    sys.modules["mlx_lm.tuner.trainer"].train = save_adapter
    adapter = work / "trained"
    wrapper = _mlx_wrapper(adapter, epochs=1, batch_size=1, rewind_log=False)
    model = "./base"
    if base_kind == "absolute":
        model = str(base)
    elif base_kind == "hub":
        model = "org/name"
    elif base_kind == "symlink":
        (work / "link").symlink_to(base, target_is_directory=True)
        model = "./link"
    wrapper.config.base = model
    wrapper.model = _FakeMlxModel()
    wrapper.model.layers = [object()]
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hello"}], "val": []}
    wrapper.train()

    monkeypatch.chdir(other)
    expected = "org/name" if base_kind == "hub" else str(base)
    assert read_native_adapter(adapter)["model"] == expected
    assert json.loads((adapter / "adapter_config.json").read_text())["model"] == expected
    assert wrapper.config.base == model


@pytest.mark.parametrize("base_kind", ["relative", "absolute", "hub"])
def test_sft_local_base_survives_a_different_cwd(tmp_path, monkeypatch, base_kind):
    _check_sft_local_base(tmp_path, monkeypatch, base_kind)


@pytest.mark.requires_symlink
def test_sft_symlink_base_survives_a_different_cwd(tmp_path, monkeypatch):
    _check_sft_local_base(tmp_path, monkeypatch, "symlink")


def _adapter(path, model):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({
        "model": model, "fine_tune_type": "lora", "num_layers": 1,
        "lora_parameters": {"rank": 4, "scale": 2, "dropout": 0,
                            "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    }), encoding="utf-8")
    (path / "adapters.safetensors").write_bytes(b"native boundary")
    return path


def test_legacy_absolute_override_wins_over_a_same_named_cwd_base(
    tmp_path, monkeypatch, capsys,
):
    original = tmp_path / "original"
    original.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "base").mkdir()
    adapter = _adapter(tmp_path / "adapter", "./base")
    monkeypatch.chdir(other)
    monkeypatch.setenv("COLUMNS", "1000")
    config = read_native_adapter(adapter, str(original))
    assert config["model"] == str(original)
    output = " ".join(strip_ansi(capsys.readouterr().out).split())
    assert f"Native MLX legacy relative base './base' resolved to {str(original)!r}" in output
    assert json.loads((adapter / "adapter_config.json").read_text())["model"] == "./base"


def _plain(text):
    return " ".join(strip_ansi(text).split())


@pytest.mark.parametrize("model,local", [
    ("./base", True), ("../base", True), ("base", True),
    ("missing", False), ("org/name", False), ("absolute", True),
])
def test_builder_classifies_local_directories_and_preserves_hub_ids(
    tmp_path, monkeypatch, model, local,
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "base").mkdir()
    (tmp_path / "base").mkdir()
    monkeypatch.chdir(cwd)
    if model == "absolute":
        model = str(cwd / "base")
    config = build_mlx_adapter_config(LoraConfig(r=4), adapter_path="adapter", model=model)
    expected = str(tmp_path / "base") if model == "../base" else str(cwd / "base")
    assert config["model"] == (expected if local else model)


def _check_preference_and_grpo_writer(
    tmp_path, monkeypatch, task, base_kind,
):
    from soup_cli.trainer.mlx_grpo import resolve_units
    from soup_cli.trainer.mlx_optim import plan_optimizer

    work = tmp_path / "work"
    work.mkdir()
    base = work / "base"
    base.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(work)
    model = "./base"
    if base_kind == "absolute":
        model = str(base)
    elif base_kind == "hub":
        model = "org/name"
    elif base_kind == "symlink":
        (work / "link").symlink_to(base, target_is_directory=True)
        model = "./link"
    cfg = SoupConfig.model_validate({
        "base": model, "backend": "mlx", "task": task, "output": "adapter",
        "data": {"train": "rows.jsonl", "max_length": 64, "val_split": 0},
        "training": {"quantization": "none", "batch_size": 2, "rewind_log": False,
                     "num_generations": 2, "reward_fn": "format",
                     "lora": {"r": 4, "alpha": 8, "dropout": 0}},
    })
    module = importlib.import_module(f"soup_cli.trainer.mlx_{task}")
    name = "SimPO" if task == "simpo" else task.upper()
    wrapper = getattr(module, f"MLX{name}TrainerWrapper")(cfg)
    wrapper.model = _FakeMlxModel()
    wrapper.model.layers = [object()]
    if task == "grpo":
        wrapper._units = resolve_units(batch_size=2, accum=1, num_generations=2)
        wrapper._reward_name = "format"
    adapter = work / "adapter"
    adapter.mkdir()
    wrapper._write_adapter_config(
        adapter, batch_size=2, iters=1, accum=1,
        optimizer_plan=plan_optimizer(lr=0.001, optimizer="adamw", scheduler="constant",
                                      warmup_ratio=0, weight_decay=0, total_updates=1),
        seed=42, data_seed=42,
    )
    (adapter / "adapters.safetensors").write_bytes(b"native boundary")
    monkeypatch.chdir(other)
    expected = "org/name" if base_kind == "hub" else str(base)
    assert read_native_adapter(adapter)["model"] == expected
    assert json.loads((adapter / "adapter_config.json").read_text())["model"] == expected
    assert cfg.base == model


@pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto", "grpo"])
@pytest.mark.parametrize("base_kind", ["relative", "absolute", "hub"])
def test_preference_and_grpo_writers_feed_the_cross_cwd_reader(
    tmp_path, monkeypatch, task, base_kind,
):
    _check_preference_and_grpo_writer(tmp_path, monkeypatch, task, base_kind)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto", "grpo"])
def test_preference_and_grpo_symlink_bases_feed_the_cross_cwd_reader(
    tmp_path, monkeypatch, task,
):
    _check_preference_and_grpo_writer(tmp_path, monkeypatch, task, "symlink")


@pytest.mark.parametrize("stored,present,local", [
    ("./base", True, True), ("../base", True, True), ("base", True, True),
    ("base", False, False), ("org/name", False, False), ("absolute", True, False),
])
def test_legacy_identity_classification_table(
    tmp_path, monkeypatch, capsys, stored, present, local,
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    selected = tmp_path / "base" if stored == "../base" else cwd / "base"
    if present:
        selected.mkdir()
    if stored == "absolute":
        stored = str(selected)
    adapter = _adapter(tmp_path / "adapter", stored)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("COLUMNS", "1000")
    result = read_native_adapter(adapter)
    assert result["model"] == (str(selected) if local else stored)
    output = _plain(capsys.readouterr().out)
    if local:
        assert f"Native MLX legacy relative base {stored!r} resolved to {str(selected)!r}" in output
    else:
        assert output == ""


@pytest.mark.parametrize("stored", ["./base", "../base"])
@pytest.mark.parametrize("override", ["none", "missing-absolute", "file-absolute", "relative"])
def test_legacy_missing_or_invalid_resolution_is_refused(
    tmp_path, monkeypatch, stored, override,
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    adapter = _adapter(tmp_path / "adapter", stored)
    base = None
    if override == "missing-absolute":
        base = str(tmp_path / "missing")
    elif override == "file-absolute":
        (tmp_path / "file").write_text("not a directory")
        base = str(tmp_path / "file")
    elif override == "relative":
        base = stored
    with pytest.raises(ValueError) as error:
        read_native_adapter(adapter, base)
    message = str(error.value)
    assert stored in message
    assert "--base" in message and "absolute" in message and "directory" in message


@pytest.mark.parametrize("present", [False, True])
def test_legacy_absolute_override_works_with_or_without_cwd_base(tmp_path, monkeypatch, present):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    if present:
        (cwd / "base").mkdir()
    original = tmp_path / "original"
    original.mkdir()
    adapter = _adapter(tmp_path / "adapter", "./base")
    monkeypatch.chdir(cwd)
    assert read_native_adapter(adapter, str(original))["model"] == str(original)


@pytest.mark.parametrize("stored", ["org/name", "base", "absolute", "missing-absolute"])
def test_qualified_or_hub_identity_cannot_be_overridden(tmp_path, monkeypatch, stored):
    monkeypatch.chdir(tmp_path)
    actual = tmp_path / "actual"
    actual.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    if stored == "absolute":
        stored = str(actual)
    elif stored == "missing-absolute":
        stored = str(tmp_path / "gone")
    adapter = _adapter(tmp_path / "adapter", stored)
    with pytest.raises(ValueError, match="does not match the native MLX metadata model identity"):
        read_native_adapter(adapter, str(other))
    assert read_native_adapter(adapter)["model"] == stored


def test_relative_override_is_still_compared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "other").mkdir()
    adapter = _adapter(tmp_path / "adapter", "./base")
    assert read_native_adapter(adapter, "base")["model"] == str(tmp_path / "base")
    with pytest.raises(ValueError, match="does not match"):
        read_native_adapter(adapter, "./other")


@pytest.mark.requires_symlink
def test_absolute_symlink_override_resolves_to_the_real_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    adapter = _adapter(tmp_path / "adapter", "./base")
    (tmp_path / "alias").symlink_to(tmp_path / "base", target_is_directory=True)
    assert read_native_adapter(adapter, str(tmp_path / "alias"))["model"] == str(tmp_path / "base")


@pytest.mark.parametrize("command", ["chat", "infer", "merge"])
@pytest.mark.parametrize("colour", [False, True])
def test_three_commands_print_the_same_legacy_refusal(tmp_path, monkeypatch, command, colour):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "1000")
    if colour:
        monkeypatch.setenv("FORCE_COLOR", "1")
    adapter = _adapter(tmp_path / "adapter", "./base")
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"hello"}\n')
    args = [command, "--backend", "mlx"]
    if command == "merge":
        args += ["--adapter", str(adapter), "--output", "merged"]
    else:
        args += ["--model", str(adapter)]
    if command == "infer":
        args += ["--input", "prompts.jsonl", "--output", "results.jsonl"]
    result = make_cli_runner().invoke(app, ["--no-telemetry", "--no-audit-log", *args])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    expected = (
        "Native MLX legacy relative base './base' is not a directory in cwd; "
        "pass --base with the absolute path to the original local base directory"
    )
    assert expected in _plain(result.output)
    assert not (tmp_path / "merged").exists()
    assert not (tmp_path / "results.jsonl").exists()


def test_legacy_resolution_preserves_markup(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "1000")
    (tmp_path / "[bold]base[/bold]").mkdir(parents=True)
    adapter = _adapter(tmp_path / "adapter", "./[bold]base[/bold]")
    config = read_native_adapter(adapter)
    output = _plain(capsys.readouterr().out)
    assert "'./[bold]base[/bold]'" in output
    assert config["model"] == str(tmp_path / "[bold]base[/bold]")


@pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto"])
def test_new_absolute_metadata_can_warm_start_the_same_relative_recipe(
    tmp_path, monkeypatch, task,
):
    from tests.test_issue634_mlx_resume import _write_fake_safetensors

    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    cfg = SoupConfig.model_validate({
        "base": "./base", "backend": "mlx", "task": task,
        "data": {"train": "rows.jsonl"}, "training": {"quantization": "none"},
    })
    module = importlib.import_module(f"soup_cli.trainer.mlx_{task}")
    name = "SimPO" if task == "simpo" else task.upper()
    wrapper = getattr(module, f"MLX{name}TrainerWrapper")(cfg)
    wrapper.model = _FakeMlxModel()
    adapter = _adapter(tmp_path / "adapter", str(tmp_path / "base"))
    metadata = json.loads((adapter / "adapter_config.json").read_text())
    metadata["task"] = task
    (adapter / "adapter_config.json").write_text(json.dumps(metadata))
    weights = adapter / "adapters.safetensors"
    _write_fake_safetensors(weights, ["lora_a"])
    wrapper._load_checkpoint_weights(str(weights))
    assert wrapper.model.load_weights_calls == [(str(weights), False, False)]


@pytest.mark.parametrize("stored", ["./base", "../base"])
def test_invalid_adapter_does_not_announce_a_successful_resolution(
    tmp_path, monkeypatch, capsys, stored,
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "base").mkdir()
    (tmp_path / "base").mkdir()
    monkeypatch.chdir(cwd)
    adapter = _adapter(tmp_path / "adapter", stored)
    (adapter / "adapters.safetensors").unlink()
    with pytest.raises(ValueError, match="missing adapters.safetensors"):
        read_native_adapter(adapter)
    assert capsys.readouterr().out == ""


@pytest.mark.requires_symlink
@pytest.mark.parametrize("selection", [
    "relative", "bare", "parent", "override", "override-parent", "relative-override",
])
@pytest.mark.parametrize("colour", [False, True])
def test_serve_refuses_a_legacy_base_selected_through_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str, colour: bool,
) -> None:
    import importlib.util

    from rich.console import Console

    from soup_cli.cli import app
    from soup_cli.commands import serve

    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "parent" / "base").mkdir(parents=True)
    (tmp_path / "alias").symlink_to(tmp_path / "base", target_is_directory=True)
    (tmp_path / "linked-parent").symlink_to(tmp_path / "parent", target_is_directory=True)
    stored = "./alias"
    flags = []
    if selection == "bare":
        stored = "alias"
    elif selection == "parent":
        stored = "./linked-parent/base"
    elif selection in {"override", "override-parent"}:
        stored = "./missing"
        target = "alias" if selection == "override" else "linked-parent/base"
        flags = ["--base", str(tmp_path / target)]
    elif selection == "relative-override":
        flags = ["--base", "./base"]
    adapter = _adapter(tmp_path / "adapter", stored)
    original = (adapter / "adapter_config.json").read_bytes()
    monkeypatch.setattr(serve, "console", Console(
        force_terminal=colour, no_color=False, color_system="standard", width=1000,
    ))
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name == "mlx_lm" else find_spec(name),
    )
    result = make_cli_runner().invoke(app, [
        "--no-telemetry", "--no-audit-log", "serve", "--backend", "mlx",
        "--model", str(adapter), *flags,
    ], color=colour)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    if colour:
        assert "\x1b[" in result.output
    assert (
        "MLX serving model/adapter paths must stay under cwd without symlinks"
        in _plain(result.output)
    )
    assert "resolved to" not in _plain(result.output)
    assert (adapter / "adapter_config.json").read_bytes() == original


@pytest.mark.parametrize("selection", ["relative", "bare", "absolute", "override"])
@pytest.mark.parametrize("colour", [False, True])
def test_serve_accepts_an_ordinary_local_base_before_dependency_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str, colour: bool,
) -> None:
    import importlib.util

    from rich.console import Console

    from soup_cli.cli import app
    from soup_cli.commands import serve

    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    stored = {"relative": "./base", "bare": "base", "absolute": str(tmp_path / "base"),
              "override": "./missing"}[selection]
    adapter = _adapter(tmp_path / "adapter", stored)
    original = (adapter / "adapter_config.json").read_bytes()
    flags = ["--base", str(tmp_path / "base")] if selection == "override" else []
    monkeypatch.setattr(serve, "console", Console(
        force_terminal=colour, no_color=False, color_system="standard", width=1000,
    ))
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name == "mlx_lm" else find_spec(name),
    )
    result = make_cli_runner().invoke(app, [
        "--no-telemetry", "--no-audit-log", "serve", "--backend", "mlx",
        "--model", str(adapter), *flags,
    ], color=colour)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    if colour:
        assert "\x1b[" in result.output
    assert "Native MLX serving requires MLX-LM on Apple Silicon." in _plain(result.output)
    assert "without symlinks" not in _plain(result.output)
    assert (adapter / "adapter_config.json").read_bytes() == original
