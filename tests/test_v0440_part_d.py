"""v0.44.0 Part D — Standalone CLI command tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.delinearize_llama4 import (
    is_llama4_model,
    plan_delinearize,
)
from soup_cli.utils.fetch_examples import (
    CATALOG,
    fetch_examples_dir,
    get_entry,
    list_entries,
)
from soup_cli.utils.fsdp_consolidate import (
    discover_shards,
    plan_consolidation,
)
from soup_cli.utils.llama_proxy import (
    build_argv,
    known_subcommands,
    resolve,
)
from soup_cli.utils.reasoning_parser import (
    known_parsers,
    parser_description,
    validate_parser_name,
)
from soup_cli.utils.sweep_config import (
    SweepSpec,
    load_sweep_yaml,
    parse_sweep_yaml,
)

runner = CliRunner()


# --- fetch_examples ---------------------------------------------------------

def test_catalog_non_empty():
    assert len(CATALOG) >= 2


def test_get_entry_known():
    entry = get_entry("llama-3.1-8b-lora")
    assert entry is not None
    assert entry.namespace == "examples"


def test_get_entry_unknown():
    assert get_entry("nope") is None
    assert get_entry("") is None
    assert get_entry("x\x00") is None
    assert get_entry(None) is None  # type: ignore[arg-type]


def test_list_entries_filter():
    examples = list_entries("examples")
    assert all(entry.namespace == "examples" for entry in examples.values())


def test_list_entries_invalid_namespace():
    with pytest.raises(ValueError):
        list_entries("bogus")


def test_fetch_examples_dir_exists():
    assert os.path.isdir(fetch_examples_dir())


def test_cli_fetch_lists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["fetch", "examples"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "llama-3.1-8b-lora" in result.output


def test_cli_fetch_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["fetch", "examples", "llama-3.1-8b-lora"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    written = tmp_path / "llama-3.1-8b-lora.yaml"
    assert written.is_file()


def test_cli_fetch_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path.parent / "out.yaml"
    result = runner.invoke(
        app,
        ["fetch", "examples", "llama-3.1-8b-lora", "-o", str(other)],
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "outside" in result.output.lower() or "under cwd" in result.output


def test_cli_fetch_unknown_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["fetch", "bogus"])
    assert result.exit_code == 2


def test_cli_fetch_unknown_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["fetch", "examples", "nope"])
    assert result.exit_code == 2


def test_cli_fetch_overwrite_protection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["fetch", "examples", "llama-3.1-8b-lora"])
    # Second invocation without --force fails.
    result = runner.invoke(app, ["fetch", "examples", "llama-3.1-8b-lora"])
    assert result.exit_code == 1


# --- quantize CLI -----------------------------------------------------------

def test_cli_quantize_prints_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["quantize", "./out", "--to", "gguf", "--bits", "4"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "soup export" in result.output


def test_cli_quantize_invalid_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["quantize", "./out", "--to", "bogus"])
    assert result.exit_code == 2


def test_cli_quantize_invalid_bits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["quantize", "./out", "--bits", "99"])
    assert result.exit_code == 2


# --- fsdp_consolidate -------------------------------------------------------

def test_discover_shards_picks_matching(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    (out / "pytorch_model_fsdp_0.bin").write_bytes(b"")
    (out / "pytorch_model_fsdp_1.bin").write_bytes(b"")
    (out / "unrelated.txt").write_text("x")
    found = discover_shards(str(out))
    assert found == ["pytorch_model_fsdp_0.bin", "pytorch_model_fsdp_1.bin"]


def test_discover_shards_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path.parent / "shards-elsewhere"
    with pytest.raises(ValueError, match="outside cwd"):
        discover_shards(str(other))


def test_discover_shards_missing_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        discover_shards(str(tmp_path / "missing"))


def test_plan_consolidation_happy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    (out / "pytorch_model_fsdp_0.bin").write_bytes(b"")
    target = tmp_path / "merged.safetensors"
    plan = plan_consolidation(str(out), str(target))
    assert plan.shard_files == ("pytorch_model_fsdp_0.bin",)


def test_plan_consolidation_rejects_non_safetensors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    (out / "pytorch_model_fsdp_0.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="safetensors"):
        plan_consolidation(str(out), str(tmp_path / "x.bin"))


def test_plan_consolidation_no_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    with pytest.raises(FileNotFoundError):
        plan_consolidation(str(out), str(tmp_path / "x.safetensors"))


def test_cli_merge_sharded_plan_only(tmp_path, monkeypatch):
    # v0.71.14 #96: --plan-only prints the plan without loading shards, so an
    # empty placeholder .bin is fine for the planning path.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    (out / "pytorch_model_fsdp_0.bin").write_bytes(b"")
    target = tmp_path / "merged.safetensors"
    result = runner.invoke(
        app,
        [
            "merge-sharded-fsdp-weights",
            str(out),
            "-o",
            str(target),
            "--plan-only",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Plan" in result.output
    assert not target.exists()


def test_cli_merge_sharded_invalid_shard_exits_2(tmp_path, monkeypatch):
    # v0.71.14 #96: live consolidation of a non-torch .bin fails gracefully.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    (out / "pytorch_model_fsdp_0.bin").write_bytes(b"not a torch checkpoint")
    target = tmp_path / "merged.safetensors"
    result = runner.invoke(
        app,
        ["merge-sharded-fsdp-weights", str(out), "-o", str(target)],
    )
    assert result.exit_code == 2


# --- delinearize_llama4 ------------------------------------------------------

def test_is_llama4_model_word_boundary():
    assert is_llama4_model("meta-llama/Llama-4-8B")
    assert is_llama4_model("LLAMA4")
    assert not is_llama4_model("llama-3.1-8b")
    assert not is_llama4_model("ungemma-llama-4ish")  # boundary check
    assert not is_llama4_model("")
    assert not is_llama4_model(123)  # type: ignore[arg-type]


def test_plan_delinearize_happy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "model"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"")
    target = tmp_path / "out"
    target.mkdir()
    plan = plan_delinearize(str(src), str(target))
    assert plan.weight_files == ("model.safetensors",)


def test_plan_delinearize_missing_safetensors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "model"
    src.mkdir()
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(FileNotFoundError):
        plan_delinearize(str(src), str(target))


def test_plan_delinearize_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "model"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"")
    other = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError, match="outside cwd"):
        plan_delinearize(str(src), str(other))


def test_cli_delinearize_llama4(tmp_path, monkeypatch):
    """v0.71.21 #97 — the runtime is live; --plan-only keeps the old flow."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "model"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"")
    target = tmp_path / "out"
    target.mkdir()
    result = runner.invoke(
        app,
        [
            "delinearize-llama4",
            str(src),
            "--target",
            str(target),
            "--plan-only",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Plan" in result.output


# --- sweep_config -----------------------------------------------------------

def test_parse_sweep_yaml_happy():
    spec = parse_sweep_yaml(
        "strategy: random\nn_runs: 5\nseed: 42\n"
        "params:\n  lr: [0.0001, 0.00005]\n  epochs: [1, 3]\n"
    )
    assert isinstance(spec, SweepSpec)
    assert spec.strategy == "random"
    assert spec.n_runs == 5
    assert spec.seed == 42
    assert spec.params["lr"] == (0.0001, 0.00005)
    assert spec.params["epochs"] == (1, 3)


def test_parse_sweep_yaml_strategy_validation():
    with pytest.raises(ValueError):
        parse_sweep_yaml("strategy: bogus\n")


def test_parse_sweep_yaml_n_runs_bounds():
    with pytest.raises(ValueError):
        parse_sweep_yaml("n_runs: -1\n")
    with pytest.raises(ValueError):
        parse_sweep_yaml("n_runs: 99999\n")


def test_parse_sweep_yaml_top_level_must_be_mapping():
    with pytest.raises(ValueError, match="mapping"):
        parse_sweep_yaml("- 1\n- 2\n")


def test_parse_sweep_yaml_rejects_oversize():
    with pytest.raises(ValueError, match="exceeds"):
        parse_sweep_yaml("# " + "x" * (256 * 1024 + 1))


def test_parse_sweep_yaml_rejects_null_byte():
    with pytest.raises(ValueError):
        parse_sweep_yaml("strategy: grid\n# evil\x00byte")


def test_parse_sweep_yaml_param_validation():
    with pytest.raises(ValueError):
        parse_sweep_yaml("params:\n  lr: 'not a list'\n")
    with pytest.raises(ValueError):
        parse_sweep_yaml("params:\n  lr: []\n")


def test_load_sweep_yaml_happy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "sweep.yaml"
    path.write_text("strategy: grid\nparams:\n  lr: [1e-4]\n")
    spec = load_sweep_yaml(str(path))
    assert spec.strategy == "grid"


def test_load_sweep_yaml_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path.parent / "sweep.yaml"
    with pytest.raises(ValueError, match="outside cwd"):
        load_sweep_yaml(str(other))


def test_load_sweep_yaml_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_sweep_yaml(str(tmp_path / "missing.yaml"))


def test_load_sweep_yaml_rejects_null_byte(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        load_sweep_yaml("bad\x00.yaml")


# --- llama_proxy ------------------------------------------------------------

def test_known_subcommands_immutable():
    subs = known_subcommands()
    with pytest.raises(TypeError):
        subs["x"] = "y"  # type: ignore[index]


def test_resolve_unknown_subcommand():
    with pytest.raises(ValueError, match="unknown llama subcommand"):
        resolve("bogus", [])


def test_resolve_too_many_args():
    with pytest.raises(ValueError, match="too many"):
        resolve("cli", ["x"] * 99)


def test_resolve_invalid_arg():
    # Args with newlines must be rejected even if the binary exists.
    # We simulate by patching shutil.which to return a fake path.
    import shutil

    real_which = shutil.which

    def fake_which(name, path=None):  # noqa: ARG001
        return "/fake/llama-cli"

    shutil.which = fake_which  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="control"):
            resolve("cli", ["bad\narg"])
        with pytest.raises(ValueError, match="exceeds"):
            resolve("cli", ["x" * 2048])
    finally:
        shutil.which = real_which  # type: ignore[assignment]


def test_resolve_missing_binary():
    import shutil

    real_which = shutil.which
    shutil.which = lambda *_a, **_k: None  # type: ignore[assignment]
    try:
        with pytest.raises(FileNotFoundError):
            resolve("cli", ["--help"])
    finally:
        shutil.which = real_which  # type: ignore[assignment]


def test_build_argv_includes_binary_then_args():
    import shutil

    real_which = shutil.which
    shutil.which = lambda *_a, **_k: "/fake/llama-cli"  # type: ignore[assignment]
    try:
        invocation = resolve("cli", ["--help"])
        argv = build_argv(invocation)
        assert argv[0].endswith("llama-cli")
        assert argv[1:] == ["--help"]
    finally:
        shutil.which = real_which  # type: ignore[assignment]


def test_cli_llama_help_lists_subcommands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["llama", "--help"])
    assert result.exit_code == 0
    # Each subcommand must appear in the help output.
    for sub in known_subcommands():
        assert sub in result.output


# --- reasoning_parser -------------------------------------------------------

def test_known_parsers_immutable():
    parsers = known_parsers()
    with pytest.raises(TypeError):
        parsers["x"] = "y"  # type: ignore[index]


def test_validate_parser_name_known():
    assert validate_parser_name("deepseek-r1") == "deepseek-r1"
    assert validate_parser_name("DEEPSEEK-R1") == "deepseek-r1"


def test_validate_parser_name_unknown():
    with pytest.raises(ValueError, match="unknown reasoning parser"):
        validate_parser_name("bogus")


def test_validate_parser_name_invalid():
    with pytest.raises(ValueError):
        validate_parser_name("")
    with pytest.raises(ValueError):
        validate_parser_name("x\x00")
    with pytest.raises(ValueError):
        validate_parser_name("x" * 100)
    with pytest.raises(TypeError):
        validate_parser_name(123)  # type: ignore[arg-type]


def test_parser_description():
    assert parser_description("deepseek-r1") is not None
    assert parser_description("nope") is None
    assert parser_description(123) is None  # type: ignore[arg-type]


# --- top-level CLI plumbing -------------------------------------------------

def test_cli_help_lists_new_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "monitor",
        "fetch",
        "quantize",
        "merge-sharded-fsdp-weights",
        "delinearize-llama4",
        "llama",
    ):
        assert command in result.output


def test_cli_monitor_help():
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
    assert "GPU" in result.output or "monitor" in result.output


def test_path_under_cwd_smoke(tmp_path):
    """Sanity for fixtures."""
    assert isinstance(Path(tmp_path), Path)
