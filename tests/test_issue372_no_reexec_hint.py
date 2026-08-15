"""Issue #372 — ``soup train --no-reexec`` dropped the user's own flags.

The printed ``accelerate launch`` hint used to be hand-built as
``["soup", "train", "-c", config]`` a few lines above the auto-reexec path,
which assembled the real flag list. Two copies of "what the user typed", and
only one of them was maintained — so ``--gpus 4 --fsdp full_shard --no-reexec``
suggested a command WITHOUT ``--fsdp``. Following it literally trained without
FSDP, and the run succeeded, so there was no error to trace back.

The hint is now derived from the same argv the run would have re-exec'd with
(``build_train_reexec_argv`` + ``hint_argv_from_reexec``). These tests compare
against that builder's output, not a hardcoded expected string, so a future
flag added to the builder is covered without editing this file.
"""

from __future__ import annotations

import re
import shlex

import pytest

from soup_cli.utils.launcher import (
    build_accelerate_argv,
    build_train_reexec_argv,
    collect_reexec_passthrough,
    format_advice,
    hint_argv_from_reexec,
)


def _tokens_from_advice(advice: str) -> list[str]:
    """The indented ``accelerate ...`` line inside ``format_advice``."""
    for line in advice.splitlines():
        stripped = line.strip()
        if stripped.startswith("accelerate"):
            return shlex.split(stripped)
    raise AssertionError(f"no accelerate command in advice:\n{advice}")


def _printed_tokens(config: str, extra_flags: list[str], num_gpus: int = 4) -> list[str]:
    """Hint tokens produced the same way train.py prints them."""
    script_args = build_train_reexec_argv(config, extra_flags)
    hint_args = hint_argv_from_reexec(script_args)
    return _tokens_from_advice(format_advice(num_gpus, hint_args))


# ---------------------------------------------------------------------------
# Builder is the single source
# ---------------------------------------------------------------------------


class TestReexecArgvBuilder:
    def test_reexec_argv_always_includes_no_reexec_and_module_form(self):
        argv = build_train_reexec_argv("soup.yaml", ["--fsdp", "full_shard"])
        assert argv[1:4] == ["-m", "soup_cli.cli", "train"]
        assert "--config" in argv and "soup.yaml" in argv
        assert "--no-reexec" in argv
        assert argv[argv.index("--fsdp") + 1] == "full_shard"

    def test_hint_drops_no_reexec_and_uses_soup_cli_name(self):
        argv = build_train_reexec_argv("soup.yaml", ["--fsdp", "full_shard"])
        hint = hint_argv_from_reexec(argv)
        assert hint[:2] == ["soup", "train"]
        assert "--no-reexec" not in hint
        assert "--fsdp" in hint and "full_shard" in hint

    def test_hint_rejects_a_malformed_reexec_argv(self):
        with pytest.raises(ValueError, match="re-exec argv"):
            hint_argv_from_reexec(["soup", "train", "--config", "soup.yaml"])

    def test_empty_config_rejected(self):
        with pytest.raises(ValueError, match="config"):
            build_train_reexec_argv("")


# ---------------------------------------------------------------------------
# Acceptance: tokenised hint is a superset of the user-supplied flags,
# compared against the re-exec builder — not a hardcoded expected string.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_flags",
    [
        ["--fsdp", "full_shard"],
        ["--deepspeed", "zero2"],
        ["--config", "custom.yaml"],
        # train has no ``--stream-*`` CLI flag; this is the hyphenated
        # run-shaping stand-in the re-exec path already carries.
        ["--trust-remote-code"],
    ],
    ids=["fsdp", "deepspeed", "config", "hyphenated-flag"],
)
def test_printed_command_is_superset_of_user_flags(user_flags):
    config = "soup.yaml"
    extra = list(user_flags)
    if extra[:1] == ["--config"]:
        config = extra[1]
        extra = extra[2:]

    script_args = build_train_reexec_argv(config, extra)
    hint_args = hint_argv_from_reexec(script_args)
    tokens = _tokens_from_advice(format_advice(4, hint_args))

    for flag in user_flags:
        assert flag in tokens, (flag, tokens)

    # Same builder the auto-reexec wraps — a future extra flag is covered
    # without editing this assertion.
    assert tokens == build_accelerate_argv(4, hint_args)


def test_no_extra_flags_prints_a_valid_minimal_command():
    tokens = _printed_tokens("soup.yaml", [])
    assert tokens[:4] == ["accelerate", "launch", "--num_processes", "4"]
    assert tokens[4:] == ["soup", "train", "--config", "soup.yaml"]
    assert "--no-reexec" not in tokens
    assert "--fsdp" not in tokens


def test_a_flag_added_only_to_the_builder_shows_up_in_the_hint():
    """Control for 'do not hardcode the expected string'.

    If someone later extends collect_reexec_passthrough / extra_flags, the hint
    must carry whatever the builder was given — without this file listing it.
    """
    future = ["--some-future-flag", "value"]
    tokens = _printed_tokens("soup.yaml", future)
    for flag in future:
        assert flag in tokens


# train() options that must NOT appear as collect_reexec_passthrough kwargs.
# Adding a soup train flag fails this test until you either pass it through
# or write it down here as local-only — silence is no longer a decision.
EXPECTED_LOCAL_ONLY = frozenset({
    # Wired by build_train_reexec_argv, not as passthrough kwargs.
    "config",
    "no_reexec",
    # Becomes accelerate --num_processes; repeating --gpus would double-count.
    "gpus",
    # Plan-only: the multi-GPU path skips re-exec entirely.
    "dry_run",
    # Early-return before any launch.
    "find_lr",
    "find_lr_start",
    "find_lr_end",
    "find_lr_steps",
    "find_lr_output",
    # Different launch path (Modal). Incompatible with --gpus / accelerate.
    "cloud",
    "gpu",
    "cloud_submit",
})


def test_every_train_option_is_passed_through_or_deliberately_local():
    """CLI → builder drift: a new ``soup train`` flag cannot be silent.

    ``collect_reexec_passthrough`` is still a hand-written keyword list. The
    hole #372 came from is a flag accepted by the CLI and absent from the
    launch argv, with the run succeeding. This assertion forces a decision
    for every ``train()`` parameter: pass it through, or name it local-only.
    """
    import inspect

    from soup_cli.commands.train import train

    cli = set(inspect.signature(train).parameters)
    passed = set(inspect.signature(collect_reexec_passthrough).parameters)
    missing = cli - passed - EXPECTED_LOCAL_ONLY
    assert not missing, (
        f"new `soup train` flags not routed to re-exec: {sorted(missing)}"
    )
    extra = passed - cli
    assert not extra, (
        f"passthrough kwargs that are not `soup train` options: {sorted(extra)}"
    )
    # The exclusion set itself must not list a flag that IS passed through,
    # or a later deletion of the passthrough kwarg would be hidden.
    overlap = passed & EXPECTED_LOCAL_ONLY
    assert not overlap, (
        f"flags listed local-only but also passed through: {sorted(overlap)}"
    )


def test_name_and_replay_survive_the_passthrough():
    extra = collect_reexec_passthrough(
        name="exp-1",
        replay="old.jsonl",
        replay_ratio=0.2,
        replay_seed=7,
    )
    tokens = _printed_tokens("soup.yaml", extra)
    assert tokens[tokens.index("--name") + 1] == "exp-1"
    assert tokens[tokens.index("--replay") + 1] == "old.jsonl"
    assert tokens[tokens.index("--replay-ratio") + 1] == "0.2"
    assert tokens[tokens.index("--replay-seed") + 1] == "7"


# ---------------------------------------------------------------------------
# CLI: the advisory must actually be reached (no skip-if-validation-failed)
# ---------------------------------------------------------------------------


def _invoke_no_reexec(tmp_path, monkeypatch, extra_args):
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.utils import launcher as launcher_mod
    from soup_cli.utils import topology as topo_mod

    for var in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_USE_FSDP",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: test/model\n"
        "task: sft\n"
        "data: {train: data.jsonl, format: alpaca}\n"
        "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        topo_mod, "detect_topology",
        lambda: {"gpu_count": 4, "interconnect": "PCIe"},
    )
    monkeypatch.setattr(topo_mod, "resolve_num_gpus", lambda spec: 4)
    monkeypatch.setattr(launcher_mod, "is_in_distributed", lambda: False)

    result = CliRunner().invoke(
        app,
        ["train", "--config", "soup.yaml", "--gpus", "4", "--no-reexec", "--yes",
         *extra_args],
    )
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")
    return result, out


class TestNoReexecHintCli:
    def test_fsdp_appears_in_the_printed_command(self, tmp_path, monkeypatch):
        result, out = _invoke_no_reexec(
            tmp_path, monkeypatch, ["--fsdp", "full_shard"]
        )
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "Multi-GPU launch required" in out, out
        assert "--fsdp" in out, out
        assert "full_shard" in out, out
        assert "--no-reexec" not in out, out

    def test_control_without_extra_flags_still_prints_accelerate(self, tmp_path, monkeypatch):
        result, out = _invoke_no_reexec(tmp_path, monkeypatch, [])
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "Multi-GPU launch required" in out, out
        assert "accelerate" in out, out
        assert "--config" in out, out
        assert "soup.yaml" in out, out
