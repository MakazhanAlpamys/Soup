"""Tests for cloud training unsupported flag refusals and completeness (#1884)."""

import inspect
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.cloud import lambda_labs, modal, runpod
from soup_cli.commands import train as train_module
from tests.conftest import strip_ansi
from tests.test_cloud_shipped_inputs import _shipped_configs

_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: teknium/OpenHermes-2.5\n  format: chatml\n"
    "output: ./out\n"
)

# Primary refusals introduced by #1884, plus the newer diagnose-gate companion.
PRIMARY_REFUSED_FLAGS = [
    ("--resume", "auto"),
    ("--hf-resume", None),
    ("--wandb", None),
    ("--tensorboard", None),
    ("--tracker", "mlflow"),
    ("--deepspeed", "zero3"),
    ("--fsdp", "full_shard"),
    ("--gpus", "2"),
    ("--trust-remote-code", None),
    ("--profile", None),
    ("--allow-oom-attempt", None),
    ("--diagnose-gate", "ev.json"),
    ("--annex-xi", "a.md"),
    ("--repro-receipt", "r.json"),
    ("--capture-activations", "model.layers.0"),
    ("--track-energy", None),
    ("--energy-out", "e.json"),
    ("--diagnose-gate-require-all-modes", None),
]


def _setup_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(_SOUP_YAML, encoding="utf-8")
    return cfg


def _plain(output: str) -> str:
    return " ".join(strip_ansi(output).split())


@pytest.mark.parametrize(
    "flag, value", PRIMARY_REFUSED_FLAGS, ids=[f[0] for f in PRIMARY_REFUSED_FLAGS]
)
def test_modal_cloud_refuses_unsupported_flag_before_write_stub(
    tmp_path, monkeypatch, flag, value
):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    def unexpected_plan(*args, **kwargs):
        pytest.fail("an unsupported flag must be refused before planning")

    monkeypatch.setattr(modal, "plan_modal_run", unexpected_plan)
    cmd = ["train", "--config", "soup.yaml", "--cloud", "modal", flag]
    if value is not None:
        cmd.append(value)

    result = CliRunner().invoke(app, cmd)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not (tmp_path / "soup_modal_app.py").exists(), "stub must not be written on refusal"

    plain = _plain(result.output)
    assert flag in plain
    assert "not supported with --cloud" in plain
    assert "is not supported with --cloud" in plain
    assert (
        "--cloud runs 'soup train --config <yaml> --yes' remotely and forwards nothing else"
        in plain
    )


@pytest.mark.parametrize("backend", ["lambda", "runpod"])
@pytest.mark.parametrize("flag, value", [("--resume", "auto"), ("--trust-remote-code", None)])
def test_other_clouds_refuse_unsupported_flags(tmp_path, monkeypatch, backend, flag, value):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    cmd = ["train", "--config", "soup.yaml", "--cloud", backend, flag]
    if value is not None:
        cmd.append(value)

    result = CliRunner().invoke(app, cmd)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not (tmp_path / f"soup_{backend}_app.py").exists(), "stub must not be written on refusal"

    plain = _plain(result.output)
    assert flag in plain
    assert "is not supported with --cloud" in plain


def test_multiple_unsupported_flags_use_plural_verb(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    cmd = ["train", "--config", "soup.yaml", "--cloud", "modal", "--push-as", "me/soup", "--wandb"]
    result = CliRunner().invoke(app, cmd)
    assert result.exit_code == 2
    assert not (tmp_path / "soup_modal_app.py").exists()

    plain = _plain(result.output)
    assert "--wandb" in plain
    assert "--push-as" in plain
    assert "are not supported with --cloud" in plain
    assert "remotely and forwards nothing else" in plain


@pytest.mark.parametrize(
    "primary_and_companion",
    [
        ["--gpus", "2", "--no-reexec"],
        ["--capture-activations", "model.layers.0", "--capture-prompts", "prompts.jsonl"],
        ["--track-energy", "--energy-country", "US"],
    ],
)
def test_cloud_refuses_features_with_their_companions(tmp_path, monkeypatch, primary_and_companion):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)
    result = CliRunner().invoke(app, ["train", "--cloud", "modal", *primary_and_companion])

    assert result.exit_code == 2, result.output
    assert f"{primary_and_companion[0]} is not supported with --cloud" in _plain(result.output)
    assert not (tmp_path / "soup_modal_app.py").exists()


@pytest.mark.parametrize(
    "backend, module, override",
    [
        ("modal", modal, "_MODAL_SUBMIT_OVERRIDE"),
        ("lambda", lambda_labs, "_LAMBDA_SUBMIT_OVERRIDE"),
        ("runpod", runpod, "_RUNPOD_SUBMIT_OVERRIDE"),
    ],
)
@pytest.mark.parametrize("extra_flags", [[], ["--wandb"]], ids=["dry-run", "with-unsupported-flag"])
def test_cloud_submit_dry_run_refusal(
    tmp_path, monkeypatch, backend, module, override, extra_flags,
):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    submit_called = False

    def fake_submit(plan):
        nonlocal submit_called
        submit_called = True
        return 0

    monkeypatch.setattr(module, override, fake_submit)

    cmd = ["train", "--config", "soup.yaml", "--cloud", backend, "--cloud-submit", "--dry-run"]
    result = CliRunner().invoke(app, [*cmd, *extra_flags])
    assert result.exit_code == 2, result.output
    assert (
        not submit_called
    ), "submit seam must never be called when --dry-run is passed with --cloud-submit"
    assert not (tmp_path / f"soup_{backend}_app.py").exists()

    plain = _plain(result.output)
    assert "--dry-run cannot be combined with --cloud-submit:" in plain
    assert "drop --cloud-submit to preview the plan." in plain
    assert "drop the flag" not in plain


def test_plan_only_allows_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    cmd = ["train", "--config", "soup.yaml", "--cloud", "modal", "--dry-run"]
    result = CliRunner().invoke(app, cmd)
    assert result.exit_code == 0
    assert (tmp_path / "soup_modal_app.py").exists()
    plain = _plain(result.output)
    assert "modal run soup_modal_app.py" in plain


def test_find_lr_cloud_refusal_skips_sweep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    sweep_called = False

    def fake_sweep(*args, **kwargs):
        nonlocal sweep_called
        sweep_called = True
        return [], []

    monkeypatch.setattr(train_module, "_run_live_lr_sweep", fake_sweep)

    cmd = ["train", "--config", "soup.yaml", "--cloud", "modal", "--find-lr"]
    result = CliRunner().invoke(app, cmd)
    assert result.exit_code == 2
    assert not sweep_called, "live sweep must never run when --cloud is passed with --find-lr"
    assert not (tmp_path / "soup_modal_app.py").exists()

    plain = _plain(result.output)
    assert "--find-lr" in plain
    assert "is not supported with --cloud" in plain


def test_controls_forwarded_flags_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup_config(tmp_path)

    # 1. Normal modal plan
    res1 = CliRunner().invoke(
        app,
        ["train", "--config", "soup.yaml", "--cloud", "modal", "--gpu", "a100"],
    )
    assert res1.exit_code == 0
    assert (tmp_path / "soup_modal_app.py").exists()

    # 2. Modal submit with override
    submit_called = False

    def fake_submit(plan):
        nonlocal submit_called
        submit_called = True
        return 42

    monkeypatch.setattr(modal, "_MODAL_SUBMIT_OVERRIDE", fake_submit)
    res2 = CliRunner().invoke(
        app,
        ["train", "--config", "soup.yaml", "--cloud", "modal", "--cloud-submit"],
    )
    assert res2.exit_code == 42
    assert submit_called

    # 3. Multi-node check runs before cloud check and keeps existing message
    res3 = CliRunner().invoke(
        app,
        [
            "train", "--config", "soup.yaml", "--cloud", "modal",
            "--gpus", "8", "--nodes", "2", "--master-addr", "head.internal",
        ],
    )
    assert res3.exit_code == 1
    assert "--nodes > 1 cannot be combined with --cloud or --find-lr" in _plain(res3.output)


@pytest.mark.parametrize("configured", [False, True])
def test_name_reaches_cloud_config_and_config_name_wins(tmp_path, monkeypatch, configured):
    import yaml

    monkeypatch.chdir(tmp_path)
    path = _setup_config(tmp_path)
    if configured:
        path.write_text(_SOUP_YAML + "experiment_name: from-file\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["train", "--cloud", "modal", "--name", "from-cli"])

    assert result.exit_code == 0, result.output
    [shipped] = _shipped_configs(tmp_path / "soup_modal_app.py")
    assert yaml.safe_load(shipped)["experiment_name"] == ("from-file" if configured else "from-cli")


def test_train_parameter_completeness():
    from soup_cli.commands.train import FORWARDED_OR_EARLIER_CLOUD_FLAGS, REFUSED_CLOUD_FLAGS

    all_params = set(inspect.signature(train_module.train).parameters)

    overlap = REFUSED_CLOUD_FLAGS & FORWARDED_OR_EARLIER_CLOUD_FLAGS
    assert not overlap, f"Parameters present in both sets: {overlap}"

    union = REFUSED_CLOUD_FLAGS | FORWARDED_OR_EARLIER_CLOUD_FLAGS
    missing = all_params - union
    extra = union - all_params
    assert not missing, f"Parameters missing from completeness sets: {missing}"
    assert not extra, f"Parameters in sets but not in train() signature: {extra}"
