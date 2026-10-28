"""``soup profile --json`` and the MCP ``profile`` tool report gradient checkpointing twice.

``gradient_checkpointing`` is the config's own value, as it was before the estimate
resolved it (``true``, ``false``, ``"auto"`` or a tier name), so a script that tests it
as a boolean keeps working. ``gradient_checkpointing_tier`` is what the run is priced
with: ``off``, ``full``, ``medium`` or ``selective``.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.mcp_server.registry import tool_profile
from tests._tiny_checkpoint import write_tiny_model
from tests.conftest import strip_ansi

#: (what the config says, the task, the card to resolve `auto` on) -> the priced tier.
CASES = [
    (False, "sft", "off"),
    (True, "sft", "full"),
    ("full", "sft", "full"),
    ("medium", "sft", "medium"),
    ("selective", "sft", "selective"),
    # Only sft and tts apply the tiers; every other trainer checkpoints fully.
    (True, "dpo", "full"),
    ("medium", "dpo", "full"),
    ("selective", "dpo", "full"),
]


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


def _write(workdir, value, task="sft") -> None:
    document = {
        "base": str(workdir / "tiny"),
        "task": task,
        "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "gradient_checkpointing": value},
    }
    (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _cli(*extra: str) -> dict:
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json", *extra])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return json.loads(result.output)


@pytest.mark.parametrize(("value", "task", "tier"), CASES)
def test_the_cli_keeps_the_configs_value_and_adds_the_tier(workdir, value, task, tier):
    _write(workdir, value, task)
    profile = _cli()
    assert profile["gradient_checkpointing"] == value
    assert type(profile["gradient_checkpointing"]) is type(value)
    assert profile["gradient_checkpointing_tier"] == tier


@pytest.mark.parametrize(("value", "task", "tier"), CASES)
def test_the_mcp_tool_matches_the_cli(workdir, value, task, tier):
    _write(workdir, value, task)
    profile = tool_profile({"config": "soup.yaml"})
    assert profile["gradient_checkpointing"] == value
    assert type(profile["gradient_checkpointing"]) is type(value)
    assert profile["gradient_checkpointing_tier"] == tier
    assert profile["gradient_checkpointing"] == _cli()["gradient_checkpointing"]


def test_true_is_still_a_json_boolean(workdir):
    # The regression: a script reading `is True` got the string "full".
    _write(workdir, True)
    raw = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"]).output
    assert '"gradient_checkpointing": true' in raw
    assert '"gradient_checkpointing_tier": "full"' in raw


def test_false_is_still_a_json_boolean(workdir):
    _write(workdir, False)
    raw = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"]).output
    assert '"gradient_checkpointing": false' in raw
    assert '"gradient_checkpointing_tier": "off"' in raw


@pytest.mark.parametrize(("gpu", "tier"), [("rtx3060", "full"), ("h100", "medium")])
def test_auto_stays_auto_and_the_tier_is_the_one_the_card_resolves(workdir, gpu, tier):
    _write(workdir, "auto")
    profile = _cli("--gpu", gpu)
    assert profile["gradient_checkpointing"] == "auto"
    assert profile["gradient_checkpointing_tier"] == tier
    mcp = tool_profile({"config": "soup.yaml", "gpu": gpu})
    assert mcp["gradient_checkpointing"] == "auto"
    assert mcp["gradient_checkpointing_tier"] == tier


def test_the_rendered_profile_says_which_tier_is_priced(workdir):
    _write(workdir, "medium")
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--gpu", "a100"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Gradient checkpointing: enabled (medium)" in _plain(result.output)


def test_the_rendered_profile_prices_a_dpo_run_as_full(workdir):
    _write(workdir, "medium", task="dpo")
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--gpu", "a100"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Gradient checkpointing: enabled (full)" in _plain(result.output)


def test_the_direct_estimate_reports_the_same_two_fields():
    from soup_cli.utils.profiler import estimate_total

    off = estimate_total("m", 70.0, "none", 16, 32, 4, 2048, "adamw_torch", False)
    assert off["gradient_checkpointing"] is False
    assert off["gradient_checkpointing_tier"] == "off"
    on = estimate_total("m", 70.0, "none", 16, 32, 4, 2048, "adamw_torch", True)
    assert on["gradient_checkpointing"] is True
    assert on["gradient_checkpointing_tier"] == "full"
