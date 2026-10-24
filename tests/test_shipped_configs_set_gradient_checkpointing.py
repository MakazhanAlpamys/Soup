"""Shipped configs for the TRL tasks set ``training.gradient_checkpointing`` explicitly (#1862).

Since #1778 the grpo, ipo, bco, reward_model and online_dpo trainers pass the key to TRL, so a
config that leaves it unset follows the schema default ``false`` instead of TRL's own ``true``
and silently uses more activation memory. Every recipe, template (YAML and inline copy) and
example config for those tasks has to state the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from soup_cli.config.schema import TEMPLATES
from soup_cli.recipes.catalog import RECIPES
from soup_cli.templates import list_templates, load_template

TRL_TASKS = {"grpo", "ipo", "bco", "reward_model", "online_dpo"}
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "configs"


def _shipped_configs() -> list[tuple[str, str]]:
    """(id, yaml text) for every shipped config whose task is one of TRL_TASKS."""
    sources = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    sources += [(f"template:{name}", load_template(name) or "") for name in list_templates()]
    sources += [(f"inline:{name}", text) for name, text in TEMPLATES.items()]
    sources += [(f"example:{path.name}", path.read_text(encoding="utf-8"))
                for path in sorted(EXAMPLES.glob("*.yaml"))]
    picked = []
    for source_id, text in sources:
        doc = yaml.safe_load(text) or {}
        if isinstance(doc, dict) and doc.get("task") in TRL_TASKS:
            picked.append((source_id, text))
    return picked


SHIPPED = _shipped_configs()


def test_every_trl_task_has_a_shipped_config():
    """Guard against a vacuous pass: the walk must reach configs for all five tasks.

    The literal pins TRL_TASKS itself, so dropping a task from the constant fails here too.
    """
    assert TRL_TASKS == {"grpo", "ipo", "bco", "reward_model", "online_dpo"}
    tasks = {yaml.safe_load(text)["task"] for _, text in SHIPPED}
    assert tasks == TRL_TASKS


@pytest.mark.parametrize(("source_id", "text"), SHIPPED, ids=[s for s, _ in SHIPPED])
def test_shipped_config_sets_gradient_checkpointing(source_id, text):
    training = yaml.safe_load(text).get("training") or {}
    assert "gradient_checkpointing" in training, (
        f"{source_id} leaves training.gradient_checkpointing unset, so it now follows the schema "
        "default false instead of TRL's true (#1862)"
    )


@pytest.mark.parametrize(("source_id", "text"), SHIPPED, ids=[s for s, _ in SHIPPED])
def test_shipped_config_still_loads(source_id, text):
    from tests.recipe_contracts import load_shipped_config

    load_shipped_config(source_id, text)
