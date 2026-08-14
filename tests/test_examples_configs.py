"""Every shipped example config must parse against the current schema.

This file exists because the examples rotted silently. Seven of the eight
configs in ``examples/configs/`` were written against a pre-nesting schema
(``model:`` / ``data.path:`` / top-level ``lora_r:``) and stayed that way
through several schema changes, so ``soup train --config`` on any of them
failed at validation. Nothing caught it: ``tests/test_dpo_example.py`` pins
``dpo_example.yaml`` by name, which was the one file that happened to be
current.

The parametrization globs the directory rather than listing names, so a new
example is covered the moment it is added — a hand-maintained list would have
the same blind spot that caused this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.config.loader import load_config
from soup_cli.config.schema import SoupConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "examples" / "configs"

EXAMPLE_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


def test_example_configs_were_actually_found():
    """Guard the parametrization above.

    An empty glob makes every parametrized test below vanish and the file
    reports green — the exact failure mode this file was written to prevent.
    The bound is the eight configs that shipped; it only needs to catch a
    glob that silently found nothing.
    """
    assert len(EXAMPLE_CONFIGS) >= 8, (
        f"Expected at least 8 example configs under {CONFIGS_DIR}, "
        f"found {len(EXAMPLE_CONFIGS)}: {[p.name for p in EXAMPLE_CONFIGS]}"
    )


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_parses(config_path: Path):
    """Each example must load as a valid SoupConfig.

    This is what ``soup train --config <path>`` does before it touches a
    model, so a failure here is a user hitting a validation error on a file
    we shipped.
    """
    cfg = load_config(config_path)
    assert isinstance(cfg, SoupConfig)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_data_file_exists(config_path: Path):
    """A config pointing into ``examples/`` must point at a file that is there.

    ``vision_llama.yaml`` referenced ``examples/data/vision_dataset.jsonl``
    and ``examples/data/images/``, neither of which has ever existed in the
    repo — a config can be perfectly schema-valid and still be impossible to
    run. Templates opt out by using a placeholder path (``./your_data.jsonl``)
    rather than claiming a bundled fixture, so only ``examples/``-relative
    references are checked here.
    """
    cfg = load_config(config_path)
    train = cfg.data.train
    if not train.startswith("examples/"):
        pytest.skip(f"{config_path.name} is a template: data.train={train!r}")

    assert (REPO_ROOT / train).is_file(), (
        f"{config_path.name} references {train!r}, which does not exist"
    )

    if cfg.data.image_dir and cfg.data.image_dir.startswith("examples/"):
        assert (REPO_ROOT / cfg.data.image_dir).is_dir(), (
            f"{config_path.name} references image_dir={cfg.data.image_dir!r}, "
            "which does not exist"
        )
