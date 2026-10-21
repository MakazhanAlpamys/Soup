"""#1814 — ``training.warmup_auto`` is deprecated: it warns at load, and v0.78 refuses it.

No trainer reads it. Its only reader was ``autopilot/generate_config.py``,
which nothing in ``src/`` calls, and which copied the autopilot decision INTO a
config rather than reading the field. ``warmup_ratio`` already resolves over the
actual optimizer steps (#1507), which is what ``warmup_auto`` promised. So an
explicit ``true`` loads with a warning naming the release that refuses it, one
minor after the warning first ships, and the generator stops writing the key.
"""

from __future__ import annotations

import io
import re
import warnings

import pytest
from rich.console import Console

import soup_cli
import soup_cli.config.loader as loader
from soup_cli.config import deprecation
from soup_cli.config.deprecation import (
    DEPRECATED_VALUE_REJECTION_VERSION,
    LATER_REJECTION_VERSIONS,
    SoupConfigDeprecationWarning,
)
from soup_cli.config.schema import TrainingConfig

_VALID = "base: org/m\ntask: sft\ndata:\n  train: ./t.jsonl\n"


def _major_minor(version: str) -> tuple[int, int]:
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()
    return int(major), int(minor)


@pytest.fixture
def printed(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


def test_an_explicit_true_warns_with_the_v0_78_deadline():
    with pytest.warns(SoupConfigDeprecationWarning) as record:
        TrainingConfig(warmup_auto=True)
    message = str(record[0].message)
    assert message.startswith("training.warmup_auto: true has no effect")
    assert "warmup_ratio" in message
    assert message.endswith("Soup v0.78 will refuse it.")


@pytest.mark.parametrize("value", [False, None])
def test_the_default_or_false_is_silent(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error", SoupConfigDeprecationWarning)
        kwargs = {} if value is None else {"warmup_auto": value}
        TrainingConfig(**kwargs)


def test_the_loader_prints_it_once(printed):
    cfg = loader.load_config_from_string(_VALID + "training:\n  warmup_auto: true\n")
    out = printed.getvalue()
    assert out.count("training.warmup_auto: true has no effect") == 1
    assert "Soup v0.78 will refuse it." in out
    assert cfg.training.warmup_auto is True  # still loads: this is the warning release


def test_the_later_deadline_is_one_minor_after_the_shared_one_and_not_reached():
    assert LATER_REJECTION_VERSIONS == {"training.warmup_auto": "0.78"}
    shared = _major_minor(DEPRECATED_VALUE_REJECTION_VERSION)
    for field, version in LATER_REJECTION_VERSIONS.items():
        assert _major_minor(version) == (shared[0], shared[1] + 1), field
        assert _major_minor(soup_cli.__version__) < _major_minor(version), (
            f"soup_cli is {soup_cli.__version__}: refuse {field} now"
        )


def test_the_shared_clause_is_unchanged():
    assert deprecation.deadline_clause() == (
        f"Soup v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it."
    )
    assert deprecation.deadline_clause("0.78") == "Soup v0.78 will refuse it."


def test_the_autopilot_generator_no_longer_writes_it(tmp_path, monkeypatch):
    import yaml

    from soup_cli.autopilot.generate_config import generate_config

    monkeypatch.chdir(tmp_path)
    decisions = {
        "task": "sft", "format": "alpaca", "max_length": 2048, "quantization": "4bit",
        "lora": {"r": 16, "alpha": 32, "use_dora": False}, "lr": 2e-4, "epochs": 3,
        "batch_size": 4, "grad_accum": 2,
        "perf": {"use_flash_attn": False, "use_liger": False, "gradient_checkpointing": False},
        "warmup_auto": True, "mixed_precision": "bf16",
    }
    out = generate_config(base="org/m", data_path="data.jsonl", decisions=decisions,
                          output_path="soup.yaml")
    config = yaml.safe_load(out.read_text(encoding="utf-8"))
    # write_yaml dumps the whole model, defaults included, so the key may be
    # present as false; what matters is that a decision no longer turns it on.
    assert config["training"].get("warmup_auto") in (None, False)
