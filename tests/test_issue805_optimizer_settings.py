"""Regression guards for issue #805's trainer configuration passthroughs.

These checks intentionally use the Python AST so they run in the lightweight
test environment where the optional training stack is not installed.  They pin
the wiring at each trainer's construction boundary, where a config value can
otherwise be accepted and silently discarded.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "soup_cli" / "trainer"


def _tree(name: str) -> ast.AST:
    return ast.parse((TRAINER / name).read_text(encoding="utf-8"))


def _call_keyword_names(tree: ast.AST, function: str) -> set[str]:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function
    ]
    assert calls, f"{function} construction disappeared"
    return {keyword.arg for keyword in calls[-1].keywords if keyword.arg is not None}


def test_prm_and_mole_forward_optimizer_schedule_and_batch_resolution() -> None:
    required = {
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "optim",
        "lr_scheduler_type",
        "gradient_checkpointing",
    }
    for name in ("prm.py", "mole_routing.py"):
        source = (TRAINER / name).read_text(encoding="utf-8")
        assert "estimate_batch_size(" in source
        assert "Auto batch size" in source
        assert "bs = 1" not in source
        assert required <= _call_keyword_names(_tree(name), "TrainingArguments")


def test_ppo_forwards_non_default_training_values_with_capability_probes() -> None:
    source = (TRAINER / "ppo.py").read_text(encoding="utf-8")
    assert '"warmup_steps"' in source
    assert '"weight_decay"' in source
    assert '"max_grad_norm"' in source
    assert '"optim"' in source
    assert '"lr_scheduler_type"' in source
    assert '"save_steps"' in source
    assert 'if name in ppo_params' in source


def test_unlearn_honors_optimizer_loop_and_model_loading_settings() -> None:
    source = (TRAINER / "unlearn.py").read_text(encoding="utf-8")
    tree = _tree("unlearn.py")
    adamw = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AdamW"
    ]
    assert adamw
    assert "weight_decay" in {keyword.arg for keyword in adamw[-1].keywords if keyword.arg}
    assert "get_scheduler(" in source
    assert "clip_grad_norm_(" in source
    assert "self.config.data.max_length" in source
    assert "quantization=tcfg.quantization" in source
    for field in ("target_modules", "use_dora", "use_rslora", "rank_pattern"):
        assert f"{field}=" in source
    assert "batch_size='auto' is unsupported" in source
