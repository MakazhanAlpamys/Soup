"""Issue #806 (remainder): ``reasoning_effort`` / ``train_on_eot`` only where they are read.

Both are applied by ``data.sft_format.build_format_row``. Its trainer callers
are sft's text path (inherited by tts through ``TTSTrainerWrapper.setup``) and
distill; pretrain and the classifier family never call it, so the flags loaded
there and changed nothing, while tts, which does honour them, was refused.
``train_on_eot`` also keys the pre-tokenized cache that pretrain loads (#1054),
so pretrain keeps that one.

Every refused config is first loaded without the flag, so a refusal can only
come from the flag itself.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import REASONING_EFFORT_AWARE_TASKS, TRAIN_ON_EOT_AWARE_TASKS
from tests.test_issue1325_expand_layers_scope import _ALL_TASKS, _DATA, _ROOT, _TRAINING

_REASONING = "  reasoning_effort: medium\n"
_EOT = "  train_on_eot: true\n"


def _yaml(task, flag=""):
    return (
        "base: meta-llama/Llama-3.1-8B\n"
        f"task: {task}\n"
        "backend: transformers\n"
        f"{_ROOT.get(task, '')}"
        "data:\n  train: data.jsonl\n"
        + _DATA.get(task, "")
        + "training:\n  quantization: none\n"
        + _TRAINING.get(task, "")
        + flag
    )


def test_the_allowlists_match_the_consumers():
    assert REASONING_EFFORT_AWARE_TASKS == {"sft", "tts", "distill"}
    assert TRAIN_ON_EOT_AWARE_TASKS == {"sft", "tts", "distill", "pretrain"}


@pytest.mark.parametrize("task", sorted(REASONING_EFFORT_AWARE_TASKS))
def test_reasoning_effort_loads_where_the_formatter_runs(task):
    cfg = load_config_from_string(_yaml(task, _REASONING))
    assert cfg.training.reasoning_effort == "medium"


@pytest.mark.parametrize("task", sorted(TRAIN_ON_EOT_AWARE_TASKS))
def test_train_on_eot_loads_where_it_is_read(task):
    cfg = load_config_from_string(_yaml(task, _EOT))
    assert cfg.training.train_on_eot is True


@pytest.mark.parametrize(
    "task", sorted(set(_ALL_TASKS) - REASONING_EFFORT_AWARE_TASKS)
)
def test_reasoning_effort_refused_everywhere_else(task):
    load_config_from_string(_yaml(task))
    match = rf"reasoning_effort='medium' requires task in .*task='{task}'"
    with pytest.raises(ValueError, match=match):
        load_config_from_string(_yaml(task, _REASONING))


@pytest.mark.parametrize("task", sorted(set(_ALL_TASKS) - TRAIN_ON_EOT_AWARE_TASKS))
def test_train_on_eot_refused_everywhere_else(task):
    load_config_from_string(_yaml(task))
    match = rf"train_on_eot=true requires task in .*task='{task}'"
    with pytest.raises(ValueError, match=match):
        load_config_from_string(_yaml(task, _EOT))


@pytest.mark.parametrize("task", ["pretrain", "classifier", "reranker", "cross_encoder"])
def test_reasoning_effort_no_longer_loads_on_the_tasks_that_ignored_it(task):
    """The four tasks the old allowlist admitted although none reads it."""
    with pytest.raises(ValueError, match="reasoning_effort"):
        load_config_from_string(_yaml(task, _REASONING))
