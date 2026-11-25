"""classifier / reranker / cross_encoder: gradient accumulation must not scale the loss.

The wrapper trains with a plain ``transformers.Trainer``. A sequence-classification
model's ``forward`` takes ``**kwargs``, so Trainer sets ``model_accepts_loss_kwargs``
and skips its ``1 / gradient_accumulation_steps`` division, but the loss is a
per-micro-batch mean that never reads ``num_items_in_batch``. Left that way, the
logged loss and the gradient came out G times too large (same class as #1444).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
for _mod in ("peft", "datasets"):
    pytest.importorskip(_mod, reason=f"{_mod} is only in the [train] extra")

_GA = 4


def _run(tmp_path, monkeypatch, ga):
    from soup_cli.trainer.classifier import ClassifierTrainerWrapper
    from tests.test_issue1223_evaluate_val_split import _cfg, _tiny_llama_dir
    from tests.test_issue1444_prm_raft_loss_scaling import _hide_mps

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    base = _tiny_llama_dir(tmp_path)
    _hide_mps(monkeypatch)
    cfg = _cfg(
        "classifier",
        base,
        tmp_path / "out",
        gradient_accumulation_steps=ga,
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    )
    wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
    # _GA identical rows at batch_size 1: one optimizer step at G=_GA, whose
    # micro-batch losses all equal the G=1 run's first logged loss.
    wrapper.setup({"train": [{"text": "hello world", "label": 1} for _ in range(_GA)]})
    wrapper.train()
    entry = next(e for e in wrapper.trainer.state.log_history if "loss" in e)
    return entry["loss"], entry["grad_norm"]


def test_classifier_loss_and_grad_norm_match_a_g1_run(tmp_path, monkeypatch):
    g1_loss, g1_norm = _run(tmp_path / "a", monkeypatch, 1)
    g4_loss, g4_norm = _run(tmp_path / "b", monkeypatch, _GA)
    assert g4_loss == pytest.approx(g1_loss, rel=1e-3)
    assert g4_norm == pytest.approx(g1_norm, rel=1e-3)
