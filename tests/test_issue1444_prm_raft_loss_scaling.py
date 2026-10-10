"""#1444 — PRM and RAFT must let Trainer apply 1/gradient_accumulation_steps.

Both custom ``compute_loss`` implementations return a per-micro-batch mean and
cannot honour ``num_items_in_batch``: Trainer derives it by counting
``labels[..., 1:] != -100``, which is meaningless for PRM's per-step float
rewards and ignores RAFT's ``loss_weights`` mass (citation-span boosts skew
it). transformers documents that a subclass in this position must set
``model_accepts_loss_kwargs = False`` so ``training_step`` divides by the
accumulation window. Without it, logged ``loss`` / ``train_loss`` /
``grad_norm`` come out exactly G times too large.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

_GA = 4


def _tiny_llama():
    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )


def _training_args(tmp_path):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(tmp_path),
        gradient_accumulation_steps=_GA,
        use_cpu=True,
        report_to="none",
    )


def _step_ratio(trainer, model, batch):
    """The issue's probe: what ``training_step`` returns over what
    ``compute_loss`` reports, for a one-micro-batch window of size _GA."""
    trainer.current_gradient_accumulation_steps = _GA
    n_items = trainer._get_num_items_in_batch([batch], torch.device("cpu"))
    with torch.no_grad():
        loss = float(trainer.compute_loss(model, batch))
    step = float(trainer.training_step(model, batch, n_items))
    return step / loss, step


class TestPrmTrainerClass:
    def test_training_step_divides_by_the_window(self, tmp_path):
        from transformers import Trainer

        from soup_cli.trainer.prm import _build_collator, make_prm_trainer_class

        model = _tiny_llama()
        model.reward_head = torch.nn.Linear(16, 1)
        batch = _build_collator(types.SimpleNamespace(pad_token_id=0))(
            [
                {
                    "input_ids": [1, 2, 3, 4],
                    "attention_mask": [1, 1, 1, 1],
                    "step_positions": [1, 3],
                    "labels": [1.0, 0.0],
                }
            ]
        )
        trainer = make_prm_trainer_class(Trainer)(model=model, args=_training_args(tmp_path))
        ratio, _ = _step_ratio(trainer, model, batch)
        assert ratio == pytest.approx(1.0 / _GA)


def _raft_batch(rows=None):
    from soup_cli.trainer.raft import _pad_raft_features

    rows = rows or [
        {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "labels": [-100, -100, 3, 4, 5, 6],
            "loss_weights": [0, 0, 1, 1, 1, 1],
        }
    ]
    return _pad_raft_features(rows, 0)


class TestRaftTrainerClass:
    def test_training_step_divides_by_the_window(self, tmp_path):
        from transformers import Trainer

        from soup_cli.trainer.raft import make_raft_trainer_class

        model = _tiny_llama()
        trainer = make_raft_trainer_class(Trainer)(model=model, args=_training_args(tmp_path))
        ratio, _ = _step_ratio(trainer, model, _raft_batch())
        assert ratio == pytest.approx(1.0 / _GA)

    def test_citation_faithful_weighted_mean_survives(self, tmp_path):
        """A boosted span still contributes its weight share; the fix changes
        the window scaling, not the per-micro-batch weighted mean."""
        from transformers import Trainer

        from soup_cli.trainer.raft import make_raft_trainer_class

        model = _tiny_llama()
        trainer = make_raft_trainer_class(Trainer)(model=model, args=_training_args(tmp_path))
        boosted = _raft_batch(
            [
                {
                    "input_ids": [1, 2, 3, 4, 5, 6],
                    "labels": [-100, -100, 3, 4, 5, 6],
                    "loss_weights": [0, 0, 2.0, 1.0, 1.0, 1.0],
                }
            ]
        )
        ratio, step = _step_ratio(trainer, model, boosted)
        assert ratio == pytest.approx(1.0 / _GA)
        with torch.no_grad():
            weighted = float(trainer.compute_loss(model, boosted))
        assert step == pytest.approx(weighted / _GA)


def _cfg(task: str, base: str, out: Path, *, fmt: str, over: dict):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("", encoding="utf-8")  # placeholder
    doc = {
        "base": base,
        "task": task,
        "backend": "transformers",
        "output": str(out),
        "data": {"train": "train.jsonl", "format": fmt, "max_length": 64},
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": _GA,
            "quantization": "none",
            "epochs": 1,
            "logging_steps": 1,
            "save_steps": 1000,
            **over,
        },
    }
    if task != "prm":
        # PRM fine-tunes every parameter and refuses a lora block. dropout
        # must be 0: the schema default 0.05 makes lora_B gradients
        # stochastic, which would break the cross-run comparisons below.
        doc["training"]["lora"] = {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        }
    return load_config_from_string(yaml.safe_dump(doc))


def _logged_first(wrapper):
    """First logged training record: (loss, grad_norm)."""
    history = wrapper.trainer.state.log_history
    entry = next(e for e in history if "loss" in e)
    return entry["loss"], entry["grad_norm"]


def _hide_mps(monkeypatch):
    """Keep the real wrapper on CPU: Soup does not wire use_cpu into
    TrainingArguments, and MPS backward reductions are nondeterministic
    enough to move logged grad_norm by ~1% between otherwise identical
    runs. Patching must happen after the lazy transformers import, so the
    call sites run _tiny_llama_dir first."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    # Transformers caches MPS availability across tests. Override the Trainer's
    # lookup too, so a prior MPS probe cannot move this CPU fixture's model.
    monkeypatch.setattr("transformers.training_args.is_torch_mps_available", lambda: False)


def _requires_train_extra():
    for mod in ("peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


class TestRealWrapperRuns:
    """4 identical rows at batch_size 1, gradient_accumulation_steps 4: one
    optimizer step. The logged loss must equal the micro-batch loss and the
    logged grad_norm the gradient norm of that loss — the same values a G=1
    run of the same data produces.
    """

    def _prm_run(self, tmp_path, monkeypatch, ga):
        _requires_train_extra()
        from soup_cli.trainer.prm import PRMTrainerWrapper
        from tests.test_issue1223_evaluate_val_split import _tiny_llama_dir

        tmp_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)
        base = _tiny_llama_dir(tmp_path / f"m{ga}")
        _hide_mps(monkeypatch)
        cfg = _cfg(
            "prm",
            base,
            tmp_path / f"o{ga}",
            fmt="prm",
            over={"gradient_accumulation_steps": ga},
        )
        wrapper = PRMTrainerWrapper(cfg, device="cpu")
        row = {"prompt": "hi", "completions": ["good answer"], "labels": [1.0]}
        wrapper.setup({"train": [dict(row) for _ in range(_GA)]})
        wrapper.train()
        return _logged_first(wrapper)

    def _raft_run(self, tmp_path, monkeypatch, ga):
        _requires_train_extra()
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from tests.test_issue1223_evaluate_val_split import _tiny_llama_dir

        tmp_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)
        base = _tiny_llama_dir(tmp_path / f"m{ga}")
        _hide_mps(monkeypatch)
        cfg = _cfg(
            "sft",
            base,
            tmp_path / f"o{ga}",
            fmt="raft",
            over={"gradient_accumulation_steps": ga},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        # No distractors: a single document has only one permutation, so all
        # four prepared rows are byte-identical whatever row_index the
        # shuffle gets (a distractor would make each row differ slightly and
        # the G=1 comparison below would mix per-row variance into the
        # scaling check).
        row = {
            "query": "hi",
            "golden_doc": "good answer",
            "distractor_docs": [],
            "answer": "good answer",
        }
        wrapper.setup({"train": [dict(row) for _ in range(_GA)]})
        wrapper.train()
        return _logged_first(wrapper)

    def test_prm_loss_and_grad_norm_match_a_g1_run(self, tmp_path, monkeypatch):
        g1_loss, g1_norm = self._prm_run(tmp_path / "a", monkeypatch, 1)
        g4_loss, g4_norm = self._prm_run(tmp_path / "b", monkeypatch, _GA)
        assert g4_loss == pytest.approx(g1_loss, rel=1e-4)
        assert g4_norm == pytest.approx(g1_norm, rel=1e-3)

    def test_raft_loss_and_grad_norm_match_a_g1_run(self, tmp_path, monkeypatch):
        g1_loss, g1_norm = self._raft_run(tmp_path / "a", monkeypatch, 1)
        g4_loss, g4_norm = self._raft_run(tmp_path / "b", monkeypatch, _GA)
        assert g4_loss == pytest.approx(g1_loss, rel=1e-4)
        assert g4_norm == pytest.approx(g1_norm, rel=1e-3)
