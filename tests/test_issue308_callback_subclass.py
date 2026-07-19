"""Issue #308 — duck-typed TrainerCallbacks crash on HF event dispatch.

HF ``transformers.trainer_callback.CallbackHandler.call_event`` dispatches
**every** Trainer lifecycle event via ``getattr(callback, event)(...)`` with no
``hasattr`` guard. A callback added via ``trainer.add_callback(obj)`` that does
NOT subclass ``transformers.TrainerCallback`` therefore raises ``AttributeError``
on the first event it does not implement (e.g. ``on_epoch_begin``, fired right
after ``on_train_begin`` — before the first optimizer step).

``ReLoRACallback`` (``utils/relora.py``) and ``HFPushCallback``
(``monitoring/hf_push.py``) were plain duck-typed classes. This suite pins the
fix: both must be real ``TrainerCallback`` subclasses (via the lazy
``_try_import_callback_base`` factory that keeps the modules transformers-free at
import time) so they inherit the no-op defaults for the ~13 unimplemented events.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# isinstance + inherited-no-op-event checks (light, no torch)
# ---------------------------------------------------------------------------
class TestReLoRACallbackSubclass:
    def test_is_real_trainer_callback_subclass(self):
        from transformers import TrainerCallback

        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        cb = ReLoRACallback(ReLoRAPolicy(steps=10))
        assert isinstance(cb, TrainerCallback)

    def test_inherits_noop_unimplemented_event(self):
        # on_epoch_begin is NOT overridden — it must exist as an inherited
        # no-op stub, or HF's getattr dispatch crashes.
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        cb = ReLoRACallback(ReLoRAPolicy(steps=10))
        assert callable(cb.on_epoch_begin)
        # calling it must not raise (inherited no-op)
        cb.on_epoch_begin(None, None, None)

    def test_none_policy_still_subclass(self):
        from transformers import TrainerCallback

        from soup_cli.utils.relora import ReLoRACallback

        cb = ReLoRACallback(None)
        assert isinstance(cb, TrainerCallback)

    def test_no_top_level_transformers_import(self):
        import soup_cli.utils.relora as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for nm in names:
                    assert nm.split(".")[0] not in {
                        "torch",
                        "transformers",
                        "peft",
                    }, f"top-level import of {nm} in relora.py"


class TestHFPushCallbackSubclass:
    def test_is_real_trainer_callback_subclass(self):
        from transformers import TrainerCallback

        from soup_cli.monitoring.hf_push import HFPushCallback

        cb = HFPushCallback(repo_id="user/repo")
        assert isinstance(cb, TrainerCallback)

    def test_inherits_noop_unimplemented_event(self):
        from soup_cli.monitoring.hf_push import HFPushCallback

        cb = HFPushCallback(repo_id="user/repo")
        assert callable(cb.on_epoch_begin)
        cb.on_epoch_begin(None, None, None)

    def test_no_top_level_transformers_import(self):
        import soup_cli.monitoring.hf_push as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for nm in names:
                    assert nm.split(".")[0] not in {
                        "torch",
                        "transformers",
                        "peft",
                    }, f"top-level import of {nm} in hf_push.py"


# ---------------------------------------------------------------------------
# Real tiny Trainer.train() — reproduces the AttributeError on on_epoch_begin
# ---------------------------------------------------------------------------
def _tiny_causal_lm():
    """A from-config 2-layer Llama — no download, runs on CPU in ms."""
    import torch  # noqa: F401
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


class _TinyLMDataset:
    """Minimal map-style dataset yielding input_ids + labels for causal LM."""

    def __init__(self, n=8, seq=8, vocab=64):
        import torch

        self._rows = [
            {
                "input_ids": torch.randint(0, vocab, (seq,)),
                "labels": torch.randint(0, vocab, (seq,)),
                "attention_mask": torch.ones(seq, dtype=torch.long),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _run_two_steps(callback, tmp_path):
    """Build a real Trainer with ``callback`` and run 2 steps."""
    import torch  # noqa: F401
    from transformers import Trainer, TrainingArguments

    model = _tiny_causal_lm()
    ds = _TinyLMDataset()
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=1,
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.add_callback(callback)
    trainer.train()
    return trainer


class TestRealTrainerDispatch:
    def test_relora_callback_survives_train(self, tmp_path):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        cb = ReLoRACallback(ReLoRAPolicy(steps=1, warmup_ratio=0.0))
        # Must NOT raise AttributeError on on_epoch_begin during train().
        _run_two_steps(cb, tmp_path)

    def test_hfpush_callback_survives_train(self, tmp_path):
        from soup_cli.monitoring.hf_push import HFPushCallback

        cb = HFPushCallback(repo_id="user/repo", output_dir=str(tmp_path / "out"))
        # Mock the Hub API so on_train_begin/on_save never touch the network.
        with patch("soup_cli.monitoring.hf_push.get_hf_api", return_value=MagicMock()):
            _run_two_steps(cb, tmp_path)
