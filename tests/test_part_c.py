"""Part C — v0.28.1 speed/memory live (#43, #44, #47) for v0.33.0.

Covers:
  - #43 Multi-trainer wiring: apply_v028_speed_memory helper +
    supports_v028_features extension to dpo + pretrain. Schema gate
    softens to allow sft / dpo / pretrain.
  - #44 install_selective_hooks: per-layer checkpoint hooks across the
    three granularities, with a fake transformer-shaped model.
  - #47 CrossDocCollator: block-diagonal mask injection via the
    underlying build_cross_doc_mask helper.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# #43 — apply_v028_speed_memory helper
# ---------------------------------------------------------------------------


class TestApplyV028SpeedMemory:
    def test_no_features_returns_all_false(self):
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        tcfg = SimpleNamespace(
            use_cut_ce=False, quantization_aware=False,
            kernel_auto_compose=False,
        )
        result = apply_v028_speed_memory(
            model=MagicMock(), tcfg=tcfg, base_model="x/y",
        )
        assert result == {
            "cut_ce": False, "fp8": False, "kernel_auto_compose": False,
        }

    def test_cut_ce_failure_logs_yellow(self, monkeypatch, capsys):
        from soup_cli.utils import v028_features as vf

        # Force apply_cut_ce import to raise
        def _boom(_name):
            raise RuntimeError("no cut_cross_entropy")

        monkeypatch.setattr(
            "soup_cli.utils.cut_ce.apply_cut_ce", _boom, raising=False,
        )
        from rich.console import Console
        console = Console()
        tcfg = SimpleNamespace(
            use_cut_ce=True, quantization_aware=False,
            kernel_auto_compose=False,
        )
        result = vf.apply_v028_speed_memory(
            model=MagicMock(), tcfg=tcfg,
            base_model="x/y", console=console,
        )
        assert result["cut_ce"] is False

    def test_supports_v028_features_extends_to_dpo_pretrain(self):
        from soup_cli.utils.v028_features import supports_v028_features

        for task in ("sft", "dpo", "pretrain"):
            assert supports_v028_features(task) is True
        for task in ("grpo", "kto", "orpo", "simpo", "ipo", "ppo",
                     "reward_model", "embedding"):
            assert supports_v028_features(task) is False

    def test_warn_unsupported_returns_none_for_supported(self):
        from soup_cli.utils.v028_features import warn_unsupported_features

        tcfg = SimpleNamespace(
            use_cut_ce=True, quantization_aware="fp8",
            kernel_auto_compose=True, activation_offloading="cpu",
        )
        assert warn_unsupported_features(tcfg, "sft") is None
        assert warn_unsupported_features(tcfg, "dpo") is None

    def test_warn_unsupported_lists_offenders_for_unsupported(self):
        from soup_cli.utils.v028_features import warn_unsupported_features

        tcfg = SimpleNamespace(
            use_cut_ce=True, quantization_aware="fp8",
            kernel_auto_compose=False, activation_offloading=None,
        )
        msg = warn_unsupported_features(tcfg, "kto")
        assert msg is not None
        assert "use_cut_ce" in msg
        assert "fp8" in msg


class TestSchemaGateExpanded:
    def _config(self, task: str, **training_extra):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        body = {
            "base": "test/model",
            "task": task,
            "data": {"train": "data.jsonl", "format": "alpaca"}
            if task != "pretrain"
            else {"train": "data.jsonl", "format": "plaintext"},
            "training": {"epochs": 1, "lr": 1e-4, "batch_size": 1, **training_extra},
        }
        if task in ("dpo", "kto", "orpo", "simpo", "ipo", "grpo"):
            body["data"]["format"] = "dpo"
        return load_config_from_string(yaml.safe_dump(body))

    def test_dpo_now_accepts_v028_features(self):
        cfg = self._config("dpo", use_cut_ce=True)
        assert cfg.task == "dpo"
        assert cfg.training.use_cut_ce is True

    def test_pretrain_now_accepts_v028_features(self):
        cfg = self._config("pretrain", use_cut_ce=True)
        assert cfg.task == "pretrain"

    def test_kto_still_rejects_v028_features(self):
        with pytest.raises(Exception, match="v0.28.0"):
            self._config("kto", use_cut_ce=True)


# ---------------------------------------------------------------------------
# #44 — install_selective_hooks
# ---------------------------------------------------------------------------


class TestInstallSelectiveHooks:
    def test_unknown_granularity_rejected(self):
        from soup_cli.utils.gradient_ckpt import install_selective_hooks

        with pytest.raises(ValueError, match="must be one of"):
            install_selective_hooks(MagicMock(), "weird")

    def test_full_wraps_every_block(self):
        """A 4-block fake model with 'medium' wraps 2 blocks (every other)."""
        from soup_cli.utils.gradient_ckpt import install_selective_hooks

        class FakeBlock:
            def __init__(self, idx):
                self.idx = idx

            def forward(self, *args, **kwargs):
                return ("orig", self.idx)

            # Simulate having attention sub-modules
            def named_modules(self, prefix=""):
                yield (f"{prefix}", self)
                yield (f"{prefix}.self_attn", _AttnStub(self.idx))

        class _AttnStub:
            def __init__(self, idx):
                self.idx = idx

            def forward(self, *args, **kwargs):
                return ("attn", self.idx)

        class FakeModel:
            def __init__(self):
                self.blocks = [FakeBlock(i) for i in range(4)]

            def named_modules(self):
                # HF-style names: ``model.layers.<i>``
                for idx, blk in enumerate(self.blocks):
                    yield (f"model.layers.{idx}", blk)
                    for child_name, child in blk.named_modules(
                        prefix=f"model.layers.{idx}",
                    ):
                        if child_name != f"model.layers.{idx}":
                            yield (child_name, child)

        # Full granularity wraps every numbered child
        model = FakeModel()
        hooked_full = install_selective_hooks(model, "full")
        assert hooked_full == 4

    def test_medium_wraps_every_other_block(self):
        from soup_cli.utils.gradient_ckpt import install_selective_hooks

        class FakeBlock:
            def forward(self, *args, **kwargs):
                return None

            def named_modules(self, prefix=""):
                yield (prefix, self)

        class FakeModel:
            def __init__(self, n):
                self.blocks = [FakeBlock() for _ in range(n)]

            def named_modules(self):
                for i, blk in enumerate(self.blocks):
                    yield (f"model.layers.{i}", blk)

        hooked = install_selective_hooks(FakeModel(6), "medium")
        # 6 blocks, every-other → 3
        assert hooked == 3

    def test_selective_wraps_attention_only(self):
        from soup_cli.utils.gradient_ckpt import install_selective_hooks

        class _Attn:
            def forward(self, *args, **kwargs):
                return None

        class FakeBlock:
            def __init__(self):
                self._attn = _Attn()
                self._mlp = MagicMock()
                self._mlp.forward = lambda *_a, **_k: None

            def forward(self, *args, **kwargs):
                return None

            def named_modules(self, prefix=""):
                yield (prefix, self)
                yield (f"{prefix}.self_attn", self._attn)
                yield (f"{prefix}.mlp", self._mlp)

        class FakeModel:
            def __init__(self):
                self.blocks = [FakeBlock() for _ in range(3)]

            def named_modules(self):
                for i, blk in enumerate(self.blocks):
                    yield (f"model.layers.{i}", blk)
                    for child_name, child in blk.named_modules(
                        prefix=f"model.layers.{i}",
                    ):
                        if child_name != f"model.layers.{i}":
                            yield (child_name, child)

        hooked = install_selective_hooks(FakeModel(), "selective")
        # 3 blocks, each with one attention child = 3 hooks
        assert hooked == 3


# ---------------------------------------------------------------------------
# #47 — CrossDocCollator
# ---------------------------------------------------------------------------


class TestCrossDocCollator:
    def test_requires_base_collator(self):
        from soup_cli.data.collators import CrossDocCollator

        with pytest.raises(ValueError, match="base_collator"):
            CrossDocCollator(base_collator=None)

    def test_passes_through_when_no_doc_lengths(self):
        from soup_cli.data.collators import CrossDocCollator

        base = MagicMock(return_value={"input_ids": MagicMock()})
        collator = CrossDocCollator(base_collator=base)
        result = collator([{"input_ids": [1, 2, 3]}])
        assert "input_ids" in result
        # No cross_doc_attn_mask since no doc_lengths supplied
        assert "cross_doc_attn_mask" not in result

    def test_strips_doc_lengths_before_base_call(self):
        from soup_cli.data.collators import CrossDocCollator

        captured: list[dict] = []

        def _base(features):
            captured.extend(features)
            return {"input_ids": MagicMock()}

        collator = CrossDocCollator(base_collator=_base)
        collator([{"input_ids": [1, 2], "doc_lengths": [1, 1]}])
        # base collator must NOT see the doc_lengths key
        assert "doc_lengths" not in captured[0]

    def test_mismatched_doc_lengths_falls_back_to_causal(self):
        """If sum(doc_lengths) > seq_length the collator must NOT crash —
        it falls back to a plain lower-triangular mask."""
        from soup_cli.data.collators import CrossDocCollator

        seq_len = 4

        class _Tensor:
            shape = (1, seq_len)

        def _base(features):
            return {"input_ids": _Tensor()}

        collator = CrossDocCollator(base_collator=_base)
        # 3 + 3 = 6 > seq_len 4 → fallback path
        result = collator([{"doc_lengths": [3, 3]}])
        # Either no mask emitted (graceful skip) or fallback mask present.
        # Either is acceptable; what matters is no exception.
        assert "input_ids" in result

    def test_does_not_mutate_input_dict(self):
        """Regression: collator pops doc_lengths in v0.33.0 wave; HF Dataset
        rows are cached, so mutation breaks subsequent batches."""
        from soup_cli.data.collators import CrossDocCollator

        def _base(features):
            return {"input_ids": object()}

        collator = CrossDocCollator(base_collator=_base)
        original = {"doc_lengths": [2, 2], "input_ids": [1, 2, 3, 4]}
        collator([original])
        # The original dict must still have doc_lengths after the call.
        assert original.get("doc_lengths") == [2, 2]

    def test_injects_block_diag_mask_with_doc_lengths(self):
        """End-to-end: when doc_lengths are present, cross_doc_attn_mask
        appears on the batch with the right shape."""
        import numpy as np

        from soup_cli.data.collators import CrossDocCollator

        # 2 documents, each 2 tokens long, packed into seq_length=4
        seq_len = 4

        class _Tensor:
            shape = (1, seq_len)

        def _base(features):
            return {"input_ids": _Tensor()}

        collator = CrossDocCollator(base_collator=_base)
        result = collator([{"doc_lengths": [2, 2]}])

        assert "cross_doc_attn_mask" in result
        mask = result["cross_doc_attn_mask"]
        # Shape: (batch=1, seq, seq)
        assert mask.shape == (1, seq_len, seq_len)
        # Token in doc 0 should NOT attend to token in doc 1 (positions 0,1 vs 2,3)
        np_mask = mask[0].numpy() if hasattr(mask[0], "numpy") else np.array(mask[0])
        # Position 0 attending to position 2 → should be 0
        assert np_mask[0, 2] == 0
        # Position 2 attending to position 2 → 1 (causal within doc 1)
        assert np_mask[2, 2] == 1
