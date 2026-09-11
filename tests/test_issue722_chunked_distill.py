"""Tests for token-chunked checkpointed KL distillation (Issue #722).

Validates:
1. Exact loss and gradient agreement between dense and chunked calculations.
2. Non-reentrant activation checkpointing parity and memory reduction.
3. Causal shift and response-only label masking integrity.
4. Edge cases (all-masked tokens, attention_mask only, unmasked inputs).
5. Argument validation (chunk_size, use_checkpoint).
6. Schema and config validation (field types, rejection outside task='distill').
"""

from __future__ import annotations

import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


class TestChunkedDistillKernel:
    @pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 32, 1000])
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
    def test_chunked_matches_dense_loss_and_gradients(
        self, divergence: str, chunk_size: int, dtype_name: str
    ) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        dtype = getattr(torch, dtype_name)
        torch.manual_seed(42)
        batch, seq, vocab = 2, 8, 16
        temp = 2.0

        s_dense = torch.randn(batch, seq, vocab, dtype=dtype, requires_grad=True)
        t_dense = torch.randn(batch, seq, vocab, dtype=dtype)
        labels = torch.tensor([
            [-100, 1, 2, -100, 4, -100, 6, -100],
            [-100, -100, 2, 3, -100, 5, 6, -100],
        ])

        # Dense baseline (chunk_size=None, use_checkpoint=False)
        loss_dense = _compute_distill_term(
            s_dense, t_dense, divergence, temp, labels=labels, chunk_size=None, use_checkpoint=False
        )
        loss_dense.backward()
        grad_dense = s_dense.grad.clone()

        # Chunked
        s_chunk = s_dense.detach().clone().requires_grad_(True)
        loss_chunk = _compute_distill_term(
            s_chunk,
            t_dense,
            divergence,
            temp,
            labels=labels,
            chunk_size=chunk_size,
            use_checkpoint=False,
        )
        loss_chunk.backward()
        grad_chunk = s_chunk.grad.clone()

        tol = 5e-3 if dtype_name == "bfloat16" else 1e-6
        assert abs(loss_dense.item() - loss_chunk.item()) < tol
        assert (grad_dense - grad_chunk).abs().max().item() < tol

    @pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
    def test_checkpointed_matches_dense_loss_and_gradients(
        self, divergence: str, dtype_name: str
    ) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        dtype = getattr(torch, dtype_name)
        torch.manual_seed(101)
        batch, seq, vocab = 2, 10, 16
        temp = 1.5

        s_dense = torch.randn(batch, seq, vocab, dtype=dtype, requires_grad=True)
        t_dense = torch.randn(batch, seq, vocab, dtype=dtype)
        labels = torch.tensor([
            [-100, 0, 1, 2, -100, -100, 3, 4, 5, -100],
            [-100, -100, 1, -100, 2, 3, 4, -100, 5, -100],
        ])

        loss_dense = _compute_distill_term(
            s_dense, t_dense, divergence, temp, labels=labels, chunk_size=None, use_checkpoint=False
        )
        loss_dense.backward()
        grad_dense = s_dense.grad.clone()

        s_ckpt = s_dense.detach().clone().requires_grad_(True)
        loss_ckpt = _compute_distill_term(
            s_ckpt, t_dense, divergence, temp, labels=labels, chunk_size=3, use_checkpoint=True
        )
        loss_ckpt.backward()
        grad_ckpt = s_ckpt.grad.clone()

        tol = 5e-3 if dtype_name == "bfloat16" else 1e-6
        assert abs(loss_dense.item() - loss_ckpt.item()) < tol
        assert (grad_dense - grad_ckpt).abs().max().item() < tol

    @pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
    @pytest.mark.parametrize("chunk_size", [None, 1, 3, 7, 32])
    @pytest.mark.parametrize("use_checkpoint", [False, True])
    def test_matches_pre_pr_reference_kernel(
        self, divergence: str, chunk_size: int | None, use_checkpoint: bool
    ) -> None:
        """BLOCKING 3: pin against the pre-PR origin/main reference formula, not ourselves."""
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        def _reference_pre_pr_distill(s_in, t_in, div, temperature, labels_in=None):
            kl_div = torch.nn.functional.kl_div
            if labels_in is not None:
                s_in = s_in[:, :-1, :]
                t_in = t_in[:, :-1, :]
                labels_in = labels_in[:, 1:]
            temp_val = float(temperature)
            s_scaled = s_in / temp_val
            t_scaled = t_in / temp_val

            def _masked_mean(per_token):
                if labels_in is not None:
                    mask = (labels_in != -100).to(per_token.dtype)
                    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
                return per_token.mean()

            if div == "forward_kl":
                log_s = torch.log_softmax(s_scaled, dim=-1)
                p_t = torch.softmax(t_scaled, dim=-1)
                per_tok = kl_div(log_s, p_t, reduction="none").sum(dim=-1)
                return _masked_mean(per_tok) * (temp_val * temp_val)
            if div == "reverse_kl":
                log_t = torch.log_softmax(t_scaled, dim=-1)
                p_s = torch.softmax(s_scaled, dim=-1)
                per_tok = kl_div(log_t, p_s, reduction="none").sum(dim=-1)
                return _masked_mean(per_tok) * (temp_val * temp_val)
            if div == "js":
                log_s = torch.log_softmax(s_scaled, dim=-1)
                log_t = torch.log_softmax(t_scaled, dim=-1)
                p_s = log_s.exp()
                p_t = log_t.exp()
                m = 0.5 * (p_s + p_t)
                log_m = m.clamp(min=1e-12).log()
                kl_pm = kl_div(log_m, p_s, reduction="none").sum(dim=-1)
                kl_qm = kl_div(log_m, p_t, reduction="none").sum(dim=-1)
                return _masked_mean(0.5 * (kl_pm + kl_qm)) * (temp_val * temp_val)
            raise ValueError(f"Unknown divergence {div}")

        torch.manual_seed(999)
        batch, seq, vocab = 2, 8, 16
        temp = 2.0
        s = torch.randn(batch, seq, vocab, requires_grad=True)
        t = torch.randn(batch, seq, vocab)
        labels = torch.tensor([
            [-100, 1, 2, -100, 4, -100, 6, -100],
            [-100, -100, 2, 3, -100, 5, 6, -100],
        ])

        s_ref = s.detach().clone().requires_grad_(True)
        loss_ref = _reference_pre_pr_distill(s_ref, t, divergence, temp, labels_in=labels)
        loss_ref.backward()
        grad_ref = s_ref.grad.clone()

        s_new = s.detach().clone().requires_grad_(True)
        loss_new = _compute_distill_term(
            s_new, t, divergence, temp, labels=labels,
            chunk_size=chunk_size, use_checkpoint=use_checkpoint
        )
        loss_new.backward()
        grad_new = s_new.grad.clone()

        assert abs(loss_ref.item() - loss_new.item()) < 1e-6
        assert (grad_ref - grad_new).abs().max().item() < 1e-6

    def test_checkpoint_explicitly_uses_non_reentrant(self) -> None:
        """Verify that checkpointing explicitly passes use_reentrant=False."""
        torch = _torch_or_skip()
        from unittest.mock import patch

        from soup_cli.trainer.distill import _compute_distill_term

        calls = []
        orig_checkpoint = torch.utils.checkpoint.checkpoint

        def intercepted_checkpoint(*args, **kwargs):
            calls.append(kwargs.get("use_reentrant"))
            return orig_checkpoint(*args, **kwargs)

        with patch("torch.utils.checkpoint.checkpoint", side_effect=intercepted_checkpoint):
            s = torch.randn(2, 4, 8, requires_grad=True)
            t = torch.randn(2, 4, 8)
            _compute_distill_term(s, t, "forward_kl", 2.0, chunk_size=2, use_checkpoint=True)

        assert len(calls) > 0
        assert all(reentrant is False for reentrant in calls)

    def test_distill_trainer_compute_loss_threads_chunk_and_checkpoint_flags(
        self, monkeypatch
    ) -> None:
        """BLOCKING 2: Verify training.distill_chunk_size and distill_checkpoint reach kernel."""
        torch = _torch_or_skip()
        from unittest.mock import MagicMock, patch

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer import distill as distill_mod
        from soup_cli.trainer.distill import DistillTrainerWrapper

        cfg = load_config_from_string("""
base: dummy-student
task: distill
data:
  train: dummy.jsonl
  format: chatml
output: ./out
training:
  teacher_model: dummy-teacher
  distill_chunk_size: 128
  distill_checkpoint: true
""")

        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"vocab_size": 16})()
                self.lin = torch.nn.Linear(16, 16)

            def forward(self, **kw):
                return type("Out", (), {"logits": torch.randn(1, 4, 16, requires_grad=True)})()

        dummy_row = {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
        }
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MockModel()),
            patch("peft.get_peft_model", side_effect=lambda m, c: m),
            patch(
                "soup_cli.utils.peft_wiring.resolve_lora_target_modules",
                return_value=["lin"],
            ),
            patch(
                "soup_cli.data.sft_format.build_format_row",
                return_value=lambda r: dummy_row,
            ),
        ):
            wrapper = DistillTrainerWrapper(cfg, device="cpu")
            wrapper.setup({"train": [{"dummy": 1}]})

        captured_kwargs = {}

        def mock_compute(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor(1.0)

        monkeypatch.setattr(distill_mod, "_compute_distill_term", mock_compute)

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        wrapper.trainer.compute_loss(wrapper.model, inputs)

        assert captured_kwargs.get("chunk_size") == 128
        assert captured_kwargs.get("use_checkpoint") is True

    def test_chunk_size_controls_invocation_count(self, monkeypatch) -> None:
        """Prove distill_chunk_size is not a no-op by verifying invocation count."""
        import math

        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        def _calls(chunk_size: int | None, s_len: int = 64, vocab: int = 32) -> int:
            torch.manual_seed(0)
            s = torch.randn(1, s_len, vocab, requires_grad=True)
            t = torch.randn(1, s_len, vocab)
            count = 0
            real_log_softmax = torch.log_softmax

            def spy(*args, **kwargs):
                nonlocal count
                count += 1
                return real_log_softmax(*args, **kwargs)

            monkeypatch.setattr(torch, "log_softmax", spy)
            _compute_distill_term(
                s,
                t,
                temperature=1.0,
                divergence="forward_kl",
                chunk_size=chunk_size,
                use_checkpoint=False,
            )
            return count

        for cs in (None, 8, 16, 64):
            expected = 1 if cs is None else math.ceil(64 / cs)
            assert _calls(cs) == expected

    def test_all_masked_labels_returns_zero_and_finite_grad(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        s = torch.randn(2, 4, 8, requires_grad=True)
        t = torch.randn(2, 4, 8)
        labels = torch.full((2, 4), -100, dtype=torch.long)

        loss = _compute_distill_term(
            s, t, "forward_kl", 2.0, labels=labels, chunk_size=2, use_checkpoint=True
        )
        assert loss.item() == 0.0
        loss.backward()
        assert s.grad is not None
        assert (s.grad == 0.0).all()

    def test_attention_mask_only_equivalence(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        torch.manual_seed(202)
        s1 = torch.randn(2, 5, 8, requires_grad=True)
        t = torch.randn(2, 5, 8)
        att_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

        loss_dense = _compute_distill_term(
            s1, t, "forward_kl", 2.0, attention_mask=att_mask, chunk_size=None, use_checkpoint=False
        )
        loss_dense.backward()
        grad_dense = s1.grad.clone()

        s2 = s1.detach().clone().requires_grad_(True)
        loss_chunk = _compute_distill_term(
            s2, t, "forward_kl", 2.0, attention_mask=att_mask, chunk_size=2, use_checkpoint=True
        )
        loss_chunk.backward()
        grad_chunk = s2.grad.clone()

        assert abs(loss_dense.item() - loss_chunk.item()) < 1e-6
        assert (grad_dense - grad_chunk).abs().max().item() < 1e-6

    def test_unmasked_inputs_equivalence(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        torch.manual_seed(303)
        s1 = torch.randn(2, 4, 8, requires_grad=True)
        t = torch.randn(2, 4, 8)

        loss_dense = _compute_distill_term(
            s1, t, "forward_kl", 2.0, chunk_size=None, use_checkpoint=False
        )
        loss_dense.backward()
        grad_dense = s1.grad.clone()

        s2 = s1.detach().clone().requires_grad_(True)
        loss_chunk = _compute_distill_term(
            s2, t, "forward_kl", 2.0, chunk_size=2, use_checkpoint=True
        )
        loss_chunk.backward()
        grad_chunk = s2.grad.clone()

        assert abs(loss_dense.item() - loss_chunk.item()) < 1e-6
        assert (grad_dense - grad_chunk).abs().max().item() < 1e-6

    def test_causal_shift_and_response_only_gradient_isolation(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        # seq=4, vocab=4.
        # Logit position 1 predicts token index 2 after causal shift.
        s = torch.zeros(1, 4, 4, requires_grad=True)
        t = torch.zeros(1, 4, 4)
        t[0, 1, 0] = 5.0  # discrepancy at position 1

        # Only token index 2 is supervised
        labels = torch.tensor([[-100, -100, 1, -100]])
        loss = _compute_distill_term(
            s, t, "forward_kl", 1.0, labels=labels, chunk_size=1, use_checkpoint=True
        )
        loss.backward()

        assert loss.item() > 0.0
        assert s.grad is not None
        # Position 1 should receive gradient; positions 0, 2, 3 must have zero gradient
        assert s.grad[0, 1].abs().sum() > 0.0
        assert s.grad[0, 0].abs().sum() == 0.0
        assert s.grad[0, 2].abs().sum() == 0.0
        assert s.grad[0, 3].abs().sum() == 0.0

    def test_invalid_arguments_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        s = torch.randn(1, 2, 4)
        t = torch.randn(1, 2, 4)

        with pytest.raises(TypeError, match="chunk_size must not be bool"):
            _compute_distill_term(s, t, "forward_kl", 2.0, chunk_size=True)

        with pytest.raises(TypeError, match="chunk_size must be int"):
            _compute_distill_term(s, t, "forward_kl", 2.0, chunk_size=1.5)  # type: ignore

        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            _compute_distill_term(s, t, "forward_kl", 2.0, chunk_size=0)

        with pytest.raises(TypeError, match="use_checkpoint must be bool"):
            _compute_distill_term(s, t, "forward_kl", 2.0, use_checkpoint="yes")  # type: ignore

    def test_autograd_retained_activation_savings(self) -> None:
        """Assert that token selection + checkpointing reduces retained autograd tensor bytes."""
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        torch.manual_seed(42)
        batch, seq, vocab = 4, 128, 512
        temp = 2.0

        s = torch.randn(batch, seq, vocab, requires_grad=True)
        t = torch.randn(batch, seq, vocab)
        # 50% response tokens
        labels = torch.full((batch, seq), -100, dtype=torch.long)
        labels[:, 64:] = 1

        def measure_bytes(fn):
            saved = 0
            def pack(tensor):
                nonlocal saved
                saved += tensor.numel() * tensor.element_size()
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                fn()
            return saved

        s_dense = s.detach().clone().requires_grad_(True)
        dense_bytes = measure_bytes(
            lambda: _compute_distill_term(
                s_dense, t, "forward_kl", temp, labels=labels, chunk_size=None, use_checkpoint=False
            )
        )

        s_ckpt = s.detach().clone().requires_grad_(True)
        ckpt_bytes = measure_bytes(
            lambda: _compute_distill_term(
                s_ckpt, t, "forward_kl", temp, labels=labels, chunk_size=32, use_checkpoint=True
            )
        )

        # Selected tokens drop the unmasked 50%, and checkpointing eliminates
        # intermediate log_softmax/kl tensors
        assert ckpt_bytes < dense_bytes
        # Expect at least a 1.5x - 2x reduction on retained autograd bytes
        assert dense_bytes / ckpt_bytes >= 1.5


class TestDistillConfigSchema:
    def test_distill_chunk_size_validates(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: test/model\n"
            "data:\n"
            "  train: dummy.jsonl\n"
            "  format: chatml\n"
            "task: distill\n"
            "training:\n"
            "  teacher_model: teacher/model\n"
            "  distill_chunk_size: 64\n"
            "  distill_checkpoint: true\n"
        )
        assert cfg.training.distill_chunk_size == 64
        assert cfg.training.distill_checkpoint is True

    def test_distill_chunk_size_rejects_bool(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="distill_chunk_size must not be bool"):
            load_config_from_string(
                "base: test/model\n"
                "data:\n"
                "  train: dummy.jsonl\n"
                "  format: chatml\n"
                "task: distill\n"
                "training:\n"
                "  teacher_model: teacher/model\n"
                "  distill_chunk_size: true\n"
            )

    def test_distill_chunk_size_rejects_zero_and_negative(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="distill_chunk_size must be >= 1"):
            load_config_from_string(
                "base: test/model\n"
                "data:\n"
                "  train: dummy.jsonl\n"
                "  format: chatml\n"
                "task: distill\n"
                "training:\n"
                "  teacher_model: teacher/model\n"
                "  distill_chunk_size: 0\n"
            )

    def test_distill_checkpoint_rejects_non_bool(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="distill_checkpoint must be bool"):
            load_config_from_string(
                "base: test/model\n"
                "data:\n"
                "  train: dummy.jsonl\n"
                "  format: chatml\n"
                "task: distill\n"
                "training:\n"
                "  teacher_model: teacher/model\n"
                "  distill_checkpoint: 123\n"
            )

    def test_distill_chunk_fields_require_task_distill(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="Distillation fields.*require task='distill'"):
            load_config_from_string(
                "base: test/model\n"
                "data:\n"
                "  train: dummy.jsonl\n"
                "  format: chatml\n"
                "task: sft\n"
                "training:\n"
                "  distill_chunk_size: 32\n"
            )

        with pytest.raises(ValueError, match="Distillation fields.*require task='distill'"):
            load_config_from_string(
                "base: test/model\n"
                "data:\n"
                "  train: dummy.jsonl\n"
                "  format: chatml\n"
                "task: sft\n"
                "training:\n"
                "  distill_checkpoint: true\n"
            )
