"""#342 — prevent ``exp(log_ratio)`` overflow at padding positions.

When ``old_per_token_logps`` at padding positions diverge beyond ~88 from
the current model's log-probs, ``torch.exp(log_ratio)`` overflows to
``inf``.  PyTorch's ``ExpBackward`` then computes ``0.0 × inf = NaN``
(IEEE 754), corrupting all gradients while the forward loss stays finite.

Clamping ``old_per_token_logps`` to ``>= -20.0`` ensures the maximum
``log_ratio`` is 20 (since ``per_token_logps ≤ 0``), giving
``exp(20) ≈ 4.85 × 10⁸`` which is finite.  The PPO clip at
``[1 − ε, 1 + ε]`` further constrains the ratio, and the completion mask
zeros the loss contribution at padding positions anyway.
Bit-identical for all normal tokens (``log_prob ∈ [-15, 0]``).

``old_per_token_logps`` feeds only the importance ratio; the KL penalty
reads from ``ref_per_token_logps``, a separate tensor.  The clamp does
not reach the KL term.

Design decision (recorded per maintainer approval on 2026-09-18):
clamping an *input* rather than the computed ``exp(log_ratio)`` quantity
avoids coupling to TRL internals (``_compute_loss``) and uses the stable
``training_step`` API.
"""

from __future__ import annotations

LOG_PROB_CLAMP_MIN: float = -20.0


def build_safe_grpo_trainer(base_cls: type) -> type:
    """Create a thin subclass that clamps ``old_per_token_logps``.

    Injected **before** the variant gate in ``grpo.py`` ``setup()`` so
    both ``variant=None`` and named-variant paths are protected.

    The subclass overrides ``training_step`` — a stable HuggingFace
    ``Trainer`` API — to clamp ``old_per_token_logps`` to
    ``>= LOG_PROB_CLAMP_MIN`` before delegating to ``super()``.
    """

    class _SafeGRPOTrainer(base_cls):
        def training_step(self, model, inputs, num_items_in_batch=None):
            import torch

            old = inputs.get("old_per_token_logps")
            if old is not None:
                inputs = {
                    **inputs,
                    "old_per_token_logps": torch.clamp(
                        old, min=LOG_PROB_CLAMP_MIN
                    ),
                }
            return super().training_step(
                model, inputs, num_items_in_batch
            )

    _SafeGRPOTrainer.__name__ = (
        f"_SafeGRPOTrainer_{base_cls.__name__}"
    )
    _SafeGRPOTrainer.__qualname__ = _SafeGRPOTrainer.__name__
    return _SafeGRPOTrainer
