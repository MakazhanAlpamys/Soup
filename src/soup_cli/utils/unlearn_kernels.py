"""v0.71.9 #193 — live NPO / SimNPO / RMU unlearning loss kernels.

Pure loss-math helpers (torch tensors in → scalar loss out) shared by
:class:`soup_cli.trainer.unlearn.UnlearnTrainerWrapper`. Keeping them as
standalone functions lets the unit tests exercise the maths with synthetic
tensors (no model load), while the trainer orchestrates the model forward
passes + the two-dataset (forget / retain) loop.

References:
* NPO — Zhang et al. 2024 (arXiv:2404.05868). Negative-only DPO-shaped loss
  that pushes the policy away from the forget set relative to a frozen
  reference.
* SimNPO — Liu et al. 2024. Length-normalised NPO without a reference model.
* RMU — Li et al. 2024 (arXiv:2403.03218). Steers the residual stream toward
  a random control vector on forget inputs while preserving retain activations.

Every ``import torch`` is local (project lazy-import policy).
"""

from __future__ import annotations

import math


def _check_beta(beta: object) -> float:
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be a number")
    fval = float(beta)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ValueError("beta must be a finite positive number")
    return fval


def sequence_logprob(logits, labels):
    """Sum of per-token log-probabilities of the label span, per example.

    ``logits``: ``[batch, seq, vocab]``; ``labels``: ``[batch, seq]`` with
    ``-100`` masking the prompt / padding span. Returns a ``[batch]`` tensor
    of summed log-probs over the supervised (non-masked) tokens.
    """
    import torch.nn.functional as functional

    # Shift so token t predicts label t+1 (standard causal LM convention).
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = functional.log_softmax(shift_logits.float(), dim=-1)
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)


def sequence_lengths(labels):
    """Count supervised (non ``-100``) tokens per example, shifted. ``[batch]``."""
    shift_labels = labels[:, 1:]
    return (shift_labels != -100).sum(dim=-1).clamp(min=1)


def npo_loss(policy_logps, ref_logps, *, beta: float = 0.1):
    """Negative Preference Optimization loss over forget-set sequences.

    ``L = (2/beta) * mean( -logsigmoid(-beta * (policy_logps - ref_logps)) )``.

    Lower policy log-prob (vs the reference) ⇒ lower loss ⇒ the fact is being
    forgotten. ``policy_logps`` / ``ref_logps`` are ``[batch]`` summed
    log-probs from :func:`sequence_logprob`.
    """
    import torch
    import torch.nn.functional as functional

    b = _check_beta(beta)
    ratio = policy_logps - ref_logps
    return (2.0 / b) * torch.mean(-functional.logsigmoid(-b * ratio))


def simnpo_loss(policy_logps, lengths, *, beta: float = 0.1, gamma: float = 0.0):
    """Length-normalised NPO loss without a reference model.

    ``r = policy_logps / lengths - gamma`` ; ``L = (2/beta) * mean(-logsigmoid(-beta*r))``.
    """
    import torch
    import torch.nn.functional as functional

    b = _check_beta(beta)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be a number")
    g = float(gamma)
    if not math.isfinite(g):
        raise ValueError("gamma must be finite")
    # Defend against zero / negative lengths producing inf/NaN — the trainer
    # uses sequence_lengths() (clamped) but the kernel is public.
    norm_logps = policy_logps / lengths.float().clamp(min=1.0)
    ratio = norm_logps - g
    return (2.0 / b) * torch.mean(-functional.logsigmoid(-b * ratio))


def rmu_loss(forget_acts, control_vec, retain_acts, retain_frozen, *, alpha: float = 1.0):
    """Representation Misdirection Unlearning loss.

    Steers the forget-input residual stream toward a fixed ``control_vec`` and
    keeps the retain-input residual close to its frozen reference:

    ``L = mean(||forget_acts - control_vec||^2) + alpha * mean(||retain_acts - retain_frozen||^2)``.
    """
    import torch

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number")
    a = float(alpha)
    if not math.isfinite(a) or a < 0.0:
        raise ValueError("alpha must be a finite non-negative number")
    forget_term = torch.mean((forget_acts - control_vec) ** 2)
    retain_term = torch.mean((retain_acts - retain_frozen) ** 2)
    return forget_term + a * retain_term
