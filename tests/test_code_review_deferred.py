"""Regression tests for the 3 items deferred in the first code-review pass:
windowed EMA smoothing, the hardware-fit OOM preflight, and real MoD
token-dropping.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string

# ─────────────────────────── EMA smoothing window ───────────────────────────


def test_ema_smoothing_now_respects_window_size():
    from soup_cli.utils.reward_hack_control import smooth_signal

    # Windowed EMA folds alpha over the whole retained window, so a longer
    # window (bounded by reward_hack_smoothing_window) changes the result.
    assert smooth_signal(0.4, [0.1, 0.2], method="ema") == pytest.approx(0.275)
    assert smooth_signal(0.4, [0.2], method="ema") == pytest.approx(0.3)
    assert smooth_signal(1.0, [0.0], method="ema") != smooth_signal(
        1.0, [1.0, 0.0, 0.0], method="ema"
    )


# ─────────────────────────── hardware-fit OOM preflight ─────────────────────

_FIT_YAML = """
base: meta-llama/Llama-2-7b-hf
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: 8
  quantization: none
"""

_AUTO_BS_YAML = """
base: meta-llama/Llama-2-7b-hf
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: auto
  quantization: none
"""


def test_build_hardware_fit_input_from_config():
    from soup_cli.commands.train import _build_hardware_fit_input

    inp = _build_hardware_fit_input(load_config_from_string(_FIT_YAML))
    assert inp is not None
    assert inp.batch_size == 8
    assert inp.seq_len == 2048
    assert inp.params_b >= 6.0  # a 7B base

    # batch_size="auto" isn't statically predictable -> skip the gate.
    assert _build_hardware_fit_input(load_config_from_string(_AUTO_BS_YAML)) is None


def test_hardware_fit_preflight_gate_and_optout():
    import typer

    from soup_cli.commands import train as train_mod

    cfg = load_config_from_string(_FIT_YAML)

    # 4 GB can't hold a 7B model -> refuse (exit) by default.
    with pytest.raises(typer.Exit):
        train_mod._hardware_fit_preflight(
            cfg, {"memory_total_bytes": int(4e9)}, allow_oom_attempt=False
        )

    # --allow-oom-attempt -> warn, don't refuse.
    train_mod._hardware_fit_preflight(
        cfg, {"memory_total_bytes": int(4e9)}, allow_oom_attempt=True
    )

    # No detectable VRAM (CPU / CI) -> skip silently.
    train_mod._hardware_fit_preflight(
        cfg, {"memory_total_bytes": 0}, allow_oom_attempt=False
    )

    # Plenty of VRAM -> fits, no refuse.
    train_mod._hardware_fit_preflight(
        cfg, {"memory_total_bytes": int(500e9)}, allow_oom_attempt=False
    )


# ─────────────────────────── MoD real token-dropping ────────────────────────


def _mod_pieces(hidden: int, capacity_factor: float):
    import torch.nn as nn

    from soup_cli.utils.mod import _make_mod_forward

    class _RecordingLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seqs: list = []
            self.kw: list = []
            self.lin = nn.Linear(hidden, hidden)

        def forward(self, hs, *args, **kwargs):
            self.seqs.append(hs.shape[1])
            self.kw.append(kwargs)
            return self.lin(hs)  # per-token (no attention mixing) — plumbing test

    layer = _RecordingLayer()
    router = nn.Linear(hidden, 1, bias=False)
    fwd = _make_mod_forward(layer.forward, router, capacity_factor)
    return layer, router, fwd


def test_mod_forward_gathers_and_saves_compute():
    torch = pytest.importorskip("torch")

    hidden, seq, batch = 4, 8, 2
    torch.manual_seed(0)
    layer, router, fwd = _mod_pieces(hidden, 0.5)  # cap = 4
    x = torch.randn(batch, seq, hidden)

    cap = 4
    topk = torch.topk(router(x).squeeze(-1), k=cap, dim=-1).indices
    topk, _ = torch.sort(topk, dim=-1)

    out = fwd(x)
    # The block ran on ONLY the cap tokens — the whole point (real savings).
    assert layer.seqs == [cap]
    assert out.shape == x.shape
    # Unselected tokens pass through unchanged.
    selected = torch.zeros(batch, seq, dtype=torch.bool)
    selected.scatter_(1, topk, True)
    for b in range(batch):
        for t in range(seq):
            if not selected[b, t]:
                assert torch.allclose(out[b, t], x[b, t])


def test_mod_forward_gathers_positional_inputs():
    torch = pytest.importorskip("torch")

    hidden, seq, batch, head_dim, heads = 4, 8, 2, 6, 1
    torch.manual_seed(1)
    layer, router, fwd = _mod_pieces(hidden, 0.5)  # cap = 4
    x = torch.randn(batch, seq, hidden)
    cos = torch.randn(batch, seq, head_dim)
    sin = torch.randn(batch, seq, head_dim)
    attn = torch.zeros(batch, heads, seq, seq)

    fwd(x, position_embeddings=(cos, sin), attention_mask=attn)
    kw = layer.kw[-1]
    # RoPE + mask narrowed to the sub-sequence (cap), proving real savings with
    # correctly-gathered positional inputs.
    assert layer.seqs == [4]
    assert kw["position_embeddings"][0].shape == (batch, 4, head_dim)
    assert kw["attention_mask"].shape == (batch, heads, 4, 4)


def test_mod_forward_falls_back_on_positional_args():
    torch = pytest.importorskip("torch")

    hidden, seq, batch = 4, 8, 2
    torch.manual_seed(2)
    layer, router, fwd = _mod_pieces(hidden, 0.5)
    x = torch.randn(batch, seq, hidden)

    # A positional forward arg aborts the gather path -> full block (seq == T)
    # + gate-blend fallback (correct, no savings).
    out = fwd(x, object())
    assert layer.seqs == [seq]
    assert out.shape == x.shape
