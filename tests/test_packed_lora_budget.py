"""``soup_cli.kernels.packed_lora.budget``: the GPU memory planner (pure arithmetic, CPU).

The golden table is the output of the original planner
(``tests/fixtures/packed_lora_plan_golden.json``; shapes of Qwen2.5-7B and 1.5B, rank-16
LoRA on all seven projections): wherever the original kept every 4-bit pack on the GPU,
the planner here must make the same plan to the byte.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.kernels.packed_lora import budget
from soup_cli.kernels.packed_lora.budget import BudgetTooSmallError, Plan, Shape
from soup_cli.kernels.packed_lora.layout import layer_bytes

MIB = 1 << 20
GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "packed_lora_plan_golden.json").read_text("utf-8"))
DIMS = {
    "7B": dict(layers=28, hidden=3584, inter=18944, q_dim=3584, kv_dim=512, vocab=152064,
               tied=False),
    "1.5B": dict(layers=28, hidden=1536, inter=8960, q_dim=1536, kv_dim=256, vocab=151936,
                 tied=True),
}


def shape_of(name, rank=16):
    d = DIMS[name]
    lora = d["layers"] * rank * ((d["hidden"] + d["q_dim"]) + 2 * (d["hidden"] + d["kv_dim"])
                                 + (d["q_dim"] + d["hidden"]) + 3 * (d["hidden"] + d["inter"]))
    return Shape(d["layers"], d["hidden"], d["inter"], d["q_dim"], d["kv_dim"], d["vocab"], lora,
                 True, d["tied"])


def plan_at(name, mib, tokens=4096, int8=True, acts=None):
    return Plan(shape_of(name), mib * MIB, tokens, int8, acts)


# --- the original planner's plans, reproduced -------------------------------------------------


@pytest.mark.parametrize("row", GOLDEN["plans"],
                         ids=lambda r: f"{r['model']}-{r['tokens']}-{r['mib']}")
def test_plan_matches_the_original_planner(row):
    plan = plan_at(row["model"], row["mib"], row["tokens"])
    assert (plan.k, plan.head_gpu, plan.embed_gpu) == (row["k"], row["head_gpu"], row["embed_gpu"])
    assert plan.est_peak == pytest.approx(row["est_peak"], abs=0.01)
    assert plan.host_bytes() == row["host_bytes"]
    assert plan.pcie_bytes() == (row["h2d"], row["d2h"])
    assert plan.shape.layer(row["tokens"]) == row["layer"]


@pytest.mark.parametrize("row", GOLDEN["refused"],
                         ids=lambda r: f"{r['model']}-{r['tokens']}-{r['mib']}")
def test_budgets_the_original_had_to_stream_for_are_refused(row):
    with pytest.raises(BudgetTooSmallError) as excinfo:
        plan_at(row["model"], row["mib"], row["tokens"])
    assert excinfo.value.minimum_mib > row["mib"]


# --- what a plan promises --------------------------------------------------------------------


def test_plan_spends_the_budget_in_order():
    budgets = (6000, 7000, 9000, 12000, 16000, 20000, 24000)
    plans = [plan_at("7B", mib) for mib in budgets]
    order = [(p.k, p.head_gpu, p.embed_gpu) for p in plans]
    assert order == sorted(order)  # more budget never means less resident
    for plan, mib in zip(plans, budgets):
        assert plan.est_peak <= mib * MIB  # the margin is inside the limit
    assert 0 < plans[1].k < 28 and not plans[1].head_gpu  # 7 GB: some activations, tables on host
    assert plans[-1].k == 28 and plans[-1].head_gpu and plans[-1].embed_gpu


def test_the_7000_mib_plan_of_the_7b_batch_of_8x512():
    plan = plan_at("7B", 7000, tokens=4096)
    assert plan.k == 6 and not plan.head_gpu and not plan.embed_gpu
    assert plan.describe().startswith("plan: weights resident, lm_head+embedding on host, "
                                      "activations 6/28 resident (int8 gate/up stash)")


def test_a_budget_below_the_floor_is_refused_with_the_minimum():
    with pytest.raises(BudgetTooSmallError) as excinfo:
        plan_at("7B", 3000)
    minimum = excinfo.value.minimum_mib
    assert "gpu_budget_mib=3000 is below the minimum of" in str(excinfo.value)
    assert f"{minimum} MiB" in str(excinfo.value)
    plan_at("7B", minimum)  # the minimum itself works ...
    with pytest.raises(BudgetTooSmallError):
        plan_at("7B", minimum - 1)  # ... and one MiB less does not


def test_minimum_budget_mib_is_the_floor_of_the_shape():
    shape = shape_of("1.5B")
    minimum = budget.minimum_budget_mib(shape, 4096)
    assert Plan(shape, minimum * MIB, 4096, True).k <= 1  # the floor, plus rounding up to a MiB
    floor = shape.peak(False, False, 0, 4096) + budget.SAFETY_BYTES
    assert floor <= minimum * MIB * (1 - budget.SAFETY_FRAC) < floor + MIB


def test_activations_offload_keeps_no_layer_resident():
    plan = plan_at("7B", 24000, acts="offload")
    assert plan.k == 0 and not plan.head_gpu and not plan.embed_gpu


def test_activations_resident_keeps_all_layers_or_refuses():
    assert plan_at("7B", 24000, acts="resident").k == 28
    with pytest.raises(BudgetTooSmallError) as excinfo:
        plan_at("7B", 7000, acts="resident")
    assert "activations: resident" in str(excinfo.value)


def test_the_minimum_quoted_for_resident_activations_is_one_the_planner_accepts():
    with pytest.raises(BudgetTooSmallError) as excinfo:
        plan_at("7B", 7000, acts="resident")
    minimum = excinfo.value.minimum_mib
    # the number, not only the text: every layer's saved tensors plus the margin the limit keeps
    # (the 64 MiB and the 2 % of the budget, as minimum_budget_mib counts them)
    peak = shape_of("7B").peak(False, False, 28, 4096)
    assert minimum == math.ceil((peak + budget.SAFETY_BYTES) / (1 - budget.SAFETY_FRAC) / MIB)
    assert "the budget is 7000 MiB" in str(excinfo.value)
    assert plan_at("7B", minimum, acts="resident").k == 28  # the quoted minimum works ...
    with pytest.raises(BudgetTooSmallError):
        plan_at("7B", minimum - 1, acts="resident")  # ... and one MiB less does not


def test_acts_resident_scales_with_tokens_and_the_cap_lowers_it():
    plan = plan_at("1.5B", 3000, tokens=4096)
    full = plan.acts_resident(4096)
    assert plan.acts_resident(2048) >= full
    plan.cap = max(0, full - 3)
    assert plan.acts_resident(4096) == max(0, full - 3)


def test_the_int8_stash_changes_the_layer_size_and_the_count():
    plan = plan_at("7B", 7000)
    with_int8 = plan.k
    plan.set_int8(False)
    assert plan.k < with_int8 and plan.int8 is False
    assert plan.est_peak <= plan.limit
    plan.set_int8(True)
    assert plan.k == with_int8


def test_layer_bytes_match_the_stash_formula():
    # 78 KB per token for Qwen2.5-7B with the int8 gate/up stash + the x of the LoRA
    assert abs(shape_of("7B").layer(4096) / 4096 - 78_144) < 64
    assert layer_bytes(8, 64, 256, 64, 32) < layer_bytes(8, 64, 256, 64, 32, int8=False) + 1
    # 16 tokens or fewer never take the int8 path (torch._int_mm wants more rows)
    assert layer_bytes(16, 64, 256, 64, 32) == layer_bytes(16, 64, 256, 64, 32, int8=False)
    # an intermediate size that is not a multiple of 64 cannot be packed
    assert layer_bytes(64, 64, 250, 64, 32) == layer_bytes(64, 64, 250, 64, 32, int8=False)
    # Qwen3's q/k norms save x_hat of q and k
    assert layer_bytes(64, 64, 256, 64, 32, qk_norm=True) - layer_bytes(64, 64, 256, 64, 32) == (
        64 * 2 * (64 + 32))


# --- shapes from a model or a config ------------------------------------------------------------


def _config(**over):
    base = dict(num_hidden_layers=28, hidden_size=3584, intermediate_size=18944,
                num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
                tie_word_embeddings=False, model_type="qwen2")
    return SimpleNamespace(**{**base, **over})


def test_shape_of_config_matches_the_hand_made_7b_shape():
    mine = Shape.of_config(_config())
    ref = shape_of("7B")
    for name in ("layers", "hidden", "inter", "q_dim", "kv_dim", "vocab", "lora_params", "slot",
                 "aux", "table", "small", "tied", "qk_norm"):
        assert getattr(mine, name) == getattr(ref, name), name


def test_shape_of_config_reads_an_explicit_head_dim_and_qwen3():
    shape = Shape.of_config(_config(head_dim=256, model_type="qwen3"))
    assert shape.q_dim == 28 * 256 and shape.kv_dim == 4 * 256 and shape.qk_norm


def test_shape_of_model_counts_trainable_parameters_as_the_lora():
    class Param:
        def __init__(self, n, grad):
            self._n, self.requires_grad = n, grad

        def numel(self):
            return self._n

    class Embedding:
        weight = object()

    shared = Embedding()
    attn = SimpleNamespace(q_norm=object())
    model = SimpleNamespace(
        config=_config(tie_word_embeddings=True),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        parameters=lambda: [Param(1000, True), Param(10**9, False), Param(24, True)],
        get_input_embeddings=lambda: shared, get_output_embeddings=lambda: shared)
    shape = Shape.of_model(model)
    assert shape.lora_params == 1024 and shape.tied and shape.qk_norm


def test_shape_of_model_without_embedding_accessors_falls_back_to_the_config():
    model = SimpleNamespace(
        config=_config(tie_word_embeddings=True),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]),
        parameters=lambda: [])
    assert Shape.of_model(model).tied is True


# --- the guard ----------------------------------------------------------------------------------


def test_observe_lowers_the_cap_only_after_a_step_went_over():
    plan = plan_at("1.5B", 3000, tokens=4096)
    before = plan.acts_resident(4096)
    lines = []
    budget.observe(plan, 2000 * MIB, lines.append)  # the first call only notes the baseline
    assert plan.base_peak == 2000 * MIB and plan.cap == plan.shape.layers and not lines
    budget.observe(plan, 2900 * MIB, lines.append)  # under the budget: nothing
    assert plan.cap == plan.shape.layers and not lines
    over = plan.budget + 2 * plan.shape.layer(4096)
    budget.observe(plan, over, lines.append)
    assert plan.cap < before and plan.acts_resident(4096) == plan.cap
    assert len(lines) == 1 and "budget exceeded" in lines[0]
    cap = plan.cap
    budget.observe(plan, over, lines.append)  # the same peak again does not cut twice
    assert plan.cap == cap and len(lines) == 1


def test_observe_never_raises_the_cap_and_never_goes_below_zero():
    plan = plan_at("1.5B", 3000, tokens=4096)
    budget.observe(plan, 100 * MIB, lambda _: None)
    budget.observe(plan, plan.budget * 5, lambda _: None)
    assert plan.cap == 0
    assert plan.acts_resident(4096) == 0


def test_set_int8_to_the_same_value_changes_nothing():
    plan = plan_at("7B", 7000)
    k, est = plan.k, plan.est_peak
    plan.set_int8(True)
    assert (plan.k, plan.est_peak) == (k, est)


def test_step_reserve_bytes():
    from soup_cli.kernels.packed_lora.layout import step_reserve_bytes

    tokens, hidden, inter = 4096, 3584, 18944
    assert step_reserve_bytes(tokens, hidden, inter) == (
        24 * tokens * inter + 16 * tokens * hidden + 6 * inter * hidden)
    assert step_reserve_bytes(tokens, hidden, inter, 100) == step_reserve_bytes(
        tokens, hidden, inter) + 300
