"""Measurement harness for issue #725: LoRA-FA operating point vs standard LoRA.

Measures:
1. Trainable adapter parameter counts.
2. AdamW optimizer state tensor allocations.
3. Multi-step loss progression and gradient presence.
4. Step latency using time.perf_counter.
5. Adapter activation memory retention differences (analytic r / d_in).

Reference:
- LoRA-FA: Frozen-A LoRA for Low-Rank Adaptation (Zhang et al., 2023, arXiv:2308.03303)

Run:
  python benchmarks/harness/issue725_lorafa_operating_point.py
"""

from __future__ import annotations

import math
import time

import torch
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer
from transformers import AutoConfig, AutoModelForCausalLM


def count_optimizer_state_elements(optimizer: torch.optim.Optimizer) -> int:
    total_elements = 0
    for state_val in optimizer.state.values():
        for v in state_val.values():
            if isinstance(v, torch.Tensor):
                total_elements += v.numel()
    return total_elements


def measure_operating_point(
    vocab_size: int = 1000,
    hidden_size: int = 256,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    r: int = 16,
    alpha: int = 32,
    steps: int = 5,
    device: str = "cpu",
) -> dict:
    print("=" * 75)
    print(f"LoRA-FA vs LoRA Operating Point Benchmark (device={device}, steps={steps})")
    print("=" * 75)

    cfg = AutoConfig.for_model(
        "llama",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
    )

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 1. Standard LoRA model
    torch.manual_seed(42)
    base_lora = AutoModelForCausalLM.from_config(cfg).to(device)
    model_lora = get_peft_model(base_lora, lora_cfg)
    opt_lora = torch.optim.AdamW(model_lora.parameters(), lr=1e-3)

    lora_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in model_lora.parameters())

    # 2. LoRA-FA model & optimizer
    torch.manual_seed(42)
    base_lorafa = AutoModelForCausalLM.from_config(cfg).to(device)
    model_lorafa = get_peft_model(base_lorafa, lora_cfg)
    opt_lorafa = create_lorafa_optimizer(
        model=model_lorafa,
        r=r,
        lora_alpha=alpha,
        lr=1e-3,
    )

    lorafa_trainable = sum(p.numel() for p in model_lorafa.parameters() if p.requires_grad)

    print(f"Total model parameters:             {lora_total:,}")
    print(f"Standard LoRA trainable parameters: {lora_trainable:,}")
    pct = lorafa_trainable / lora_trainable
    print(f"LoRA-FA trainable parameters:       {lorafa_trainable:,} ({pct:.1%} of LoRA)")
    assert lorafa_trainable == lora_trainable // 2, (
        "LoRA-FA must train exactly half the adapter parameters (B only)"
    )

    # Deterministic multi-step training inputs
    torch.manual_seed(123)
    batches = [
        torch.randint(0, vocab_size, (2, 32), device=device)
        for _ in range(steps)
    ]

    # Run standard LoRA multi-step
    lora_losses = []
    lora_step_times = []
    for batch in batches:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt_lora.zero_grad()
        out = model_lora(batch, labels=batch)
        loss = out.loss
        loss.backward()
        opt_lora.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lora_losses.append(loss.item())
        lora_step_times.append(dt)

    # Run LoRA-FA multi-step
    lorafa_losses = []
    lorafa_step_times = []
    for batch in batches:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt_lorafa.zero_grad()
        out = model_lorafa(batch, labels=batch)
        loss = out.loss
        loss.backward()
        opt_lorafa.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lorafa_losses.append(loss.item())
        lorafa_step_times.append(dt)

    # Optimizer state counting
    lora_opt_elements = count_optimizer_state_elements(opt_lora)
    lorafa_opt_elements = count_optimizer_state_elements(opt_lorafa)

    print(f"\nStandard LoRA AdamW state elements: {lora_opt_elements:,}")
    opt_pct = lorafa_opt_elements / lora_opt_elements
    print(f"LoRA-FA optimizer state elements:   {lorafa_opt_elements:,} ({opt_pct:.1%} of LoRA)")
    assert math.isclose(lorafa_opt_elements / lora_opt_elements, 0.5, rel_tol=0.01), (
        f"LoRA-FA optimizer state must be ~50% of standard LoRA (got {lorafa_opt_elements})"
    )

    print("\nMulti-step Loss Progression:")
    print("Step | Standard LoRA Loss | LoRA-FA Loss | LoRA Step (ms) | LoRA-FA Step (ms)")
    print("-" * 75)
    for s in range(steps):
        print(
            f"{s + 1:4d} | {lora_losses[s]:18.4f} | {lorafa_losses[s]:12.4f} | "
            f"{lora_step_times[s] * 1000:14.2f} | {lorafa_step_times[s] * 1000:17.2f}"
        )

    # Assert loss finiteness
    assert all(
        math.isfinite(loss_val) for loss_val in lora_losses
    ), "All LoRA losses must be finite"
    assert all(
        math.isfinite(loss_val) for loss_val in lorafa_losses
    ), "All LoRA-FA losses must be finite"

    # Verify parameter gradients and freezing in LoRA-FA
    for name, p in model_lorafa.named_parameters():
        if "lora_A" in name:
            assert not p.requires_grad, f"{name} must have requires_grad=False"
            assert p.grad is None, f"{name} must have grad=None"
        elif "lora_B" in name:
            assert p.requires_grad, f"{name} must have requires_grad=True"

    # Verify first/second moments in LoRA-FA state
    lora_states = [s for s in opt_lorafa.state.values() if "exp_avg_B" in s]
    assert lora_states, "Expected LoRA-FA optimizer state to contain exp_avg_B"
    assert all(s["exp_avg_B"].abs().sum().item() > 0 for s in lora_states), (
        "Non-zero first moment expected on LoRA B"
    )
    assert all(s["exp_avg_sq_B"].abs().sum().item() > 0 for s in lora_states), (
        "Non-zero second moment expected on LoRA B"
    )

    # Analytic adapter activation ratio:
    activation_ratio = r / hidden_size
    analytic_saved_ratio = 1 / activation_ratio
    print("\nAnalytic Adapter Activation Retention:")
    print(f"- Target rank r:                    {r}")
    print(f"- Hidden dimension d_in:            {hidden_size}")
    print(
        f"- Analytic ratio (r / d_in):        {activation_ratio:.4f} "
        f"(~{analytic_saved_ratio:.1f}x adapter activation reduction)"
    )
    print("\n[NOTE] These values represent a micro-benchmark operating point and an analytic")
    print("saved-tensor ratio for the adapter projections. They are NOT total or peak LLM")
    print("VRAM savings, an end-to-end throughput result, or a quality claim.")
    print("=" * 75)
    print("Operating point verification: PASS")
    print("=" * 75)

    return {
        "lora_total": lora_total,
        "lora_trainable": lora_trainable,
        "lorafa_trainable": lorafa_trainable,
        "lora_opt_elements": lora_opt_elements,
        "lorafa_opt_elements": lorafa_opt_elements,
        "lora_losses": lora_losses,
        "lorafa_losses": lorafa_losses,
        "lora_step_times": lora_step_times,
        "lorafa_step_times": lorafa_step_times,
    }


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    measure_operating_point(device=dev)
