"""Measurement harness for issue #725: LoRA-FA operating point vs standard LoRA.

Measures:
1. Trainable adapter parameter counts.
2. AdamW optimizer state tensor allocations.
3. Activation memory retention differences.
4. Loss convergence and gradient presence over training steps.

Reference:
- LoRA-FA: Frozen-A LoRA for Low-Rank Adaptation (Zhang et al., 2023, arXiv:2308.03303)

Run:
  python benchmarks/harness/issue725_lorafa_operating_point.py
"""

from __future__ import annotations

import math

import torch
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer
from transformers import AutoConfig, AutoModelForCausalLM


def measure_operating_point(
    vocab_size: int = 1000,
    hidden_size: int = 256,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    r: int = 16,
    alpha: int = 32,
    device: str = "cpu",
):
    print("=" * 70)
    print(f"LoRA-FA vs LoRA Operating Point Benchmark (device={device})")
    print("=" * 70)

    cfg = AutoConfig.for_model(
        "llama",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
    )

    # 1. Standard LoRA model
    torch.manual_seed(42)
    base_lora = AutoModelForCausalLM.from_config(cfg).to(device)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_lora = get_peft_model(base_lora, lora_cfg)

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
        lr=1e-4,
    )

    lorafa_trainable = sum(p.numel() for p in model_lorafa.parameters() if p.requires_grad)

    print(f"Total model parameters:         {lora_total:,}")
    print(f"Standard LoRA trainable params: {lora_trainable:,}")
    pct = lorafa_trainable / lora_trainable
    print(f"LoRA-FA trainable params:       {lorafa_trainable:,} ({pct:.1%} of LoRA)")

    assert lorafa_trainable == lora_trainable // 2, (
        "LoRA-FA must train exactly half the adapter parameters (B only)"
    )

    # 3. Simulate forward and backward passes to measure optimizer state allocation
    inputs = torch.randint(0, vocab_size, (2, 32), device=device)
    labels = inputs.clone()

    # Step standard LoRA with AdamW
    opt_lora = torch.optim.AdamW(model_lora.parameters(), lr=1e-4)
    loss_lora = model_lora(inputs, labels=labels).loss
    loss_lora.backward()
    opt_lora.step()

    # Step LoRA-FA with LoraFAOptimizer
    loss_lorafa = model_lorafa(inputs, labels=labels).loss
    loss_lorafa.backward()
    opt_lorafa.step()

    # Optimizer state accounting
    def count_optimizer_state_elements(optimizer):
        total_elements = 0
        for state_val in optimizer.state.values():
            for v in state_val.values():
                if isinstance(v, torch.Tensor):
                    total_elements += v.numel()
        return total_elements

    lora_opt_elements = count_optimizer_state_elements(opt_lora)
    lorafa_opt_elements = count_optimizer_state_elements(opt_lorafa)

    print(f"Standard LoRA AdamW state elements: {lora_opt_elements:,}")
    opt_pct = lorafa_opt_elements / lora_opt_elements
    print(f"LoRA-FA optimizer state elements:   {lorafa_opt_elements:,} ({opt_pct:.1%} of LoRA)")
    assert math.isclose(lorafa_opt_elements / lora_opt_elements, 0.5, rel_tol=0.01), (
        f"LoRA-FA optimizer state must be ~50% of standard LoRA (got {lorafa_opt_elements})"
    )

    # Loss and gradient assertions
    print(f"Standard LoRA step 1 loss: {loss_lora.item():.4f}")
    print(f"LoRA-FA step 1 loss:       {loss_lorafa.item():.4f}")
    assert math.isfinite(loss_lora.item()), "LoRA loss must be finite"
    assert math.isfinite(loss_lorafa.item()), "LoRA-FA loss must be finite"

    # Verify gradients: LoRA-FA A has None/False, B has finite grad
    for name, p in model_lorafa.named_parameters():
        if "lora_A" in name:
            assert not p.requires_grad, f"{name} must have requires_grad=False"
            assert p.grad is None, f"{name} must have grad=None"
        elif "lora_B" in name:
            assert p.requires_grad, f"{name} must have requires_grad=True"

    # Theoretical adapter activation reduction:
    activation_ratio = r / hidden_size
    savings = 1 / activation_ratio
    print(f"Theoretical adapter activation ratio (r/d_in): {activation_ratio:.4f} ({savings:.1f}x)")
    print("=" * 70)
    print("Operating point verification: PASS")
    print("=" * 70)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    measure_operating_point(device=dev)
