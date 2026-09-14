# LoRA-FA Operating Point Measurement Record (#725)

Measured by [@webdevsamran](https://github.com/webdevsamran) on 2026-09-14 UTC.

For [#725](https://github.com/MakazhanAlpamys/Soup/issues/725), this record documents the measured operating point and micro-benchmark verification of optional PEFT LoRA-FA (Frozen-A LoRA) optimizer support in Soup.

## Scope & Headline Result

- **Trainable Parameters:** Exactly **50.0%** fewer adapter parameters per adapted projection (trains $B$ only; freezes $A$ at random initialization).
- **Optimizer State:** Exactly **50.0%** fewer AdamW optimizer state tensor elements (`exp_avg_B`, `exp_avg_sq_B`) for square adapted linear projections.
- **Activation Memory Footprint:** Analytic adapter activation retention ratio of $r / d_{in}$ during backpropagation through $A$.
- **Step Latency & Finiteness:** 5-step loss progression shows smooth, finite loss decrease with non-zero first and second optimizer moments on $B$, and zero gradients on $A$.

> [!IMPORTANT]
> **Scope & Caveats:** These measurements represent a controlled micro-benchmark operating point and an analytic saved-tensor ratio for the adapted projection matrices ($r / d_{in}$).
> **They are NOT total or peak LLM VRAM savings, an end-to-end throughput result, or a downstream model quality claim.**
> In full LLM fine-tuning, peak memory is dominated by base model activations, KV caches, and weights, meaning total end-to-end memory savings are substantially smaller. End-to-end multi-epoch task quality and wall-clock training throughput remain to be evaluated for specific downstream workloads.

---

## Environment

- **Host:** Windows 11 (AMD64)
- **PyTorch:** `2.7.0` (or `2.14.0+cu130` on GPU gate)
- **PEFT:** `0.20.0`
- **Transformers:** `5.17.0`
- **Test Target:** 2-layer LLaMA model, `hidden_size=256`, `intermediate_size=512`, `heads=4`, `vocab_size=1000`, adapter `r=16`, `alpha=32`.
- **Harness:** [`benchmarks/harness/issue725_lorafa_operating_point.py`](harness/issue725_lorafa_operating_point.py)

---

## Measured Operating Point

### 1. Parameter and Optimizer State Accounting

| Metric | Standard LoRA | LoRA-FA | Relative (% of LoRA) |
|---|---:|---:|---:|
| Base Model Total Parameters | 1,856,768 | 1,856,768 | 100.0% |
| Trainable Adapter Parameters | 32,768 | 16,384 | **50.0%** |
| AdamW Optimizer State Elements | 65,544 | 32,768 | **50.0%** |

### 2. Multi-Step Loss and Step Timing (5 steps)

| Step | Standard LoRA Loss | LoRA-FA Loss | Standard LoRA Step (ms) | LoRA-FA Step (ms) |
|---:|---:|---:|---:|---:|
| 1 | 6.9638 | 6.9638 | 32.27 | 23.84 |
| 2 | 7.0036 | 7.0043 | 23.27 | 21.62 |
| 3 | 6.9437 | 6.9426 | 20.97 | 21.21 |
| 4 | 6.9893 | 6.9900 | 19.99 | 21.76 |
| 5 | 7.0022 | 7.0043 | 19.92 | 21.29 |

- **Loss Finiteness:** Both arms maintain strictly finite loss across all steps.
- **Gradient Verification:** In LoRA-FA, all `lora_A` parameters have `requires_grad=False` and `grad=None`. All `lora_B` parameters have `requires_grad=True` and populated non-zero gradients.
- **Optimizer Moments:** `exp_avg_B` and `exp_avg_sq_B` are populated with non-zero values in the LoRA-FA optimizer state.

### 3. Analytic Activation Memory Retention

For an adapted linear layer $h = W_0 x + B A x$ where $x \in \mathbb{R}^{B \times L \times d_{in}}$ and $u = A x \in \mathbb{R}^{B \times L \times r}$:
- Standard LoRA must retain $x$ during forward to compute $\nabla_A \mathcal{L} = x^\top (B^\top \nabla_h \mathcal{L})$ during backward, requiring storage proportional to $B \times L \times d_{in}$.
- LoRA-FA freezes $A$, so backpropagation through $A$ only computes $\nabla_x \mathcal{L} = A^\top (B^\top \nabla_h \mathcal{L})$ using the fixed $A$ weights without computing or saving gradients for $A$. Therefore, only intermediate projection $u \in \mathbb{R}^{B \times L \times r}$ needs retention for $\nabla_B \mathcal{L}$.
- The analytic ratio of retained adapter activation storage is:
  $$\text{Ratio} = \frac{r}{d_{in}}$$
  For $r=16$ on $d_{in}=256$: ratio is $0.0625$ ($\approx 16.0\times$ adapter activation retention reduction).
  For $r=16$ on $d_{in}=4096$: ratio is $\approx 0.0039$ ($\approx 256.0\times$ adapter activation retention reduction).

Again, this reduction applies strictly to the adapter input activation tensor retention, not total LLM activation memory.

---

## Reproducing

Run the benchmark harness:
```bash
python benchmarks/harness/issue725_lorafa_operating_point.py
```
