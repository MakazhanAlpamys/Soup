# GRPO non-finite gradient numerical probes

`benchmarks/harness/grpo_nan_numeric_probes.py` contains CPU-only arithmetic and autograd controls for Issue #342.

## Purpose and evidence boundary

This harness validates specific numerical bounds and small PyTorch autograd mechanisms. It does not load a model, download data, initialise CUDA, replay the reported H100 incident, or identify a root cause. It makes no tagged-version source conclusions.

- **Arithmetic:** Conditional contraction bound and PyTorch dtype limits.
- **CPU autograd probes:** Deliberately constructed minimal graphs demonstrating autograd behavior on CPU.
- **Evidence boundary:** Not an H100 reproduction, SDPA backend attribution, or root-cause determination.

## Claims and boundaries

- **`split_k_bound`:** Dense contraction bound of 6,553,600 is >31 orders of magnitude below single-precision `float32` max (3.4028e38).  
  *What it does not show:* The operand bounds of 10 (|X|, |G| <= 10) and K = 65,536 are assumed, not measured on any run. Measuring them is the responsibility of the opt-in capture harness.
- **`dtype_ranges`:** Documents bfloat16 max (3.3895e38), float16 max (65,504.0), and ln(bf16_max) ≈ 88.719.  
  *What it does not show:* Does not demonstrate that intermediate tensors in attention or logit computation approached these boundaries during actual runs.
- **`constructed_masked_exponential`:** Demonstrates that an ostensibly masked, overflowing exponential can produce a non-finite gradient (0 * inf = NaN in `ExpBackward` at position 2) alongside a finite scalar loss.  
  *What it does not show:* Does not show that this specific computation graph ran in TRL 0.26.2 or in the Issue #342 runs.
- **`detached_logp_ratio`:** Demonstrates that detaching the baseline log-probability `exp(logp - logp.detach())` yields a unit gradient d/dx exp(x - c) = 1 rather than 0.  
  *What it does not show:* Shows PyTorch behavior only, not that TRL's GRPO at the incident's tag computed its ratio this way.
- **`boolean_selection`:** Demonstrates that `torch.where` can avoid multiplying an existing infinity when that infinity is already a leaf tensor.  
  *What it does not show:* Holds only for an infinity that is already a leaf tensor. When infinity is produced by `exp` in the same graph, the gradient remains non-finite.
- **`selection_after_overflow`:** Contrast control showing that `torch.where` following an in-graph overflowing `exp` yields a finite loss but a non-finite gradient.
- **`adamw_skip_semantics`:** Confirms that `grad is None` leaves the parameter unchanged with 0 optimizer state entries, whereas an explicit zero gradient applies weight decay and creates 1 optimizer state entry.
- **`nanmean_visibility`:** Demonstrates that `torch.nanmean` conceals an invalid non-finite token position, returning a finite mean.  
  *What it does not show:* Shows PyTorch behavior only, not that TRL's GRPO computed its ratio or KL metric this way at the incident's tag.
- **`view_backward`:** Confirms that a non-contiguous transposed view slice produces a finite backward pass on CPU.  
  *What it does not show:* Does not verify GPU kernel behavior or strided memory access under distributed backends.

## Execution

```bash
python benchmarks/harness/grpo_nan_numeric_probes.py --self-test --json
```

- Exits `0` when all assertions pass.
- Exits `1` and outputs `selftest_passed: false` with `selftest_error` naming the failing probe if any assertion fails.
