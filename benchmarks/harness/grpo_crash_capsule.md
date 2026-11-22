# Issue #342 GRPO crash capsule

`grpo_crash_capsule.py` is an **opt-in forensic harness** for a real GPU
reproduction of [#342](https://github.com/MakazhanAlpamys/Soup/issues/342).
It is not imported by Soup's CLI or trainer code. It does not reproduce the
incident on CPU, assign an SDPA backend, or identify a root cause.

**Evidence labels.** The source-reading sections describe this module's
contract, starting from [Soup `ad364463`][base] and including the changes
for [#2667]. The CPU-control section describes synthetic tests on CPU Torch.
Neither kind of evidence identifies the historical H100 cause. This document
claims no new GPU measurement.

## Source reading: safety contract

- The caller names a previously nonexistent `output_dir`; the parent must
  already exist. The harness refuses to reuse or overwrite that directory.
- Each rank writes only in its own `rankNNN` child. The selected total cap is
  divided across ranks, so the aggregate capture remains bounded. The total
  must be at least the distributed world size.
- The caller must explicitly acknowledge that a failure payload contains
  tokenized training rows, RNG state, and the pre-step trainable weights.
  Do not share the payload.
- The current optimizer window is retained in memory and written only when
  an observer sees a non-finite loss, incoming or accumulated gradient,
  selected LoRA tensor, or returned norm. A clean window writes only the
  small manifest/policy files.
- A payload is a Torch pickle (`.pt`) containing tensors and basic containers.
  A payload created by you can be inspected with `torch.load(path,
  weights_only=True)`. Never load a capsule supplied by somebody else.
- Installation fails rather than capturing after Soup's watchdog if the
  installed Trainer does not expose the mutable callback ordering required to
  put the forensic callback first.

The default cap is 2 GiB. Choose a fresh path and a larger explicit cap only
when the trainable LoRA state and one accumulated step require it. Before it
installs hooks, the harness forecasts the pre-step trainable-state snapshot:
when Soup will run FP16 with BF16 LoRA adapters, that forecast includes the
BF16-to-FP32 alignment performed by `wrapper.train()`. It refuses a rank whose
projected snapshot alone cannot fit within that rank's share; the serialized
payload, token rows, RNG state, and summary need additional space.

## Source reading: capture run

After the normal Soup `GRPOTrainerWrapper.setup(...)` call has initialized
`wrapper.trainer`, and before the wrapper calls `train()`, install the harness
explicitly:

```python
from benchmarks.harness.grpo_crash_capsule import Issue342CrashCapsule

capsule = Issue342CrashCapsule(
    wrapper.trainer,
    output_dir="/raid/issue342-capsule-001",  # must not exist yet
    max_total_bytes=2 << 30,
    acknowledge_training_rows=True,
).install()

try:
    with capsule.sdpa_backend("baseline"):
        wrapper.train()
finally:
    capsule.close()
```

`wrapper.train()` is intentional. Do not replace it with
`wrapper.trainer.train()`: the wrapper owns Soup's production pre-training
setup, including BF16-LoRA-to-FP32 alignment for FP16 GradScaler runs. The
harness only observes the already-created inner trainer.

Run each SDPA arm in a fresh process with the same committed Soup revision,
config, model checkpoint, data snapshot, and output-cap policy:

- `baseline` (the process's native SDPA policy; no backend switch is forced)
- `math`
- `mem_efficient`
- `flash`

An unavailable or failing backend is a result to record, not a reason to
silently fall back to another backend. The harness prefers
`torch.nn.attention.sdpa_kernel` whenever it exists and falls back to the
legacy `torch.backends.cuda.sdp_kernel` context manager only when necessary.

## Source reading: evidence written on an anomaly

The capsule records, per rank:

- environment stamp and selected cap;
- the requested SDPA arm plus the selected SDPA policy flags, cuDNN,
  determinism, TF32, and relevant CUDA-environment settings at the affected
  step; this does not identify the kernel the dispatcher actually executed;
- CPU/CUDA/Python RNG state at the beginning of the affected optimizer step;
- CPU copies of the token tensors passed to the loss for every accumulated
  microbatch in that step;
- loss statistics for every microbatch (`microbatch_loss_stats`, zero-based
  `microbatch_index`) and the first non-finite loss
  (`first_nonfinite_loss`). The existing `loss` field remains the latest
  loss, so it can be finite even when an earlier microbatch was non-finite;
- pre-step trainable state, first non-finite LoRA input/output or
  parameter-gradient name, and every parameter name observed non-finite by the
  backward hooks before clipping (`first_nonfinite_parameter` and
  `nonfinite_parameter_names_before_clip`); `post_clip_nonfinite_gradient_names`
  and `post_clip_pre_watchdog_gradient_stats` are sampled after `max_grad_norm`
  clipping but before the watchdog can clear them;
- the first observed non-finite accumulated parameter gradient
  (`first_nonfinite_accumulated_parameter`) and the names ever observed
  non-finite by post-accumulation hooks in that step
  (`nonfinite_accumulated_parameter_names_before_clip`). Incoming gradient
  contributions and accumulated `p.grad` are checked separately because
  finite contributions can overflow when added. Accumulated observations
  precede unscale and may precede distributed reduction. They do not establish
  the final unscaled gradients on another rank. The environment records the
  number of supported accumulated-gradient hooks. The observer uses
  [`torch.Tensor.register_post_accumulate_grad_hook`][post-hook]. Its source
  is explicit in `accumulated_gradient_source`. When the API is missing or
  non-callable, that parameter remains unobserved; its name appears in
  `accumulated_gradient_unavailable_parameter_names`. This also identifies
  partial coverage. If no observer is installed, the source and phase are
  `unavailable`. The source and unavailable names appear in the manifest,
  captured environment and summary. Phase and hook count are environment
  metadata; a zero hook count is not evidence of finite gradients;
- the actual value returned by this trainer's `accelerator.clip_grad_norm_`
  (`gradient_norm_observation` and `gradient_norm_source`). A non-finite
  observed norm triggers capture even if every post-clip gradient is finite.
  The method is called once with its original arguments and its return value
  is passed through. Original exceptions propagate. The method is restored
  on close and on failed installation. Supported observations are one-element
  real tensors and native Python numeric scalars representable as float64.
  Unsupported return forms create no observation. Non-finite values are distinguished as
  `inf`, `-inf` and `nan`.
  The manifest records whether this observer is available. Custom trainer
  paths that bypass this method leave the source as `unavailable`; their
  norm has not been measured by this observer;
- input/gradient-output bounds for LoRA-named modules and the resulting
  conditional FP32 contraction bound.

This can identify the first named bad parameter, show whether the assumed
Split-K operands were actually bounded, preserve a replay candidate, and
separate SDPA-policy arms. It also synchronizes at named capture boundaries and
writes a small error record if that surfaces an asynchronous CUDA failure. The
added synchronization changes timing, so it cannot prove that an observed
association is the original H100 mechanism.
The additional post-accumulation checks and norm statistics add host/device
synchronization too. A clean run under this instrumentation cannot exclude
a timing-sensitive failure in an uninstrumented run.

These additional fields extend the existing payload without renaming the
incoming-gradient or latest-loss fields. Step-end cleanup discards all
per-microbatch, accumulation and norm observations before the next step.

## CPU controls: synthetic assertions

```bash
python -m pytest tests/test_grpo_crash_capsule.py --no-cov -q
python -m ruff check benchmarks/harness/grpo_crash_capsule.py tests/test_grpo_crash_capsule.py
```

On CPU Torch `2.13.0+cpu`, the loss controls put a NaN constant on either of
two microbatches while the parameter gradients remain finite. They assert
the indexed history and preserve the latest-loss field's meaning. The
accumulation control uses two finite FP32 contributions of `2e38` and checks
the first bad accumulated parameter at microbatch 1. The norm control uses
sixteen finite FP32 entries of `1e20`: the returned norm is infinite, and
clipping at 1 leaves finite zeros. Capture must still occur. These are CPU
observations, not a measurement of a GPU kernel or the GRPO objective.

The tests also cover: lazy Torch import, explicit sensitive-data acknowledgement,
new-directory/no-overwrite containment, the per-rank cap, a weights-only
loadable post-clipping/pre-watchdog anomaly capsule with pre-clipping
attribution, no rows persisted by a finite or evaluation step, and both current
and legacy SDPA context-manager API surfaces, plus a simulated CUDA-sync failure
record. Additional controls assert unavailable and partial hook coverage,
unsupported norm returns, original method arguments/result identity and
exceptions, cleanup after late installation failure, step reset, signed
infinities/NaN, and zero or large finite float64-range norm values. They do
not exercise CUDA, distributed ranks, TRL, or a real LoRA model.

[#2667]: https://github.com/MakazhanAlpamys/Soup/issues/2667
[base]: https://github.com/MakazhanAlpamys/Soup/commit/ad36446383cecb0707303141987945757a1db19c
[post-hook]: https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.register_post_accumulate_grad_hook.html
