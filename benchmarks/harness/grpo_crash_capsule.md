# Issue #342 GRPO crash capsule

`grpo_crash_capsule.py` is an **opt-in forensic harness** for a real GPU
reproduction of [#342](https://github.com/MakazhanAlpamys/Soup/issues/342).
It is not imported by Soup's CLI or trainer code. It does not reproduce the
incident on CPU, assign an SDPA backend, or identify a root cause.

## Safety contract

- The caller names a previously nonexistent `output_dir`; the parent must
  already exist. The harness refuses to reuse or overwrite that directory.
- Each rank writes only in its own `rankNNN` child. The selected total cap is
  divided across ranks, so the aggregate capture remains bounded. The total
  must be at least the distributed world size.
- The caller must explicitly acknowledge that a failure payload contains
  tokenized training rows, RNG state, and the pre-step trainable weights.
  Do not share the payload.
- The harness retains the current optimizer step in memory and writes it only
  when it observes a non-finite gradient before Soup's watchdog. A finite step
  writes only the small manifest/policy files.
- A payload is a Torch pickle (`.pt`) containing tensors and basic containers.
  A payload created by you can be inspected with `torch.load(path,
  weights_only=True)`. Never load a capsule supplied by somebody else.
- Installation fails rather than capturing after Soup's watchdog if the
  installed Trainer does not expose the mutable callback ordering required to
  put the forensic callback first.

The default cap is 2 GiB. Choose a fresh path and a larger explicit cap only
when the trainable LoRA state and one accumulated step require it. Before it
installs hooks, the harness refuses a rank whose raw pre-step trainable-state
snapshot alone cannot fit within that rank's share; the serialized payload,
token rows, RNG state, and summary need additional space.

## Capture run

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
silently fall back to another backend. PyTorch 2.13 documents both
`torch.nn.attention.sdpa_kernel` and the legacy
`torch.backends.cuda.sdp_kernel` context manager. The harness prefers the
former whenever it exists and falls back to the latter only when the current
API is unavailable.

## Evidence written on an anomaly

The capsule records, per rank:

- environment stamp and selected cap;
- the requested SDPA arm plus the selected SDPA policy flags, cuDNN,
  determinism, TF32, and relevant CUDA-environment settings at the affected
  step; this does not identify the kernel the dispatcher actually executed;
- CPU/CUDA/Python RNG state at the beginning of the affected optimizer step;
- CPU copies of the token tensors passed to the loss for every accumulated
  microbatch in that step;
- pre-step trainable state, first non-finite LoRA input/output or
  parameter-gradient name, every parameter name observed non-finite by the
  backward hooks before clipping, and gradient statistics sampled after
  `max_grad_norm` clipping but before the watchdog can clear them;
- input/gradient-output bounds for LoRA-named modules and the resulting
  conditional FP32 contraction bound.

This can identify the first named bad parameter, show whether the assumed
Split-K operands were actually bounded, preserve a replay candidate, and
separate SDPA-policy arms. It also synchronizes at named capture boundaries and
writes a small error record if that surfaces an asynchronous CUDA failure. The
added synchronization changes timing, so it cannot prove that an observed
association is the original H100 mechanism.

## CPU checks

```bash
python -m pytest tests/test_grpo_crash_capsule.py --no-cov -q
python -m ruff check benchmarks/harness/grpo_crash_capsule.py tests/test_grpo_crash_capsule.py
```

The tests cover: lazy Torch import, explicit sensitive-data acknowledgement,
new-directory/no-overwrite containment, the per-rank cap, a weights-only
loadable post-clipping/pre-watchdog anomaly capsule with pre-clipping
attribution, no rows persisted by a finite or evaluation step, and both current
and legacy SDPA context-manager API surfaces, plus a simulated CUDA-sync failure
record. They do not exercise CUDA, distributed ranks, TRL, or a real LoRA model.
