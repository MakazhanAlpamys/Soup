# Issue #342 H100 non-finite-gradient workflow

This is a source audit and a set of standalone CPU controls. The historical
incident still lacks a captured failing batch. The single-T4 watchdog trip used
a different precision, workload, and dependency stack.

## What the incident record establishes

**Recorded benchmark evidence.** The six detailed STEP 27 rows show a finite
logged loss and a NaN gradient norm. STEP 27 ran separate training jobs, one per
card. A single H100 can test that topology. The same-card seed retry matched the
first reported reward mean and failed at a different point. That comparison
does not establish identical batches or model state at the failing step.
[STEP 27 benchmark and launcher record][step-27]

**Source reading (Soup v0.72.4).** The historical config defaults to 4-bit base
weights and four accumulation microbatches. The resolved benchmark config is
unavailable, so neither default establishes the settings of the failing run.
With no nonstandard variant selected, the GRPO wrapper uses the stock TRL
trainer. The later Soup watchdog and its parameter attribution are separate
from the original failure. [Tagged schema][soup-config] · [Tagged wrapper][soup-grpo]

## Rollout, loss, backward, and sampling

| Evidence type | Finding and limit |
|---|---|
| Source reading, TRL v0.26.2 | Prompt IDs and completion IDs are padded separately and concatenated. Completion-length metrics alone do not determine padded attention length. [Pinned trainer][trl] |
| Source reading, TRL v0.26.2 and Transformers v4.57.6 | Reward aggregation and token/rank reductions lose per-element information. Trainer can filter a non-finite loss for logging after backward has run. A finite logged scalar leaves the raw loss inputs and intermediates unresolved. [Pinned TRL trainer][trl] · [Pinned Trainer][trainer] |
| CPU probe, Torch 2.13.0+cpu | Separate padding gave a width of 192 with a longest real row of 188, then 193 with identical completion lengths and a one-token prompt change. The synthetic padded width does not reveal the incident width. |
| CPU probe, Torch 2.13.0+cpu | An explicit old-log-probability overflow yielded a finite clipped loss with a NaN derivative. In the detached on-policy branch the ratio was one, and a constructed KL exponential overflow gave an infinite loss. These scalar expressions do not identify the branch or operands of the incident. |
| CPU probe, Torch 2.13.0+cpu | A single bad gradient propagated through clipping; a separate norm-overflow control gave an infinite norm even with finite gradients. A modeled grad-to-None guard kept a fresh optimizer from updating either parameter. The actual first bad tensor and optimizer state must be measured. |
| CPU probe, Torch 2.13.0+cpu | The modeled loss filter reported zero for a NaN contribution at the reset boundary and a finite accumulator for a finite contribution followed by NaN. This does not establish that the historical log used that filter. |
| CPU probe, Torch 2.13.0+cpu | A reentrant checkpoint's first attention call ran without a query gradient requirement; backward recomputation did require it. A non-reentrant control required it in both calls. A finite CPU Math forward/backward does not establish CUDA dispatch. |
| CPU probe, Torch 2.13.0+cpu | Synthetic reward, denominator, and sampling cases distinguished missing rewards from infinities, a positive denominator from zero, and one forbidden token from an invalid probability row. The controls use synthetic operands and execute no Soup training. |
| CPU probe, Torch 2.13.0+cpu | Implicit causal GQA and explicitly masked expanded K/V gave matching FP64 Math results while K/V layout changed. The test does not show which layout or backend ran during the incident. |
| CPU probe, Torch 2.13.0+cpu | Meta tensors illustrated a packed storage span below 4 GiB and a gapped layout above it without allocating either span. This does not establish the strides or index width of any CUDA kernel used by the run. |

## Candidate cuDNN backward mechanism

**Source reading (PyTorch report #196678, v2.13.0 CUDA 13.0 build).**
The author reports “EFFICIENT_ATTENTION and MATH are clean on the same inputs”
while cuDNN returned NaN query gradients and a finite forward. The directed
input used low logsumexp and a key length of 64 modulo 128. [Primary report][torch-report]

**Source reading (NVIDIA cuDNN Frontend PR #628, head
`1fb3fc2c779e0209e65642dbd80b838f623441dc`).** NVIDIA describes how the
“TMA-zero-filled KV tail entered `exp(0 - LSE)` unmasked” at that remainder.
The PR reports a directed H100 NVL regression. It is evidence about the
vendor's test case. [Vendor regression description][nvidia-report]

**Arithmetic.** A zero tail score reconstructed as `exp(-LSE)` can exceed
FP32 range for `LSE < -ln(FLT_MAX) ≈ -88.72`. Multiplication by a padded zero
can then produce NaN. The CPU witness distinguishes a low diagonal score
from a low row logsumexp. The actual score rows and kernel in #342 are unknown.

**Source reading (PyTorch commit `c962c84527a8a907caaadc6bab9c84143d992845`).**
The newer dispatch guard is conservative. It rejects affected older-cuDNN
SM90 calls based on noncausal execution, query gradient requirement, and key
length remainder. It does not need to see a mask or low logsumexp to reject a
call. The guard is distinct from the directed reproducer's chosen inputs.
[PyTorch change][torch-fix]

## Recheck on a reproducing GPU

1. Record the actual Soup/model revisions, resolved config, CUDA driver and
   runtime cuDNN. At the first bad batch keep the pre-step adapter state,
   inputs, RNG, per-rank raw loss statistics, and pre-clip gradient boundaries.
2. Trace the policy attention call and any checkpoint recomputation. Record
   the actual ATen operator and CUDA kernel, padded Q/K/V lengths and strides,
   mask shape/type, causal flag, gradient requirement, and incoming gradient.
3. If the first bad boundary is SDPA, replay identical tensors and layout
   through cuDNN, Efficient, and Math in fresh processes. Record backend
   rejection separately from finite or non-finite results. Compare with the
   padded key-length remainder and row logsumexp from that call.
4. If the first bad boundary is upstream of SDPA or belongs to a GEMM, follow
   that boundary. A finite log line and the watchdog's later parameter name
   cannot resolve the first failing operation by themselves.

## Run the CPU controls

Run explicitly from the repository root with CPU PyTorch 2.13.0:

```bash
python benchmarks/harness/grpo_nonfinite_workflow_cpu.py
```

It prints JSON after its assertions pass. It uses synthetic tensors and makes
no Hub call, reads no training rows, writes no capsule, and runs no CUDA
operation. It does not reproduce #342 or identify its backend or root cause.

[step-27]: https://github.com/MakazhanAlpamys/Soup/blob/dcb37d9d47062a2a5e97a7fd538151a36d578b8e/benchmarks/gate-h100-validation.md#L4466-L4519
[soup-config]: https://github.com/MakazhanAlpamys/Soup/blob/v0.72.4/src/soup_cli/config/schema.py#L900-L925
[soup-grpo]: https://github.com/MakazhanAlpamys/Soup/blob/v0.72.4/src/soup_cli/trainer/grpo.py
[trl]: https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_trainer.py
[trainer]: https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer.py
[torch-report]: https://github.com/pytorch/pytorch/issues/196678
[nvidia-report]: https://github.com/NVIDIA/cudnn-frontend/pull/628
[torch-fix]: https://github.com/pytorch/pytorch/commit/c962c84527a8a907caaadc6bab9c84143d992845
