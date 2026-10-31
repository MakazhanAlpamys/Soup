# Sharing FP32 normalization in the chunked head loss

Measured 2026-10-31 on macOS 26.6.2 arm64, Python 3.11.8, torch 2.15.0,
transformers 5.22.1, one CPU thread, FP32. Reference: `_chunk_terms` at
`7fd7df5`, copied into the [harness](harness/stream_loss_normalization.py).

## Change and scope

FP32 head outputs previously underwent two vocabulary normalizations: one for
entropy/rewind metrics and another inside cross-entropy. The new path computes
`log_softmax` once, uses NLL for the differentiable loss and detached values for
metrics. BF16/FP16 still use native-dtype metrics plus FP32 cross-entropy. No
mask, denominator, logging frequency or chunk size changes.

This compares **old chunked vs new chunked** loss, not chunked vs unchunked.
It measures full forward/backward/AdamW steps in a tiny resident Llama with the
production `ChunkedHeadLoss` hook. It is not a disk-streaming, CUDA, MPS, LoRA,
pretrained-model quality or memory benchmark. Speed depends on the share of
training time spent in the FP32 head/loss; no speedup is claimed for BF16/FP16.

## Protocol

- Randomly initialized two-layer Llama: hidden 64, intermediate 128, four
  attention heads, two KV heads, vocabulary 8192; frozen output head.
- Batch 2, sequence 128, chunk 64; AdamW learning rate 0.003; deterministic
  algorithms; all other model parameters trainable.
- Synthetic modular sequences, prompt mask in one row and padding mask in the
  other. Training starts in offsets 0–47; evaluation starts in 48–63, so complete
  input sequences do not overlap. This is a narrow synthetic learning task.
- Seeds 1234–1239, alternating old/new order. Each arm has ten warmup training
  steps followed by 300 timed training steps. Evaluation uses eight separate
  batches after training, via the ordinary unchunked model forward.
- Timing includes zero-grad, model forward, backward and optimizer update;
  excludes setup, eval and metric serialization. No concurrent test suite was
  running. This desktop was not isolated from other OS activity.
- Compare the complete training-loss curve, per-step entropy/per-token loss/
  correct counts, first-step gradients, final model state, eval loss and accuracy.
  The harness exits unsuccessfully if a delta is nonfinite or exceeds 1e-6.
  Wall-clock timing is evidence, not a CI assertion.

## Results

The median ratio of old/new total timed training duration was **1.1605×**,
corresponding to **13.83% less timed training duration**. Individual ratios
ranged from 1.027× to 1.283×. This is descriptive timing from one host, with
visible scheduling noise; it is not a universal performance guarantee.

| Seed | Old median step, ms | New median step, ms | Total-duration speedup | Held-out loss, both | Held-out accuracy, both |
|---|---:|---:|---:|---:|---:|
| 1234 | 19.190 | 16.350 | 1.283× | 6.438384 | 97.8488% |
| 1235 | 19.523 | 16.536 | 1.075× | 6.476580 | 98.7209% |
| 1236 | 19.150 | 16.686 | 1.027× | 6.462359 | 98.6628% |
| 1237 | 19.519 | 16.862 | 1.269× | 6.522264 | 98.3721% |
| 1238 | 19.138 | 16.387 | 1.246× | 6.421858 | 98.6628% |
| 1239 | 19.099 | 16.497 | 1.066× | 6.539859 | 98.3140% |

All measured deltas were **exactly zero**, including final weights and the
complete training-loss curve. Thus no degradation was observed on this test.
This does not establish quality on natural-language tasks or other hardware.
The high argmax accuracy with relatively high NLL reflects the frozen random
head and the narrow synthetic task; it is not a language benchmark score.

[Full raw results](results/stream-loss-normalization/cpu-fp32.json) include both
arms, environment, configuration and all parity deltas. Step medians and the
ratio of total durations are intentionally distinct statistics.

## Preliminary runs retained

Before the final disjoint-offset protocol, evaluation used independent draws
from the same set of possible starting offsets, so complete sequences could
repeat between training and evaluation. These runs are retained as exploratory
timing/parity evidence, not held-out quality evidence:

- [Three seeds, 60 timed steps](results/stream-loss-normalization/pilot-60-steps.json):
  median 1.1593×, but one pair regressed to 0.6054× under timing noise. This prompted
  longer runs; no negative pair was removed from the reported run.
- [Six seeds, 300 timed steps](results/stream-loss-normalization/pilot-300-steps.json):
  median 1.1318×, range 0.9875–1.3146×. All compared numerical deltas were zero.

## Reproduce

```bash
pip install -e ".[dev]"
PYTHONPATH=src python benchmarks/harness/stream_loss_normalization.py \
  --steps 300 --repeats 6 --output /tmp/soup-loss-ab.json
pytest tests/test_stream_loss_chunk.py --no-cov -q
```

The new profiler regression test fails on the reference implementation because
it observes two `aten::_log_softmax` calls per FP32 chunk instead of one. The
existing module tests cover FP32/BF16 loss and gradient parity, masks, denominator
handling, real SFT training, metric logging, rewind and gradient accumulation.

## Repository validation on this host

`ruff check src/soup_cli/ scripts/ tests/ benchmarks/` passed, as did all 124
cases in `tests/test_stream_loss_chunk.py`. The broad non-smoke run measured
87.19% coverage (required floor: 77%). It was **not one clean all-green run**:

- Initial full run: 47,715 passed, 585 skipped, 624 failed, 116 setup errors.
  The chosen `/private/tmp` basetemp was outside the application's accepted
  temporary root. Other failures involved sandbox-denied cache writes, sockets,
  multiprocessing and nested `sandbox-exec`.
- After correcting the temporary/cache paths, 675 of those 740 cases passed.
- Of the remaining 65, 64 passed outside the execution sandbox. A local fixture
  redirected only the advise cache and registry paths for three legacy CLI tests
  so they could not delete or modify this workstation's real user files.
- Latest result per case: **48,454 passed, 585 skipped, one failure**.
  `tests/test_doctor.py::test_doctor_nccl_mocked_success` fails with
  `torch._C` missing `_cuda_getCurrentRawStream`. The identical failure was
  reproduced separately against an unmodified export of `main` at `7fd7df5`.
  It is unrelated to the loss change; no doctor/NCCL code or test was changed.

The local stack differs from the repository's pinned CI stack. These results
are local evidence, not a substitute for the project's Linux/Windows/macOS CI.
