# Token-chunked distillation: real-model H200 measurement (#722)

This is the measurement half of [#722](https://github.com/MakazhanAlpamys/Soup/issues/722).
The chunked/checkpointed kernel was implemented by @kok-o and merged in
[#743](https://github.com/MakazhanAlpamys/Soup/pull/743). No training implementation
or default is changed by this record.

## Protocol

- Soup runtime: `26ffea17461d0d14a0d3641b93435343b54c29e1`.
- Student: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Frozen teacher: `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Identical token-ID vocabularies, verified before loading; V=151936.
- B=2, realized S=2048 (asserted after Soup's formatter), 1876 supervised causal
  targets per row, no padding. The deterministic repeated telescope explanation
  is a throughput fixture, not a held-out quality dataset. The formatter's
  actual tokenized-data hash is retained in every raw row.
- Native BF16 frozen models, FP32 LoRA parameters, BF16 autocast; LoRA r=8,
  alpha=16, dropout=0, q/v projections. AdamW, LR=2e-5, constant scheduler,
  accumulation=1, seed/data_seed=722, model gradient checkpointing on in both
  arms, non-reentrant, use_cache=false. Teacher parameters remain frozen.
- Shipped dense default: `distill_chunk_size: null`, `distill_checkpoint: false`.
  Treatment: chunk=128, checkpoint=true. Forward KL, T=2 in both arms. Actual
  Soup Trainer loss is 0.5 CE + 0.5 KD.
- Each run constructs `DistillTrainerWrapper.setup` and uses its actual
  `Trainer.train`: student/teacher forward, masked CE/KD, backward, clipping,
  AdamW and scheduler. Three warm-up updates precede eight measured updates.
  Each arm starts in a fresh process with the same adapter initialization.
- Eight runs in order D-C-C-D-C-D-D-C: four adjacent matched pairs with both
  execution orders. Both arms use the same physical card in one session.

The two 2-warm-up/2-measured-step pilots were excluded from the final summary.
After those pilots, the recipe, order and numerical tolerances were written to
`protocol.json` before the eight measured runs. Pilots, their exclusion and the
tolerances are retained, rather than presented as a preregistered blind study.

## Measurement boundaries

Step time uses `perf_counter` between Trainer's `on_step_begin` and
`on_step_end`, with CUDA synchronization at both boundaries. It includes the
optimizer and repeated training work, but excludes the preceding data-loader
fetch, model/tokenizer setup, downloads and final adapter serialization. It is
**not end-to-end `soup train` wall time**. Setup and Trainer-total seconds are
also recorded. There is no intermediate checkpoint saving in this short run.

Peak is `torch.cuda.max_memory_allocated()` for this process, reset at each
step, including resident model weights, live activations and optimizer state.
The reported total peak includes warm-up; steady peak and reserved bytes are
reported separately. The first warm-up step additionally copies gradients to
CPU for comparison, so its time is deliberately excluded. Captured gradient
copies do not remain on CUDA. No empty-cache calls or allocator changes are
made during training. Token/s counts input tokens; supervised token/s is a
separate raw field.

## Hardware limitation

NVIDIA H200, 143771 MiB reported by nvidia-smi; physical GPU 2, selected by UUID
`GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11`. Another user's idle process held
49850 MiB before these runs. Its allocation was left untouched. Whole-device
utilization/clock/power/temperature samples are retained, but **whole-device
memory is not attributed to this benchmark**. This is a shared-card measurement;
it is not a dedicated-device throughput ceiling. Four CPU threads were used.

The runtime, stack versions, exact allocated bytes, all per-step durations,
losses, initial-adapter/data/source hashes and sampled clocks live in
[`evidence/distill-722-h200/`](evidence/distill-722-h200/).

## Results

The table and numerical/control observations below are filled from the complete
eight-run record, without selecting the fastest repetition.

| Arm | Mean step seconds, median of 4 runs (range) | Input tokens/s, median (range) | Total allocated peak |
|---|---:|---:|---:|
| Dense | 0.2124 (0.2095–0.2201) | 19288 (18610–19554) | 31.656 GiB |
| Chunk 128 + checkpoint | 0.2820 (0.2813–0.2823) | 14525 (14508–14562) | 21.400 GiB |

At this operating point checkpointed chunks reduce total allocated peak by **32.4%** (1.479x less memory), while mean step time is **32.8% longer** by the ratio of medians. The four adjacent pairs are **28.0%, 30.9%, 34.5% and 34.8% longer**, a **28.0–34.8%** paired range. In run order the dense step times fell from 0.2201 through 0.2149 and 0.2099 to 0.2095 s, while chunked times were 0.2817, 0.2813, 0.2823 and 0.2823 s. These are observations of the recorded runs; no cause is assigned to the trend. This is a memory/time tradeoff, not a training speedup. All eight runs completed; every repetition of an arm had the same peak byte count.

All four pairs passed the frozen short-run numerical gates: first-step relative loss error <=1e-5, first-step gradient relative L2 <=1%, and maximum relative loss drift across 11 updates <=1%. Observed maxima: first loss 0, first-gradient relative L2 0.004448, trajectory loss 0.295%. Final adapter differences are recorded as descriptive values, not a held-out quality gate.

BF16 training is not bit-reproducible even within one arm on this stack. Same-arm controls show up to 0.431% first-gradient relative L2 and 0.339% trajectory loss difference. These controls distinguish ordinary repeat variability from the dense/chunked comparison; they do not prove identical final quality.

All sampled SM clocks were 1980 MHz. The two arms use the same dependency versions; see the raw runtime/dtype fields. The existing 138 kernel/config tests plus 18 comparison-guard tests pass (156 total); whole-tree Ruff passes.

These measurements compare the current dense default to the already-shipped
opt-in controls. They do not compare against the pre-#736 low-precision kernel,
and do not claim an improvement in held-out quality. Chunking does not remove
the full model logits or the FP32 copies before token selection, so the
prototype's retained-autograd-byte ratio is not a prediction of total VRAM.

## Reproduce

Install the checkout's `[train,dev]` dependencies. Select one sufficiently large
CUDA GPU, and run from the checkout root. The student/teacher downloads and all
setup are outside the steady-step window. The raw records contain the exact
versions used here; changing them is a new stack, not a repeat of this row.

```bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python - <<'PY'
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

student = snapshot_download(
    'Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775')
teacher = snapshot_download(
    'Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
root = Path('/tmp/soup-722-h200-replay')  # Must not contain prior run directories.
paths = []
for index, arm in enumerate(('dense', 'chunked', 'chunked', 'dense',
                             'chunked', 'dense', 'dense', 'chunked'), 1):
    output = root / f'run-{index:02d}-{arm}'
    paths.append(str(output))
    subprocess.run([
        sys.executable, 'benchmarks/harness/issue722_distill.py',
        '--student', student, '--teacher', teacher, '--arm', arm,
        '--output', str(output), '--warmup', '3', '--steps', '8',
    ], check=True)
subprocess.run([
    sys.executable, 'benchmarks/harness/issue722_compare.py', *paths,
    '--output', str(root / 'summary.json'),
], check=True)
PY
```

The comparator refuses mismatched model snapshots, source/harness/data/adapter
hashes, shapes, dtypes, step budgets and other training settings. Gradient and
adapter NPZ files are produced locally by the harness; the repository includes
their hashes and derived comparisons, rather than checking in repeated binary
weights. Existing distillation numerical/mask/checkpoint tests and the new
comparison guards are exercised with:

```bash
pytest tests/test_issue722_chunked_distill.py tests/test_distill_chunked_benchmark_protocol.py --no-cov
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```
