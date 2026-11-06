# Gate protocol: layer streaming against an independently configured DeepSpeed, 1x H100 (pre-registered)

Written before any run. Frozen by its sha256 (recorded with the run). Any change after the first timed
run is listed in the record as a deviation, with the reason; no threshold moves after data exists.

## 0. Why this exists

`benchmarks/gate-h100-validation.md` STEP 3 reports layer streaming at **2.93x** the throughput of
DeepSpeed ZeRO-3 with CPU parameter offload (bf16 vs bf16, 63.52 vs 21.65 tok/s, peak 3,935 vs 38,135
MiB), one H100, Llama-3.1-8B-Instruct, LoRA r=8, batch 1, 256 steps. Three things in that comparison are
open and this run closes them:

1. **Replication on current code.** STEP 3 ran Soup `2c9a078` (2026-08-06). Its run scripts and its
   64-row dataset were never committed. This run commits everything it uses.
2. **What each optimization contributes** to the streamed number (ablation).
3. **DeepSpeed as an independent engineer would set it up**: STEP 3 could not build `cpu_adam`
   (no `nvcc` on that box), so optimizer offload was never measured, and DeepSpeed was reached only
   through Soup's own `--deepspeed` flag. Here DeepSpeed and the resident reference run in a plain
   Hugging Face Trainer + PEFT script with no Soup code in the path, configured from the DeepSpeed
   documentation, so the comparison does not depend on Soup's wrapper.

If the ratio comes out lower than 2.93x, the record says so in its first paragraph.

## 1. Fixed setup (every arm)

- Hardware: 1x H100 80 GB (SXM or PCIe, recorded), rented on vast.ai. Recorded: GPU name, driver, CUDA,
  SM clock while busy (median), CPU model and cores, host RAM, `ulimit -l`, disk type.
- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct` at revision `d10aef7999a2` (the ungated copy the
  earlier records used; no token on the rented box). Its four `*.safetensors` shards match
  `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a1` by the LFS sha256 the Hub lists (checked
  before the run). The record gives both revisions, the shard hashes, and the local sha256 of every
  downloaded file; `config.json`, `tokenizer.json`, `tokenizer_config.json` and
  `generation_config.json` (not LFS) are compared with the meta-llama revision by content.
- Precision: bf16 base weights in every arm (no NF4 in any compared arm).
- LoRA: r=8, alpha=16, dropout 0.05, target modules = Soup's default for Llama (recorded from the
  resolved config), the same list in the HF script.
- Optimizer: AdamW, lr 2e-4, linear warmup 0, weight decay 0 (recorded); in DeepSpeed arms where
  DeepSpeed must own the optimizer and scheduler (torch strict-zip issue seen in STEP 3), its AdamW /
  WarmupLR with the same values.
- Batch 1, gradient accumulation 1, seed 42, max_length 256.
- Software: current `main` of soup-cli at a recorded commit; torch, transformers, peft, accelerate,
  deepspeed versions recorded; DeepSpeed built with `DS_BUILD_CPU_ADAM=1` (the build log is kept).
- Two workloads:
  - **Shape A (the STEP 3 shape):** 64 rows, 4 epochs = 256 optimizer steps, short rows: the committed
    file `ds_compare_rows_a.jsonl` holds 64 alpaca-format rows whose tokenized length targets ~32 tokens
    per step (STEP 3's own numbers give tok/s x runtime = 8,192 tokens over 256 steps). The exact token
    count per step is measured and recorded; it will not match STEP 3's unknown file exactly.
  - **Shape B (a realistic row):** 64 rows of 512 tokens each (`ds_compare_rows_b.jsonl`,
    deterministic text, max_length 512), 2 epochs = 128 steps.
  - **Shape C (a long row):** 64 rows of 2,048 tokens each (`ds_compare_rows_c.jsonl`, deterministic
    text, max_length 2048), 1 epoch = 64 steps.
  Shape A answers "does the earlier number hold"; B and C answer "does it hold at the lengths people
  train on", where the per-step transfer cost is shared by many more tokens.

## 2. Arms

Every arm is one fresh process. Streaming arms use `soup train` with a generated `soup.yaml`. DeepSpeed and
resident arms use `hf_baseline.py` (plain HF Trainer + PEFT, no Soup import).

### Streaming (Soup)

| id | what | config difference from S1 |
|---|---|---|
| S0 | STEP 3 code: Soup at `2c9a078`, same yaml as S1 where the keys exist | separate venv at that commit |
| S1 | current main, defaults: `stream_layers: true`, `stream_recompute: on`, `stream_buffers: 2`, `stream_pin` automatic, no loss chunk, `fast_lora: false`, `expandable_segments` exported by Soup | - |
| S2 | S1 + `stream_recompute: auto` | one key |
| S3 | S1 + `stream_recompute: off` | one key |
| S4 | S1 + `stream_pin: false` (leave-one-out: pinning) | one key |
| S5 | S1 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` set by the harness (leave-one-out: allocator hint) | env |
| S6 | S1 + `stream_buffers: 3` | one key |
| S7 | S1 + `stream_loss_chunk: 128` | one key |
| SB | best streaming config: S1 plus every single change from S2..S7 that was faster than S1 by more than the S1 spread, combined (decided from the Shape A data by the rule, then run) | derived |

`fast_lora` is not an arm of the comparison: it needs `lora.dropout: 0`, which changes the training
numerics. It is run as an informational arm only (S8 = S1 + `fast_lora: true` + `lora.dropout: 0`,
3 repeats) and never enters a ratio or SB.

The S0 -> S1 difference is the sum of the code-level changes since STEP 3 that have no switch (pinned
arena packing, untied embedding/head streaming, head and embedding prefetch, per-call overhead removals,
allocator hint; list with commits in `bf16-path.md`). They cannot be separated one by one without
building old revisions; the record says so.

### DeepSpeed and resident (HF Trainer + PEFT, no Soup)

| id | what |
|---|---|
| D1 | ZeRO-3, `offload_param: cpu` (pin_memory true), `offload_optimizer: none` - the STEP 3 setting, as a JSON |
| D2 | ZeRO-3, `offload_param: cpu` + `offload_optimizer: cpu` (pin_memory true, `cpu_adam` built) |
| D3 | ZeRO-2, `offload_optimizer: cpu` (pin_memory true), parameters on the GPU |
| D4 | ZeRO-3 with no offload (single GPU: parameters on the GPU) |
| R1 | resident bf16 LoRA, no DeepSpeed, no gradient checkpointing |
| R2 | R1 + gradient checkpointing (HF `gradient_checkpointing=True`) |

DeepSpeed tuning, fixed before the runs from the DeepSpeed configuration documentation: `overlap_comm:
true`, `contiguous_gradients: true`, `reduce_bucket_size: "auto"`, `stage3_prefetch_bucket_size:
"auto"`, `stage3_param_persistence_threshold: "auto"`, `stage3_max_live_parameters: 1e9`,
`stage3_max_reuse_distance: 1e9`, `sub_group_size: 1e9`, `pin_memory: true` on every offload; bf16
enabled. One extra D1/D2 variant each with `stage3_prefetch_bucket_size: 5e8` is allowed if the auto run
shows the GPU idle more than 50% of the time (recorded as a deviation-free option because it is decided by
this rule, not by the result).

### VRAM budget runs (Shape A)

A hard cap through `torch.cuda.set_per_process_memory_fraction`, applied by a `sitecustomize.py` that the
harness puts on `PYTHONPATH` for every arm alike (`SOUP_BENCH_VRAM_CAP_GB`): caps **24 GB** and **16 GB**.
Arms under each cap: SB, D1, D2, D3, R1; 2 repeats at 24 GB, 1 repeat at 16 GB (an out-of-memory run
is not repeated). An arm that runs out of memory is recorded as "does not fit at this cap", not as a
failure.

### Shape B and Shape C runs (80 GB)

Arms SB, S1, D-best (the fastest DeepSpeed arm of Shape A at 80 GB), D1, R1; 2 repeats each per shape.
If a streaming arm runs out of memory at Shape C (activations at 2,048 tokens), it is rerun once with
`stream_recompute: on` and that is stated.

## 3. Measurements (each run)

- `train_runtime`, `train_samples_per_second` and the number of trained tokens as the trainer reports
  them; **tok/s = trained tokens / train_runtime** (the STEP 3 metric).
- Steady-state tok/s: tokens / time over steps 33..end (per-step timestamps from a callback), to remove
  warm-up and first-touch costs.
- Peak VRAM: `nvidia-smi` sampled every 0.5 s (STEP 3's method) and `torch.cuda.max_memory_reserved`.
- Peak host RSS of the training process (psutil, 1 Hz) and pinned bytes where Soup reports them.
- Mean GPU utilization and median SM clock while busy (`nvidia-smi`).
- Final training loss and the loss at steps 64/128/192/256 (sanity: all arms train the same task).
- Wall time of the whole process and the rental cost of the run (from the vast.ai rate).

## 4. Repeats and order

- 3 timed repeats per arm at 80 GB on Shape A; under the caps as in section 2; 2 repeats per arm on Shapes B and C.
- One untimed warm-up run of S1 and of D1 after setup (fills the page cache and the kernel caches);
  not counted.
- Order: interleaved by repeat (all arms rep 1, then all arms rep 2, ...), so a slow drift of the box
  touches every arm alike.
- The value of an arm = the median of its repeats; its spread = (max - min) / median.

## 5. Void conditions (checked per run, before any ratio)

A run is void and is repeated once if: another process is on the GPU; the median busy SM clock is more
than 5% below the session median; the run exits non-zero for a reason other than out-of-memory under a
cap; or its final loss differs from the arm's other repeats by more than 20%. An arm with two void runs
is reported as not measured.

## 6. Gates and how they are read (computed as written)

- **G1 - replication of the STEP 3 headline on current code.** Ratio = median tok/s(S1) / median
  tok/s(D1), Shape A, 80 GB. **CONFIRMED** if >= 2.64 (2.93 - 10%); **LOWER** otherwise, stated with the
  measured value. Also reported: S0/D1 (the old code on this box).
- **G2 - against the best independent DeepSpeed.** Ratio = median tok/s(SB) / max median tok/s over the
  DeepSpeed arms D1..D4 that fit, per budget (80, 24, 16 GB). No pass mark: these are the numbers to quote,
  each with its budget, its peak VRAM pair and its spread. At a budget where no DeepSpeed arm fits, the
  record says "DeepSpeed does not fit at N GB" instead of a ratio.
- **G3 - resident reference, stated plainly.** Ratio = median tok/s(R1) / median tok/s(SB) at 80 GB,
  and the same with R2. Expected above 1 (an 8B bf16 model fits an 80 GB card, and nothing beats
  training it resident there); the record states it whichever way it comes out, together with the peak
  VRAM of each, because the case for streaming is the VRAM it needs, not speed on a card the model fits.
- **G4 - realistic lengths (separate verdict).** For Shape B (512 tokens) and Shape C (2,048 tokens),
  each at 80 GB: ratio median tok/s(SB) / median tok/s(D-best), ratio SB / D1, and ratio R1 / SB, each
  with both peak VRAM values. Read: **HOLDS** at a shape if SB / D-best >= 2.0; **SMALLER** if
  1.0 <= SB / D-best < 2.0; **REVERSED** if SB / D-best < 1.0. The record quotes the Shape A headline
  only next to the G4 values.
- **Ablation table.** For each S2..S7: change of median tok/s against S1 in %, with S1's spread; a change
  inside S1's spread is reported as "no measurable effect".
- Near-miss rule: a gate value within 2% of its threshold is reported as such and not rounded either way.

## 7. What is committed with the record

The protocol (this file, sha256), the harness (`run_all.sh`, `hf_baseline.py`, the yaml generator, the
DeepSpeed JSONs, `sitecustomize.py`, the two dataset files and their generator), the raw JSON of every
run, the nvidia-smi samples, the build log of `cpu_adam`, the environment listing, and the cost log. One
command reproduces the whole series on another H100: `bash run_all.sh --all`.

## 8. Budget and stop rule

Rental budget USD 30 and at most 8 hours of H100 time, through the self-stopping vast kit only. Planned
GPU time about 5.2 h plus about 1 h of setup. If the budget or the deadline would be exceeded, work is
dropped in this order: S8, the prefetch variants, the 16 GB cap, the second repeat of Shape B, the
second repeat of Shape C, the 24 GB cap; never G1's arms (S1, D1 at Shape A) and never the first repeat
of Shapes B and C.
