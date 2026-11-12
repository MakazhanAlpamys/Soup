<!--
Measurement record, assembled 2026-11-08 from preserved result files and extended on 2026-11-11. Three
kinds of numbers appear and are never mixed:
  EARLIER RECORDED = read from result files written when the run was made (laptop 2026-11-03; cloud
    2026-11-05 .. 2026-11-07; the Banking77 run 2026-11-07).
  RE-RUN = measured again on the same laptop on 2026-11-07 with the same experiment runner (Qwen3-32B
    only).
  SOUP-TRAIN RUN = measured on the same laptop on 2026-11-11 through `soup train` (section 13): a
    separate measurement, not a re-run of the runner behind the other laptop numbers.
Mistral numbers from earlier recorded runs, not re-run with the runner that produced them. The re-check of
the Mistral laptop numbers with that runner has not been run; the only Mistral laptop measurement made
since is the separate `soup train` one in section 13.

Laptop: NVIDIA GeForce RTX 5070 Laptop GPU 8151 MiB, driver 616.92; Intel i9-14900HX (24 cores / 32
  logical); 31.73 GiB visible RAM; two Samsung MZAL81T0HFLB-00BL2 NVMe SSDs; native Windows.
Cloud: one NVIDIA L40S per run, PyTorch allocator capped at 6,710,886,400 B (6.25 GiB).
Stack (laptop runs of 2026-11-03 and 2026-11-07, and the Banking77 cloud run): Python 3.12.10 (laptop),
  torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0, accelerate 1.15.0, bitsandbytes 0.50.2, safetensors
  0.8.0. The versions of the cloud Mistral runs are not in the copied files. For the 2026-11-11 runs the
  operator reports Python 3.12.10, torch 2.14.0+cu130 and soup_cli 0.77.2; the logs do not print versions.
Models: unsloth/Qwen3-32B-bnb-4bit @ 7f721e74a6a8cc9ee352f7e49303a2c1705f9083;
  unsloth/Mistral-Large-Instruct-2407-bnb-4bit @ a76e109f4b66807d88fa92b5d7e0dead1a354494.
Soup: laptop runs import a frozen export of commit f215941b23a5cddd2f2e1338b6956ae828461645 (2026-10-15);
  the Banking77 run uses Soup main at 74e6628fc27454a6d82502b2d8b2e68b6bf159da; the cloud Mistral runs
  use no Soup code; the 2026-11-11 laptop runs use `soup train` on a local integration tree at commit
  ee4cc5d208127bad7c50c4228516a033f36725dd (section 13).
Raw output: benchmarks/results/big-models-8gb-laptop/ (path-scrubbed copies, section 12).
-->

# Run record: LoRA on Qwen3-32B and Mistral Large (122.6 B) through layer streaming on an 8 GB laptop GPU, a same-laptop re-run, a separate same-laptop measurement through `soup train`, and a 300-step Banking77 fine-tune

**Status: Mistral numbers from earlier recorded runs, not re-run with the runner that produced them ("that
runner" below: the experiment runners of sections 5 and 6); section 13 is a separate measurement through
`soup train`.** Capacity was measured once per configuration. One Qwen3-32B result was re-run with that
runner on the same laptop on 2026-11-07: the step losses, the held-out losses and the adapter file hash were
identical to every printed digit, the torch peaks and the sampled peak were equal, and one file of the
adapter folder differs (n = 1); the Qwen3-32B reload gave no result then and passed in one reload-only run
with that runner on 2026-11-11. On 2026-11-11 the same laptop also ran Qwen3-32B and Mistral Large (128 and
512 tokens) through `soup train`, one run each, with different code, data and sampler from the earlier runs:
all three completed, with whole-device peaks of 3,042, 4,388 and 6,998 MiB (3,002, 4,598 and 5,602 MiB in
the earlier runs); the +1,396 MiB at 512 tokens is in the run's own torch peak (about 6.2 GiB against 4.63
GiB) and its cause is not determined; and the saved adapters reload identically across fresh processes, but
the reloaded model's logits are NOT bit-identical to the training process's own logits (70.5% to 84.3% of
the elements differ after the cast, largest absolute difference 1.31; cause not determined) (section 13).
This is a capacity record. It is not a throughput benchmark and it makes no model-quality claim for
Mistral Large.

- **Earlier recorded, laptop (8 GB RTX 5070 Laptop).** Three rank-8 q/v LoRA updates ran through every
  decoder layer of **Qwen3-32B** NF4 at 1 x 128 tokens (sampled whole-device peak **3,002 MiB**) and of
  **Mistral-Large-Instruct-2407** NF4, 122,610,069,504 parameters, at 1 x 128 tokens (**4,598 MiB**) and
  1 x 512 tokens (**5,602 MiB**). Each saved adapter was reloaded in a different process and reproduced the
  trained held-out losses exactly: a 128-value logits slice for each of 2 examples (256 values) for the 32B,
  the complete BF16 logits tensors for the 123B (section 9). **Mistral numbers from earlier recorded runs, not re-run with that
  runner.**
- **Re-run on the same laptop, 2026-11-07, Qwen3-32B only.** The train step **gave the same step losses,
  held-out losses and `adapter_model.safetensors` SHA-256 to every printed digit, one run (n = 1, not the
  three that were planned)**, and the same 3,002 MiB sampled peak and torch peaks; one file of the adapter
  folder, `stage32_provenance.json`, differs (section 6). The
  step time was **2.37x slower** (median 22.475 s against 9.474 s) and **that difference is unexplained**.
  The **reload in a fresh process was not reproduced** on that day: six attempts produced no result, none
  because of a mismatch (section 6). A single reload-only run with the same runner on 2026-11-11 passed
  (section 13.2).
- **Mistral numbers from earlier recorded runs, not re-run with that runner.** The Mistral weights on the
  laptop were re-hashed on 2026-11-07 and equal the recorded values
  (`RF/rerun-2026-11-07/weights-hash-checks.json`); nothing else about Mistral was re-measured with that
  runner. The Mistral Large runs through `soup train` on 2026-11-11 are a separate measurement (section 13).
- **Earlier recorded, cloud L40S, Mistral Large, no Soup code.** One full-depth update at **8,192 tokens**
  (sampled **4,323 MiB**) and one at **16,384 tokens** (sampled **6,597 MiB**, a 33 GiB activation spill to
  disk, 3,618 s of worker time), each under a 6.25 GiB PyTorch allocator cap, and the saved optimizer state
  was restored exactly in a fresh process. The 16K step is not a laptop result (section 7). Mistral numbers
  from earlier recorded runs, not re-run.
- **No Mistral quality result.** The only pre-registered quality gate (4 updates at 8,192 tokens) ended
  **22/32 against 22/32** for its control and was not met (section 7.1). Mistral numbers from earlier
  recorded runs, not re-run.
- **Qwen3-32B, 300-step Banking77 fine-tune through `soup train`** (earlier recorded, 2026-11-07, one seed):
  accuracy on the full official test split rose from **69.58% to 82.82%** (n = 3,080); MMLU (500 items)
  went 76.4% to 77.2% and GSM8K (400 items) 95.25% to 95.75%, so neither fell. The training process held
  **5,052 MiB** of PyTorch memory against **20,127 MiB** for a resident control that reached 83.18%. It
  trained on a cloud L40S under the 6.25 GiB cap, **not on the laptop** (section 8).

Unit convention: MiB = 2^20 B and GiB = 2^30 B, as `nvidia-smi` and PyTorch report them; byte counts are as
recorded. "Sampled" means the largest value of a periodic `nvidia-smi` reading of the whole device, which
can miss a short peak. "pp" is an absolute difference in accuracy, in percent. `RF` below stands for
`benchmarks/results/big-models-8gb-laptop/`; every source file named in this record is under it.

---

## 1. Provenance: which numbers were re-run

| label | what | machine | date | for this record |
|---|---|---|---|---|
| EARLIER-LAPTOP | Qwen3-32B and Mistral Large: train 3 updates, then reload in a fresh process (Q32, M128, M512) | RTX 5070 Laptop 8 GB | 2026-11-03 | read from result files; **only the Q32 train step was re-run, with that runner** |
| RE-RUN | Qwen3-32B train step; six reload attempts; hash checks of the Qwen3-32B and Mistral weights | the same laptop | 2026-11-07 | measured again; the reload gave no result; Mistral **not** re-run with that runner |
| SOUP-TRAIN RUN | Qwen3-32B (128 tokens) and Mistral Large (128 and 512 tokens) through `soup train`; reload checks of the three adapters; one reload-only check of Qwen3-32B with the earlier runner | the same laptop | 2026-11-11 | measured, one run each; section 13 |
| EARLIER-CLOUD | Mistral Large, 8K and 16K tokens, full depth, plus quality pilots | one L40S per run | 2026-11-05 to 2026-11-07 | read from result files, not re-run |
| BANKING77 | Qwen3-32B, 300 steps, held-out accuracy; plus a forward-only reload check on the laptop | one cloud L40S (training and evaluation); the laptop (reload check) | 2026-11-07 for the cloud phases (`summary_train.json` phase times); the laptop reload check carries no date in its result file | measured once |

**Mistral numbers from earlier recorded runs, not re-run**: the EARLIER-CLOUD row not at all, and the
EARLIER-LAPTOP rows for M128 and M512 not with the runner that produced them. The re-check of the Mistral
laptop numbers with that runner has not been run; the SOUP-TRAIN RUN row is a separate measurement.

The result files were copied with a filter (text, JSON, logs); model weights, adapter tensors and optimizer
checkpoints are not included and are identified only by the SHA-256 values in the result files. Several
long raw files (290 KB to 510 KB per run) are represented by small summaries that name the SHA-256 of the file
they were read from.

## 2. Code under test

- **Laptop runs (EARLIER-LAPTOP and RE-RUN)** use Soup's layer-streaming runtime from a frozen export of
  commit **f215941b** (2026-10-15, 1,575 files pinned by SHA-256) under separate experiment runners. They do
  not run `soup train`. The workers load the runtime from a hard-coded path; on 2026-11-07 the export was
  rebuilt from the Soup history and verified against the 1,575 pinned hashes, and the pre-existing copy at the
  hard-coded path verified against them too (`RF/rerun-2026-11-07/log-q32-import-and-first-train-call.txt`,
  lines "verified 1575 runtime files"). The runner for Qwen3-32B has SHA-256
  `813a9a17d40997f1152d7e8895439f7e812d5db9c6227fba9a0093030a28c0b6` and is the same file in the earlier
  run and the re-run (`RF/earlier-recorded/laptop/summary-q32.json`, `RF/rerun-2026-11-07/q32-train-n1.json`).
  The runners are not included in this record.
- **Laptop runs of 2026-11-11 (SOUP-TRAIN RUN, section 13)** run `soup train` itself, not an experiment
  runner, on a local integration tree at commit **ee4cc5d2**: not a release and not `main`, a merge of the
  branches of pull requests #2628, #2653, #2636, #2655, #2658 and #2662 as they stood on 2026-11-11 (all six
  were merged into `main` later; the merged versions were not re-tested), with `soup_cli` 0.77.2 as the
  operator reports. The tree and the wrapper that started `soup train` are not included.
- **Cloud runs (EARLIER-CLOUD)** are stand-alone experiment decoders on stock torch, bitsandbytes,
  transformers and peft; the operator reports that the launch wrappers install no Soup package and that the
  worker sources were searched for `soup_cli` imports with none found (the search output and the sources are
  not part of this record). Their SHA-256 values are listed in the handoff files
  (`RF/earlier-recorded/cloud/ROOT_CONTEXT74_HANDOFF_01.json`, `ROOT_CONTEXT83_HANDOFF_01.json`); the sources
  are not included. The cloud results are therefore evidence for the streaming method, not for Soup's
  shipped code.
- **Banking77 run** trains through `soup train` at Soup main commit
  `74e6628fc27454a6d82502b2d8b2e68b6bf159da` with `training.stream_layers: true`
  (`RF/quality-banking77/FREEZE-trimmed.json`, `soup_main.yaml`), started by a launcher outside the Soup tree
  that applies run-time shims (section 8.1).
- Base weights for the laptop and cloud Mistral/Qwen runs are the published bitsandbytes NF4 checkpoints,
  copied byte for byte into per-layer shards without de- or re-quantisation: 2,053 tensor payloads,
  19,211,023,168 B for Qwen3-32B; 2,645 tensors, 64,449,162,720 B for Mistral Large
  (`RF/earlier-recorded/laptop/q32-REPORT.txt`, `m128-REPORT.txt`). Compute is BF16; adapters and AdamW state
  are FP32. The Mistral figures here are Mistral numbers from earlier recorded runs, not re-run with that
  runner.

## 3. Hardware and resources as recorded

| | laptop | cloud |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB total, driver 616.92 (`RF/earlier-recorded/laptop/hardware.json`) | NVIDIA L40S; 46,068 MiB reported total in the 16K run (`RF/earlier-recorded/cloud/verification-16384-01.json`) |
| GPU memory limit used | the guard stops a run above 7,600 MiB whole-device; PyTorch allocator cap 6.25 GiB (`guard_limits`, `RF/rerun-2026-11-07/attempts-with-a-guard-summary.json`; `m512-REPORT35.txt`) | PyTorch allocator cap 6,710,886,400 B; no whole-device stop |
| CPU | Intel Core i9-14900HX, 24 cores / 32 logical | not recorded in the copied files |
| RAM | 33,271,672 KB visible (31.73 GiB); system commit limit 54,471,286,784 B on 2026-11-03, 93,339,557,888 B on 2026-11-07 (`RF/rerun-2026-11-07/q32-train-rerun-vs-earlier.json`) | not recorded in the copied files |
| disk | two Samsung MZAL81T0HFLB-00BL2 NVMe SSDs, 1,024,209,543,168 B each | container scratch; 86,999,345,040 B required free and 104,716,085,136 B minimum co-located free in the 16K run (`summary-16384-01.json`) |

The laptop was a working machine, not a quiet one: the guard's minimum available host RAM during the six
accepted workers was 5,645.6 to 7,126.3 MiB and the minimum system commit headroom 2,355.9 to 6,629.8 MiB
(table in section 5). Four of those six workers are Mistral runs: Mistral numbers from earlier recorded
runs, not re-run with that runner.

## 4. What was run

| ID | where | model | tokens per step | updates | trainable | data |
|---|---|---|---|---|---|---|
| Q32 | laptop | Qwen3-32B NF4, 32,762,123,264 parameters, 64 layers | 1 x 128 | 3 | rank 8, alpha 16, q_proj + v_proj, all layers: 256 FP32 tensors, 9,961,472 parameters | 381 causal targets over 3 updates; 2 held-out examples with 254 targets |
| M128 | laptop | Mistral Large NF4, 122,610,069,504 parameters, 88 layers | 1 x 128 | 3 | rank 8, alpha 16, q/v, all layers: 352 tensors, 26,673,152 parameters | 3 authored training sequences (381 targets); 2 held-out (254) |
| M512 | laptop | same | 1 x 512 | 3 | same | five authored texts repeated to 512 visible tokens |
| L8K | L40S | same | 1 x 8,192 | 1 | same | one training fixture, 124 supervised tokens |
| L16K | L40S | same | 1 x 16,384, no padding | 1 | same | one training fixture, 124 supervised tokens |
| L-Q | L40S | same | 1 x 8,192 | 4 after a shared 2-update state | same | 4 training rows; 8, then 32, synthetic held-out rows |
| B77 | cloud L40S | Qwen3-32B NF4 via `soup train` | micro-batch 4 | 300 | rank 8, alpha 16, q/v | 1,200 Banking77 rows, one epoch |

The M128 and M512 rows are Mistral numbers from earlier recorded runs, not re-run with that runner; the L8K,
L16K and L-Q rows are Mistral numbers from earlier recorded runs, not re-run.

Sources: `RF/earlier-recorded/laptop/q32-REPORT.txt`, `q32-ANALYSIS.json`, `m128-REPORT.txt`,
`m128-ANALYSIS_FINAL.json`, `m512-REPORT35.txt`; `RF/earlier-recorded/cloud/verification-8192-02.json`,
`verification-16384-01.json`, `ROOT_CONTEXT80_HANDOFF_01.json`, `QUALITY80_EVAL_PLAN_01.json`;
`RF/quality-banking77/soup_main.yaml` (1,200 rows is derived: 300 steps x batch_size 4).

Streaming configuration on the laptop: disk tier, direct I/O, read-ahead 1, **two** GPU decoder slots, one
shared GPU slot for the vocabulary matrices, NF4 scale cache off, per-layer non-reentrant checkpointing
(layer inputs kept, the layer re-read and recomputed in the backward; 868 decoder loads for 88 layers over 3
updates and the held-out passes, `summary-m128.json`). Q32 staged through a 4 GiB pinned host
arena; M128 and M512 through pageable staging (source backing 2,324,713,472 B, pinned source bytes 0)
(`q32-REPORT.txt`, `m128-REPORT.txt`, `m512-REPORT35.txt`). On the cloud, each decoder layer is
materialised, run and released in turn, the layer inputs of the forward are spilled to scratch disk and read
back for the reverse pass, and the 16K recipe releases the attention output before the attention VJP
(`verification-16384-01.json`, `ROOT_CONTEXT81_HANDOFF_01.json`).

## 5. Results: laptop, EARLIER RECORDED (Q32, M128, M512)

Four instruments, which are not interchangeable: **torch** = `torch.cuda.max_memory_allocated` / `reserved`
of the worker (exact for PyTorch's allocator, blind to the CUDA context and driver); **sampled** = the
guard's `nvidia-smi` whole-device reading, taken once per 1.1 to 1.7 s on average (guard elapsed divided by
the sample count, third row of the table); **process RAM** = peak working set and private commit of the
worker; **system** = minimum available physical memory and minimum system-wide commit headroom.

**Mistral numbers from earlier recorded runs, not re-run with that runner** (the M128 and M512 columns of
both tables below). The re-check of the Mistral numbers with that runner has not been run; section 13 is a
separate measurement through `soup train`. The Q32 train step was re-run once with that runner (section 6).

| | Q32 train | Q32 reload | M128 train | M128 reload | M512 train | M512 reload |
|---|---|---|---|---|---|---|
| torch peak allocated, B | 2,780,192,256 | 2,329,868,288 | 3,757,965,824 | 2,418,830,336 | 4,968,217,088 | 3,282,605,568 |
| torch peak reserved, B | 2,965,372,928 | 2,384,461,824 | 4,638,900,224 | 2,464,153,600 | 5,691,670,528 | 3,772,776,448 |
| **sampled whole-device peak, MiB** | **3,002** | **2,440** | **4,598** | **2,516** | **5,602** | **3,764** |
| GPU samples (guard elapsed / samples, s) | 152 (1.16) | 117 (1.11) | 589 (1.37) | 325 (1.66) | 509 (1.43) | 298 (1.61) |
| worker peak working set, MiB | 6,173.1 | 5,874.2 | 4,366.4 | 4,080.4 | 4,357.1 | 4,056.8 |
| worker peak private commit, MiB | 9,825.4 | 8,995.0 | 9,770.8 | 7,307.6 | 10,717.5 | 8,631.2 |
| min available host RAM, MiB | 6,339.7 | 5,844.9 | 5,645.6 | 7,126.3 | 6,636.9 | 7,047.7 |
| min system commit headroom, MiB | 6,457.4 | 6,629.8 | 3,890.0 | 6,105.1 | 2,355.9 | 4,569.9 |
| step times, s | 10.140, 9.474, 9.266 | - | 67.815, 62.491, 60.602 | - | 58.884, 57.389, 55.295 | - |
| worker elapsed, s | 169.354 | 123.196 | 673.724 | 394.405 | 588.668 | 348.416 |
| guard elapsed, s | 175.578 | 129.438 | 804.562 | 539.031 | 728.812 | 478.359 |
| decoder layer loads | 628 | 128 | 868 | 176 | 868 | 176 |
| PID | 41692 | 39296 | 20560 | 32524 | 39232 | 44284 |

Sources: `RF/earlier-recorded/laptop/summary-q32.json`, `summary-m128.json`, `summary-m512.json` (each read
from the run's result file and the guard's summary and samples; the SHA-256 of those originals is in
`summary-sources.json`), `q32-REPORT.txt`, `m128-REPORT.txt`, `m512-REPORT35.txt`. Q32 is Qwen3-32B, M128 and
M512 are Mistral Large. **Mistral numbers (M128 and M512) from earlier recorded runs, not re-run with that runner.**

| run | held-out loss before | after the 3 updates | reloaded in a fresh process |
|---|---|---|---|
| Q32 | 4.309379 / 4.310736 | 4.298624 / 4.307984 | 4.298624 / 4.307984 (equal to the trained losses: yes) |
| M128 | 3.774966 / 3.844886 | 3.651912 / 3.692650 | 3.651912 / 3.692650 (equal to the trained losses: yes) |
| M512 | 1.079077 / 1.098381 | 1.050543 / 1.058185 | 1.050543 / 1.058185 (equal to the trained losses: yes) |

Mistral numbers from earlier recorded runs, not re-run with that runner (the M128 and M512 rows above; sources
`summary-m128.json`, `summary-m512.json`).

- **Step times include diagnostics**: adapter clones, per-tensor gradient and update audits, durable phase
  records and (M512) synchronous cache trims; the worker's elapsed time also includes two full SHA-256 passes
  over the source and store files. They are not throughput numbers. M512 being faster per step than M128 is
  not a conclusion: different sessions, routes and cache states (`m512-TRAIN_COMPARISON35.txt`).
- **Step losses** (Q32: 3.5959742069244385, 4.572229862213135, 4.455287456512451; the others are in
  `summary-m128.json` and `summary-m512.json`) belong to different examples and are not a curve.
- **Disk.** Qwen3-32B source 19,211,935,565 B in 4 files; Mistral Large source 13 shards of 64.450 GB, "around
  129 GB SSD" for source and native store together (`q32-REPORT.txt`, `m128-REPORT.txt`). The Mistral sizes are
  Mistral numbers from earlier recorded runs, not re-run with that runner.
- **Held-out losses** moved by small amounts on 2 held-out examples (254 targets). That is a capacity check,
  not a quality result (`m512-REPORT35.txt`: "capacity metrics").

## 6. Results: laptop, RE-RUN on 2026-11-07 (Qwen3-32B only)

The re-run used the same worker (SHA-256 `813a9a17...`), the same frozen runtime export, the published
source files of Qwen3-32B (12 of 12 recorded SHA-256 values equal) and a Soup NF4 store rebuilt that day
from them (69 of 69 recorded file hashes equal), on the same laptop with the same guard limits (7,600 MiB
whole-device, 8,192 MiB available host RAM required at start, none of them lowered)
(`RF/rerun-2026-11-07/weights-hash-checks.json`, `attempts-with-a-guard-summary.json`).

| quantity | earlier recorded run | re-run, same laptop, 2026-11-07 | call |
|---|---|---|---|
| sampled whole-device peak, MiB | 3,002 (152 samples) | 3,002 (238 samples) | identical |
| torch peak allocated, B | 2,780,192,256 | 2,780,192,256 | identical |
| torch peak reserved, B | 2,965,372,928 | 2,965,372,928 | identical |
| trainable parameters / tensors | 9,961,472 / 256 | 9,961,472 / 256 | identical |
| step losses (3 updates) | 3.5959742069244385 / 4.572229862213135 / 4.455287456512451 | 3.5959742069244385 / 4.572229862213135 / 4.455287456512451 | identical, every float digit |
| held-out loss before -> after, example 1 | 4.3093791007995605 -> 4.298623561859131 | 4.3093791007995605 -> 4.298623561859131 | identical (both examples) |
| `adapter_model.safetensors` SHA-256 | `88aa4279ed57aa75b10c758810415441e22251ad71f8c0185783c40eb614b048` | `88aa4279ed57aa75b10c758810415441e22251ad71f8c0185783c40eb614b048` | identical |
| `adapter_config.json`, `README.md` SHA-256 | `dcd3efac...`, `e2efd27d...` | `dcd3efac...`, `e2efd27d...` | identical |
| decoder layer loads | 628 | 628 | identical |
| step times (3 updates), s | 10.140, 9.474, 9.266 (median 9.474) | 30.505, 22.325, 22.475 (median 22.475) | **2.37x slower, unexplained** |
| worker elapsed, s | 169.354 | 256.132 | 1.51x |
| file-hashing pass before / after the run, s | 35.249 / 36.940 | 48.708 / 42.148 | +38% / +14% |
| bytes read by the worker (last host sample) | 238,802,365,746 | 238,862,422,624 | +0.03% |
| min available host RAM during the run, MiB | 6,339.7 | 3,795.7 | lower in the re-run |
| available host RAM at guard start, B | 14,809,038,848 | 10,386,354,176 | lower in the re-run |
| min system commit headroom, MiB | 6,457.4 | 23,738.4 | different |
| system commit limit, B | 54,471,286,784 | 93,339,557,888 | different; the cause is not known |
| guard elapsed, s | 175.578 | 305.547 | |
| reload in a fresh process | exact: both held-out losses and the 128-value logits slices | **not reproduced: no reload attempt produced a result** | NOT REPRODUCED on 2026-11-07; one later reload-only run passed (section 13.2) |

Sources: `RF/earlier-recorded/laptop/summary-q32.json`; `RF/rerun-2026-11-07/q32-train-n1.json`,
`q32-train-rerun-vs-earlier.json`. The `stage32_provenance.json` hash differs between the two runs (it is
part of the adapter folder and was not examined further); the three adapter files that carry the weights and
configuration are identical.

**Reproduced (n = 1).** Peak memory, parameter count, step losses, held-out losses and the adapter file are
identical to every recorded digit. One run is not three: nothing here says how much these numbers vary from
run to run.

**Step time: 2.37x slower, unexplained.** The worker read the same number of bytes (+0.03%), made the same
628 layer loads and ran the same code, yet its median update took 22.475 s against 9.474 s and the first
update 30.505 s against 10.140 s. What differed on the day, and is not shown to cause the gap: the file-hashing
passes were 38% and 14% slower, available host RAM was lower (3,795.7 MiB minimum against 6,339.7 MiB), and
the system commit limit was 93,339,557,888 B instead of 54,471,286,784 B. "A busier machine and a colder
file cache" is a hypothesis, not a measured cause: GPU clocks, throttling and the disk queue were not sampled during this run,
and a repeat on a quiet machine (the second and third run that were planned) did not happen. The operator
reports that a later unattended window the same night was cancelled before anything ran (available RAM never
reached the level the plan required, and the laptop was on battery and went to sleep); no file in this record
shows it, and nothing from that window is reported.

**Reload in a fresh process: not reproduced on 2026-11-07.** Six attempts (a to f), none produced a result, none ended in a
mismatch. The worker's result file never reached `passed` in any of them. Five attempts have a guard summary
and are in the table; attempt c was stopped by hand before a worker started.

| attempt | start (UTC) | outcome | guard elapsed, s | whole-device max sampled, MiB |
|---|---|---|---|---|
| q32 train, first call | 2026-11-07T08:16:13Z | supervisor_error: RuntimeError: preflight resource failure: host_budget | 55.8 | - |
| q32 train (the completed run) | 2026-11-07T08:29:56Z | completed | 305.5 | 3,002 |
| q32 reload a | 2026-11-07T08:35:03Z | guard_stopped: host_emergency_floor | 170.8 | 2,593 |
| q32 reload b | 2026-11-07T09:38:43Z | guard_stopped: device_fb_budget | 154.3 | 7,843 |
| q32 reload d | 2026-11-07T10:02:34Z | worker_failed | 53.8 | 4,452 |
| q32 reload e | 2026-11-07T10:04:08Z | guard_stopped: device_fb_budget | 122.8 | 7,863 |
| q32 reload f | 2026-11-07T10:10:47Z | supervisor_error: RuntimeError: preflight resource failure: host_budget | 48.6 | - |

Reload attempt c is not in the table: the operator reports that it was stopped by hand before a worker
started, because a measurement by someone else was holding the GPU. A second train call was stopped by hand
in the same way when another job appeared on the GPU, before the completed train run; it is not in the table
either. Neither has a guard summary and no file of theirs is included, so these two reasons are the operator's
report and are not shown by any file in this record (`attempts-with-a-guard-summary.json`, field `note`).
`guard_stopped: device_fb_budget` means the whole-device reading went above the 7,600 MiB limit while this
worker's own earlier-recorded reload peak was 2,440 MiB (`summary-q32.json`). The operator's run log (not
included) attributes the rest to an unrelated job that started on the GPU during the reload; the process
listing that would show it is not part of this record, so the cause is not established here. For reload d the
worker's own preflight refused to start
(`worker_error` in the attempts file: host available 7,428,878,336 B against 7,784,628,224 B required).
Sources: `RF/rerun-2026-11-07/attempts-with-a-guard-summary.json`, `log-q32-import-and-first-train-call.txt`,
`log-q32-train-and-first-reload.txt`.

A single reload-only run of the same Qwen3-32B check with the same runner, made on 2026-11-11, passed
(`fresh_process_exact: true`; section 13.2).

**Mistral Large: not re-run with that runner.** Mistral numbers from earlier recorded runs, not re-run with
that runner. On 2026-11-07 only the
Mistral weight files on the laptop were re-hashed: the 45 recorded source files and the 93 recorded store
files all equal the earlier values (`weights-hash-checks.json`: 0 different or missing in each). A re-run of
M128 and M512 was estimated by the operator to need an exclusive window of about two hours (an estimate, not
derived from any file here) and was not made then; the earlier wrappers for it were not run on 2026-11-11
either (section 13.3). Two further
limits of any future M512 re-run, from reading the worker source (not included): the accepted 5,602 MiB came
from a trimmed variant of the worker that refuses to run unless its original absolute paths exist, and its
untrimmed predecessor was stopped by the guard after one update (section 10).

## 7. Results: cloud (one L40S, allocator cap 6,710,886,400 B), EARLIER RECORDED, Mistral Large

Mistral numbers from earlier recorded runs, not re-run. No Soup code. This section's tables, the continuation
and section 7.1 are all Mistral numbers from earlier recorded runs, not re-run.

| | L8K (8,192 tokens) | L16K (16,384 tokens) |
|---|---|---|
| run ID | l40s-strict-full88-sft8192-20261106-02 | l40s-strict-full88-sft16384-20261106-01 |
| layers forward / reverse | 88 / 88 | 88 / 88 |
| supervised tokens | 124 | 124 |
| loss | 0.8814825415611267 | 0.8498985171318054 |
| torch peak allocated, B | 3,489,752,064 | 6,107,063,296 |
| torch peak reserved, B | 3,902,799,872 | 6,295,650,304 (headroom to the cap 415,236,096) |
| **sampled whole-device peak, MiB** | **4,323** (3,050 samples) | **6,597** (14,381 samples, every 250 ms) |
| activation spill to scratch disk, B | not recorded | 35,433,480,192 |
| attention output released before the attention VJP, B per layer | not part of this run | 402,653,184, all 88 layers |
| model download in the container, s | 729.013 | 260.602 |
| worker, s | 2,128.523 | 3,618.443 (wrapper summary: 3,621.113) |
| fresh-process restore, s | 6.561 | 7.007 |
| saved state, B | 320,486,997 | 320,486,997 |
| restored exactly | 352 adapters, 704 Adam moments, 352 step counters, CPU and CUDA RNG | 352 adapters, 704 Adam moments, 352 step counters, CPU and CUDA RNG |
| PIDs update / restore | 1956 / 6629 | 450 / 485 |

Sources: `RF/earlier-recorded/cloud/verification-8192-02.json`, `verification-16384-01.json`,
`summary-16384-01.json`, `ROOT_CONTEXT74_HANDOFF_01.json`, `ROOT_CONTEXT83_HANDOFF_01.json`. The 8K run's
device samples came from a sampler run outside the frozen wrapper by the person running the job; the sampler's
command line is not in the copied files, and the 8K handoff reports only the count (3,050) and the peak.

| phase of the 16,384-token update | layer | peak allocated, B | peak reserved, B |
|---|---|---|---|
| embedding | - | 402,653,184 | 419,430,400 |
| forward | 0 | 4,783,907,840 | 6,232,735,744 |
| attention forward | 87 | 4,381,254,656 | 5,184,159,744 |
| mlp vjp | 87 | 3,800,066,048 | 4,406,116,352 |
| attention vjp | 87 | 6,030,573,568 | 6,295,650,304 |
| terminal loss | - | 6,107,063,296 | 6,167,724,032 |
| optimizer checkpoint | - | 427,950,080 | 429,916,160 |

Source: `verification-16384-01.json` (`allocator.phase_peaks`). The whole-device samples include the CUDA
context; all 14,381 were at or below 8,192 MiB (`device_samples.all_observations_within_8192_mib`).

- At 16K the release of the attention output before the attention backward is what fits the cap: the two-layer
  gate measured 6,694,109,184 B reserved with the output retained and 6,295,650,304 B with it released
  (398,458,880 B less); that gate did not sample the whole device (`ROOT_CONTEXT81_HANDOFF_01.json`).
- **Continuation at 8K.** A control process ran updates 1 and 2 (its update 1 matched the 8K run exactly); a
  fresh process restored the step-1 state and ran update 2: step-2 loss 0.6499557495117188, all 352 gradient
  hashes, 352 adapters, 704 Adam moments, 352 step counters and CPU/CUDA RNG state equal, identical state
  payload SHA-256; both at 3,809,829,888 B allocated and 4,223,664,128 B reserved (PIDs 575 and 617)
  (`ROOT_CONTEXT75_HANDOFF_01.json`).

### 7.1 Quality pilots at 8,192 tokens (all 88 layers): no quality result

Mistral numbers from earlier recorded runs, not re-run.

- **Pilot (L-Q, first design).** Four updates (steps 3 to 6, one per training row) with losses 1.3326656818389893,
  0.2171434760093689, 0.006549788638949394 and 0.005973503924906254, then 8 synthetic four-way retrieval rows:
  **6 of 8 correct before, 4 of 8 after**; mean label NLL 1.935 to 3.831 (`ROOT_CONTEXT76_HANDOFF_01.json`).
- **Pre-registered comparison.** 32 fresh synthetic rows (16 counterfactual pairs, 8,192 visible tokens each),
  scored on the shared 2-update starting state (A2), on the control after 4 more updates (A6) and on a branch
  that changed one loss coefficient after the same 4 updates (B6). The plan required B6 to have at least 3
  more correct than A6 (`QUALITY80_EVAL_PLAN_01.json`):

| arm | correct / 32 | both of a pair correct | mean label NLL | mean answer NLL |
|---|---|---|---|---|
| A2 (start) | 15 | 2 | 3.2789 | 2.4653 |
| A6 (control) | **22** | 7 | 1.2196 | 0.6389 |
| B6 (branch) | **22** | 7 | 0.8627 | 0.6509 |

B6 had 0 more correct than A6 (`b6_minus_a6_correct` 0, `advances` false): **gate not met**. Paired A6 to B6:
20 right in both, 8 wrong in both, 2 fixed, 2 broken. The branch lane trained for 2,192.814 s excluding the
download (`ROOT_CONTEXT80_HANDOFF_01.json`). Mistral numbers from earlier recorded runs, not re-run.

## 8. Qwen3-32B, 300-step Banking77 fine-tune through `soup train` (cloud L40S, 6.25 GiB cap, EARLIER RECORDED)

**This run did not train on the laptop.** It trained on a cloud L40S with the PyTorch allocator of the
training process capped at 6,710,886,400 B (6.25 GiB). The whole device was sampled at 5,804 MiB (5.67 GiB) at
most. Laptop evidence for this model is limited to the forward-only reload check in section 8.3, the Qwen3-32B re-run in
section 6 and the 3-step runs of section 13. Nothing here shows that the 300 steps fit or run on an 8 GB card, and the step times of the cloud
card do not transfer to the laptop.

Setup: Soup main at `74e6628fc27454a6d82502b2d8b2e68b6bf159da` (`soup_cli 0.77.2` from the payload), `soup
train` with `training.stream_layers: true`, `stream_buffers: 2`, `stream_vram_override: 6710886400`, rank-8
q/v LoRA (alpha 16), AdamW lr 1e-4, weight decay 0, clip 1.0, batch_size 4 with no accumulation, 1 epoch =
300 optimizer steps over 1,200 rows, seed 32017. Libraries: torch 2.14.0+cu130, transformers 5.17.0, peft
0.20.0, trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.15.0 (`RF/quality-banking77/soup_main.yaml`,
`versions.txt`, `FREEZE-trimmed.json`). Data: PolyAI/banking77 revision
`1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` (full official test split, a 308-row dev split), a 500-item MMLU
subset (cais/mmlu `c30699e8356da336a370243923dbaf21066bb9fe`) and a 400-item GSM8K subset (openai/gsm8k
`740312add88f781978c0658806c59bc2815b9866`) (`FREEZE-trimmed.json`, `SCORE.json`). The evaluation base is the
published `unsloth/Qwen3-32B-bnb-4bit`; the training source is `Qwen/Qwen3-32B` (BF16, revision
`9216db5781bf21249d130ec9da846c4624c16137`), which Soup quantised to NF4 itself.

The run follows a written protocol (SHA-256
`a00f1c1f29b2657ffc7c311b75bc09e86514d2c3c78872906ee600b032c8f120`, recorded in the freeze file and again in the
scorer output). The freeze file, which pins the data splits and the code that ran, was written after the base
model's development-split evaluation and before any fine-tuned model existed (its own `note` field). The
protocol text is not part of this record; the gate thresholds as the scorer applied them are in `SCORE.json`.

| suite | n | base | fine-tuned | delta, pp | 95% paired bootstrap CI, pp | exact McNemar p | discordant (base right, tuned wrong / base wrong, tuned right) |
|---|---|---|---|---|---|---|---|
| **Banking77 test (full official split)** | 3,080 | 69.578% (2,143) | 82.825% (2,551) | +13.247 | [+11.883, +14.643] | 7.85e-82 | 54 / 462 |
| Banking77 dev | 308 | 67.208% (207) | 82.143% (253) | +14.935 | [+10.390, +19.481] | 1.17e-10 | 5 / 51 |
| MMLU subset | 500 | 76.400% (382) | 77.200% (386) | +0.800 | [+0.000, +1.800] | 0.219 | 1 / 5 |
| GSM8K subset | 400 | 95.250% (381) | 95.750% (383) | +0.500 | [+0.000, +1.250] | 0.5 | 0 / 2 |

Source: `RF/quality-banking77/SCORE.json` (paired percentile bootstrap, 10,000 resamples, seed 77077; a bootstrap
that resamples the 77 classes gives [+8.994, +17.727] pp for the test delta). The base arm's valid-output rate
was 99.026% and the fine-tuned arm's 99.968%; macro-F1 rose from 0.690 to 0.826; strict and lenient parsing
agree for both arms. Neither general check fell; both paired intervals have a lower bound of 0.0 (MMLU p =
0.219, GSM8K p = 0.5), so no change is shown in either direction. One seed, one run: seed-to-seed variation
was not measured.

Gates as the scorer computed them (`SCORE.json`): G0, base development accuracy 67.208% against a ceiling of
90% (the task leaves room to improve); G1, delta +13.247 pp against a threshold of +5.0 with the interval's
lower bound +11.883 above 0; G2, MMLU and GSM8K interval lower bounds 0.0 against limits of -3.0 and -5.0; G3b,
a repeat of the fine-tuned evaluation in a new process byte-identical on all 3,080 test, 308 dev and 400 GSM8K
generations, with equal MMLU letter-logit SHA-256. V3: cached-prefix decoding agreed with full-prompt decoding
on 64 of 64 base items and **63 of 64** fine-tuned items (the threshold was 63: margin 0). V4: the streamed
engine's cross-entropy differed from the Hugging Face path by a mean 0.0282 (base) and 0.0047 (fine-tuned) per
item on 32 probe items, with the same sign of the fine-tuning effect on 32 of 32.

### 8.1 Memory and the non-streaming control

| | streamed (Soup main, `stream_layers: true`) | resident control (HF + PEFT, same recipe) |
|---|---|---|
| Banking77 test accuracy | 82.825% | 83.182% |
| delta over base (95% CI), pp | +13.247 [+11.883, +14.643] | +13.604 [+12.240, +14.968] |
| Banking77 dev accuracy | 82.143% | 83.117% |
| training process torch peak allocated / reserved, MiB | 5,052.4 / 5,216 | 20,126.7 / 24,236 |
| micro-batch | 4 (batch_size 4, no accumulation) | 2 |
| 300 steps, time as recorded by each run's own script | 804.0 s (sum of step-end gaps) | 879.3 s |
| trained adapter SHA-256 | `1a9847ec4ed14a3a...` | `78722e57538a408d...` |

Streamed against resident on the test split: resident higher by +0.357 pp (paired CI [+0.032, +0.682], exact McNemar p = 0.052, discordant 8 / 19); generated token ids identical on 3,033 of 3,080 items.

Sources: `RF/quality-banking77/summary_train.json` (whole device 5,804.19 MiB from a 20 ms in-process sampler,
5,805 MiB from 3,535 `nvidia-smi` samples; median step 2.646 s on a compute-bound card, no claim for the
laptop; gradient clipping at 1.0 acted in 217 of 300 steps), `control_b77_test.json`, `control_b77_dev.json`,
`control-summary.json`, `resident-control-memory.json`. The control is the same recipe written out in plain
Hugging Face + PEFT on the published NF4 checkpoint, with gradient checkpointing and a different micro-batch
(2 against 4) and no allocator cap, so the memory comparison gives the order of magnitude (about 4x), not a
tuned ratio. The two engines reach the same accuracy within about a third of a pp; the residual difference is
within what a different LoRA A initialisation (Soup draws it from a seeded CPU generator, PEFT from the global
generator), micro-batching and kernel paths could explain, but this was not isolated.

The launcher outside the Soup tree applied these shims and printed one line each
(`RF/quality-banking77/launcher-shim-lines-main-run.txt`): the memory cap (`cap_bytes=6710886400`); the
learning-rate schedule (warmup 10 steps, cosine to 1e-5, replacing the trainer's own); sequential row order
(`train_sampling_strategy=sequential`); a per-step logger; and a cast of the 256 trainable LoRA tensors from
bf16 to fp32 before the first optimizer step (`before={bfloat16: 256}`, `after={float32: 256}`), with AdamW
state in fp32. Without the cast the adapter would have been trained in bf16; whether that is a default worth
changing in Soup is open. The logged learning rate equalled the formula at all 300 steps (largest absolute
difference 2.4e-20) and all 256 saved adapter tensors are fp32 (`summary_train.json`, `V5`).

### 8.2 Caveats specific to this run

- The adapter was trained against Soup's own NF4 quantisation of the BF16 weights and evaluated on the
  published NF4 checkpoint. In layers 0, 31 and 63, every packed weight tensor, both code tables and the final
  norm are byte-equal, while per-block scale statistics differ in 6, 6 and 2 tensors, by about 1e-7 relative on
  average (mean 7.3e-8 to 1.8e-7 over four tensors checked; one block of one layer-0 tensor differs by 2.8%)
  (`nf4_compare.json`, `nf4_scale_diff_adhoc.json`). The streamed-versus-Hugging-Face check above measures the
  effect on outputs.
- The protocol's reload check in the training process itself could not be made, because `soup train` exits
  after saving; it was replaced by two fresh processes loading the saved adapter into the streamed engine, with
  identical per-item logits SHA-256 and cross-entropy on 32 of 32 items (`G3A_PRIME.json`, adapter SHA-256
  `1a9847ec4ed14a3a2b9fa3be1ba5c6ab9700898fdccd6475214b86dbb4c12659`). That shows determinism of the reload,
  not equality with the in-memory trained state.
- Banking77 may be in Qwen3's pretraining data; that would inflate the base figure. This was not tested.
- One task, one model, one seed, one rank-8 q/v adapter. No claim about other tasks, other precisions, full
  fine-tuning or the 123B model.

### 8.3 Forward-only reload check on the 8 GB laptop

The saved adapter (same SHA-256) was loaded into Soup's streamed engine on the RTX 5070 Laptop (disk tier, the
Qwen3-32B NF4 store used by the other laptop runs) and 32 probe items were scored once: torch peak allocated
**2,613,421,056 B (2,492.4 MiB)**, mean cross-entropy 0.13946 against 0.14010 for the same items on the cloud
card, mean absolute difference per item 0.0035, largest 0.0198 (`RF/quality-banking77/laptop-reload-check.json`,
read from the laptop's result file and from `STREAM.json`, the cloud card's streamed-engine scores).
The logits are not bit-identical to the cloud card (a different GPU; see the NF4 note above). A second laptop
run for same-hardware determinism was not made.

The adapter weights themselves are not part of this record; the adapter is identified only by the SHA-256
above. No download link is given here, and this record did not check any published copy against that hash.

## 9. Correctness checks

| check | run | what was compared | result |
|---|---|---|---|
| adapter reload in a fresh process | Q32 (earlier recorded) | all 256 saved adapter tensors, the 2 held-out losses, a predetermined 128-value logits slice per example, artifact hashes | exact; train PID 41692, reload PID 39296 (`q32-REPORT.txt`, `summary-q32.json`) |
| adapter reload in a fresh process | Q32 (re-run) | same | **not reproduced: no result** on 2026-11-07 (section 6); one later reload-only run with that runner passed (section 13.2) |
| adapter reload in a fresh process | M128 (earlier recorded) | all 352 adapter tensors; 2 held-out losses; SHA-256 of the bytes of each complete BF16 logits tensor [1, 128, 32768] | bit-identical; PIDs 20560 / 32524 (`m128-REPORT.txt`) |
| adapter reload in a fresh process | M512 (earlier recorded) | the same at [1, 512, 32768] | bit-identical; PIDs 39232 / 44284; 1,056 gradient and 1,056 update rows audited (`m512-REPORT35.txt`, `m512-FULL512_AUDIT_02.json`) |
| adapter reload in a fresh process | Q32, reload-only run with the earlier runner, 2026-11-11 | the recorded held-out losses and 128-value logits slices | exact: `fresh_process_exact: true` (section 13.2) |
| adapter reload in a fresh process against the training process's own logits | SOUP-TRAIN RUN: Q32, M128, M512 | complete logits [1, 128, vocabulary] of the reloaded model against the float32 logits captured from the training process | **not bit-identical**: the comparison script stopped at its dtype guard (exit 2); a later CPU comparison of the saved logits found 70.5% to 84.3% of the elements different after casting to bfloat16, largest absolute difference 1.3125 (Qwen3-32B), 0.875 and 0.625 (Mistral Large) (section 13.1) |
| adapter reload across fresh processes | SOUP-TRAIN RUN: Q32, M128, M512 | complete logits of two fresh reload processes, and the reload of the compare step | bit-identical, 0 elements differing in each (section 13.1) |
| streamed vs non-streamed module parity | Q32 preparation | q/v/up projections: packed bytes, BF16 weights, outputs, dX, LoRA gradients and two updates; 3 real layers through 2 slots with recycling: outputs, dX, 12 adapter gradients and updates over 2 updates | exact (`q32-REPORT.txt`) |
| optimizer-state restore in a fresh process | L8K, L16K | 352 adapters, 704 AdamW moments, 352 step counters, CPU and CUDA RNG | exact (section 7) |
| continuation from a restored state | L8K, second update | step-2 gradients (352 hashes), state, RNG: fresh resume against a continuous control | exact (section 7) |
| base untouched | laptop | source and store files hashed before and after each worker (81 for Q32, 138 for Mistral) | unchanged (`q32-REPORT.txt`, `m128-REPORT.txt`) |

"Exact" means equality against the same NF4 checkpoint on the same machine and stack, never against a BF16
model. The restore checks load saved state and compare it; the only restored state that was trained further
and compared is the 8K continuation. Fresh laptop processes read from a warm OS file cache (the hashing passes
warm it); a fresh process is not a cold cache (`q32-REPORT.txt`).

The M128 and M512 rows (and the Mistral part of the "base untouched" row) are Mistral numbers from earlier
recorded runs, not re-run with that runner; the L8K and L16K rows are Mistral numbers from earlier recorded
runs, not re-run.

## 10. Every attempt in these experiments

| experiment | attempt | outcome | source |
|---|---|---|---|
| Q32 | first model download | fetched metadata only (CLI pattern syntax); redone with explicit file names | `q32-REPORT.txt` |
| Q32 | module controls 01, pooled control 01 | failed before allocation: the memory-fraction API needs `cuda:0`; a reduced test vocabulary invalidated the pad token; the 02 runs passed | `q32-REPORT.txt` |
| M128 | train-01 (pinned host staging) | **stopped by the guard** (`system_commit_floor`) after two full held-out forwards and before any update | `m128-REPORT.txt`, `guard-m128-train-01-stopped.json` |
| M128 | train-02 (pageable staging), reload | passed | `m128-REPORT.txt` |
| M512 | untrimmed train | **stopped by the guard** (`system_commit_floor`) after 1 update (70.5146468 s), sampled 5,512 MiB; the trimmed variant then passed | `summary-m512-untrimmed-attempt.json`, `m512-TRAIN_COMPARISON35.txt` |
| laptop campaign, 4 x 128, 120 planned updates (Stage 36) | campaign-01 | 10 updates (median 55.099 s), then **stopped by the guard** (`system_commit_floor`) in the next chunk with no new complete update; the held-out test was never opened | `stage36-CURRENT_STATE.json`, `stage36-CHECKPOINT10_OBSERVATION_01.json`, `stage36-FAILURE_OBSERVATION_02.json` |
| laptop campaign (Stage 42) | long run | 19 updates accepted; the next chunk was lost to a Windows sleep (guard `overall_timeout` after wake); no evaluation | `stage42-LONG_FAILURE_DIAGNOSIS_01.json` |
| L8K chain | two-layer attempt 01 | rejected: the native VJP reserved 6,715,080,704 B, 4 MiB over the cap | `ROOT_CONTEXT73_ATTEMPT01_01.json` |
| L8K chain | two-layer attempt 02 | passed; sampled 6,717 MiB, max reserved 6,421,479,424 B | `ROOT_CONTEXT73_HANDOFF_01.json` |
| L8K | full-depth -01 | worker deadline at 1,800.533 s after 88 forward layers and reverse to group 80; no checkpoint | `verified-timeout-8192-01.json` |
| L8K | full-depth -02 | passed | `verification-8192-02.json` |
| L16K | two-layer gate, then full depth | passed | `ROOT_CONTEXT81_HANDOFF_01.json`, `verification-16384-01.json` |
| L-Q | controls | the local client was lost when the laptop slept; A6 was re-run in a fresh app and the A2 scores were carried | `ROOT_CONTEXT80_CONTROLS_INTERRUPTION_01.json` |
| RE-RUN | Qwen3-32B | one train run completed (after a first call refused by the guard's preflight and a second call stopped by hand before its worker started); six reload attempts without a result | section 6 |
| SOUP-TRAIN RUN | Qwen3-32B, 128 tokens | one run, completed | section 13 |
| SOUP-TRAIN RUN | Mistral Large, 128 tokens, first attempt | **discarded**: another GPU job started on the device during its first step; the run's own process was stopped by hand, with no loss line and no adapter; the one retry completed | section 13 |
| SOUP-TRAIN RUN | Mistral Large, 512 tokens | the first launch was refused by the pre-run check (another GPU job on the device) and nothing ran; one launch after about 5 minutes of waiting completed | section 13 |
| SOUP-TRAIN RUN | reload checks of the three adapters | the comparison against the training process stopped at its dtype guard (exit 2) three times; the two-process fallback passed three times | section 13.1 |
| earlier-runner reload-only run | Qwen3-32B | the first attempt was refused by the pre-run check (another GPU job on the device); one launch after about 12 minutes of waiting passed | section 13.2 |
| earlier-runner wrappers | Qwen3-32B train repeats 2 and 3; Mistral 128 and 512 | **not run**: the wait for the GPU gave up for the second; the others were not attempted | section 13.3 |

Two further cloud runs (a second 16K update and a quality evaluation) were still in progress when the records
were copied; they are not used here. Each accepted configuration ran once, apart from the single re-run in
section 6, so no spread is known for any number in sections 5, 7 and 8. Three of the four stopped laptop
attempts in this table were stopped by the system-commit floor and the fourth by a sleep; for M128 and M512 the
reports say the later success cannot be attributed only to the code change, because the rest of the machine's
memory use differed between sessions (`m128-REPORT.txt`, `m512-TRAIN_COMPARISON35.txt`).

Mistral numbers from earlier recorded runs, not re-run with that runner (the cloud rows: not re-run at all):
every row of this table above the SOUP-TRAIN RUN rows except the two Q32 rows and the RE-RUN row. The
RE-RUN row is Qwen3-32B measured again with that runner; the SOUP-TRAIN RUN rows are separate
measurements through `soup train`, and the earlier-runner reload-only row is Qwen3-32B.

## 11. What this does NOT show

- **No Mistral quality claim.** The laptop Mistral runs are 3-update capacity tests on authored texts; the
  only pre-registered Mistral quality gate was not met (22/32 against 22/32) and its pilot lost accuracy
  (6/8 to 4/8). No run on the large models trained for long: the two longer laptop campaigns recorded reached
  10 and 19 updates and were stopped, and no evaluation of a trained model is recorded for either; the
  longest cloud sequence is 6 updates at 8,192 tokens and 1 at 16,384. The 2026-11-11 `soup train` runs
  are 3 steps on 3 training rows and make no quality claim either. Mistral numbers from earlier recorded
  runs, not re-run (the laptop ones: not re-run with that runner).
- **Mistral numbers from earlier recorded runs, not re-run with that runner; the Qwen3-32B reload was not
  reproduced on 2026-11-07 and passed in one reload-only run on 2026-11-11.** The re-check of the Mistral
  numbers with that runner has not been run. The Qwen3-32B train-step reproduction is one run, and its step
  time is 2.37x the earlier record for a reason nobody has found.
- **16K is not a laptop result.** It ran on an L40S (46,068 MiB reported) with the PyTorch allocator capped at
  6.25 GiB. The cap does not bound the CUDA context, the driver or the 8 GB laptop's WDDM behaviour; the
  handoff itself says a laptop 16K reproduction is still needed (`ROOT_CONTEXT83_HANDOFF_01.json`). 8K on the
  laptop was never run.
- **The Banking77 fine-tune is not a laptop result** (section 8).
- **The laptop runs are short contexts**: 128 and 512 tokens per step.
- **Sampled peaks are lower bounds.** `nvidia-smi` was read about once per 1.1 to 1.7 s on the laptop and every
  250 ms in the 16K run; anything shorter between samples is unmeasured. "8 GB" means the card the laptop has:
  nothing here shows a 4 GB or 6 GB card works.
- **Single runs, diagnostic timings.** Step times include hashing, cloning, audits and synchronous trims; they
  say nothing about throughput or variance.
- **Exactness is self-consistency** against the same NF4 checkpoint on the same machine, not against BF16, GGUF
  or another GPU; the Q32 reload check covers 256 logit values plus losses, not full logits. For the
  `soup train` adapters of section 13 even that did not hold against the training process: the reloaded
  model's logits are not bit-identical to the training process's own logits.
- **The `soup train` adapters do not reload to the training process's logits.** In all three 2026-11-11 runs
  70.5% to 84.3% of the logits differ (largest absolute difference 1.3125, 0.875 and 0.625), the cause is
  not determined, and reload against another reload is the only exact comparison made there (section 13.1).
- **Not full-weight training**: rank-8 LoRA on q_proj and v_proj only; all base weights frozen.
- **Not Soup's shipped training path, except the Banking77 run and the 2026-11-11 laptop runs.** The earlier
  laptop runs use Soup's layer-streaming runtime as of f215941b under separate experiment runners; the cloud
  Mistral runs use no Soup code; the Banking77 run uses `soup train` through a launcher with shims (section
  8.1); the 2026-11-11 laptop runs use `soup train` on a local integration tree (a merge of the branches of six
  pull requests as they stood on 2026-11-11, not `main` and not a release) started through a wrapper that
  adds a logits capture (section 13).
- **Neither host memory nor GPU memory is shown to have slack on the laptop.** The guard stopped three
  earlier laptop attempts on system commit, not on GPU memory (section 10); it stopped two of the six
  reload attempts of 2026-11-07 on whole-device GPU use above 7,600 MiB, which the operator's log attributes
  to another job, and one on the host emergency floor (section 6); and at 512 tokens through `soup train`
  the whole device reached 6,998 of 8,151 MiB in one run (section 13). Whether a given machine has the commit
  headroom is not predicted by its installed RAM (`m128-REPORT.txt`).
- **Artifacts are not included**: weights, adapter tensors and optimizer checkpoints are known only by hash,
  and the experiment runners and launchers are not part of this record, so the runs cannot be repeated from this
  repository alone.

## 12. Files

All under `benchmarks/results/big-models-8gb-laptop/`:

| folder | contents |
|---|---|
| `earlier-recorded/laptop/` | the reports and analysis files of Q32, M128 and M512, the four summaries read from the raw result files (`summary-*.json`), the stopped-attempt records, the Stage 36 and Stage 42 records, `hardware.json` |
| `earlier-recorded/cloud/` | the handoff files for the 73, 74, 75, 76, 80, 81 and 83 steps, the 8K and 16K verification files, the 16K wrapper summary |
| `rerun-2026-11-07/` | the re-run train result, the comparison with the earlier run, the attempts table source, the weight-hash checks, the two run logs |
| `quality-banking77/` | `SCORE.json` (scorer output), the control and reload-check summaries, training summary, config, launcher lines, trimmed freeze file |
| `rerun-2026-11-11/` | section 13: for each of the three `soup train` runs the config, the training log, the driver summary, the 1 s GPU samples and the host-memory samples; the GPU samples, host-memory samples and driver summary of the discarded attempt; the step losses read from each run's `trainer_state.json`; the reload comparison result (dtype guard), the two-process fallback result and the CPU comparison of the saved logits for each adapter; the import summary of the Mistral Large store; an excerpt of the wrapper log of the earlier-runner reload-only run; the operator's scrubbed wait logs; the operator's notes; `README.txt` |

Preparation: a copy replaces the user's home path, the experiment and asset roots with `<HOME>`, `<EXP>`,
`<ASSETS>`, `<RERUN>`, `<SOUP_TREE>` and `<SNAPSHOTS>`, cloud app identifiers with `ap-<redacted>`, a GPU
device UUID with `GPU-<redacted>`, a cloud instance number with "cloud instance", and internal document names
with plain words; hashes recorded inside a
copied file refer to the unmodified original. Files that are a reading of larger raw files carry the SHA-256 of
those originals: `summary-sources.json` (for the four `summary-*.json`), `q32-train-rerun-vs-earlier.json`,
`attempts-with-a-guard-summary.json`, `laptop-reload-check.json` and `resident-control-memory.json`.
`weights-hash-checks.json` holds the output of each hash check as it was printed, and
`launcher-shim-lines-main-run.txt` the `SHIM` lines of the 300-step training log. `FREEZE-trimmed.json` is
the freeze file with three keys removed (they named unpublished documents) and its launcher field reworded.
`SCORE.json` is the scorer's output unchanged, apart from the plain-word replacements above.

In `rerun-2026-11-11/` the same kind of copy replaces the asset, shard-store and output folders with `<ASSETS>`,
`<CACHE>` and `<OUT>`, shortens the directory token of the kit's config files to `kit\`, turns the
box-drawing characters of the console panels into ASCII (two logs had been written through a console code
page and were converted back first), and drops two driver lines that print the lowest free host memory and
commit: the sampler's decimal comma made them wrong, and the figures in section 13 were computed again from
the host-memory files. `README.txt` in that folder lists every file and the column layouts. In the GPU files
`utilization.gpu` reads `[Unknown Error]` in 9 rows and `power.draw` holds impossible values (up to 1,804.68 W)
in 11 rows at the start of two runs; `memory.used` is intact in every row. The statements of section 13 that
no instrument file shows are in `operator-notes-2026-11-11.txt` and are marked as the operator's in the text.

## 13. Re-run through `soup train` on the same laptop, 2026-11-11 (a separate measurement, not a reproduction)

A separate measurement, made on 2026-11-11 (times UTC) on the same laptop and GPU as sections 5 and 6: RTX 5070
Laptop GPU, 8,151 MiB (the logs and the GPU files). The operator reports driver 616.92, Windows 11 Pro, Python
3.12.10, torch 2.14.0+cu130 and `soup_cli` 0.77.2; no file prints versions (`RF/rerun-2026-11-11/operator-notes-2026-11-11.txt`).
It ran `soup train`, not the experiment runner of sections 5 and 6, with other code, other data and another
sampler, so it does not re-run those numbers and the two sets are not a reproduction of each other. One run
per configuration (n = 1). Statements marked "operator" below are the operator's notes: no instrument file in
the tree shows them.

- **Code.** A local integration tree at commit **ee4cc5d2** (operator: clean working tree at the start of the
  window and at its end), made by merging the branches of pull requests #2628, #2653, #2636, #2655, #2658 and
  #2662 as they stood on 2026-11-11. All six are merged into `main` as of 2026-11-12; the versions that were
  merged were not re-tested here. The tree is not part of this record.
- **Command.** `soup train --config <yaml> --yes`, started through a small wrapper (`train_dump.py`) that runs
  `soup train` and then, before the process closes, runs one forward pass of the trained model on a fixed
  128-token input and saves the logits (`[train_dump] capture start` and `capture end` in the logs). The
  wrapper is not included; the config files are (`RF/rerun-2026-11-11/<config>.soup.yaml`).
- **Configuration as the logs and config files show it.** `task: sft`, `backend: transformers`, 4-bit, LoRA
  r = 8, alpha = 16, `lora.dropout: 0.0` (9,961,472 trainable parameters of 32,762,123,264 for Qwen3-32B;
  26,673,152 of 122,610,069,504 for Mistral Large, the counts of section 4), learning rate 2e-4 with a constant
  schedule, seed 1234, batch size 1, 3 training rows, 3 steps of 1 x 128 tokens (1 x 512 for the second
  Mistral run), layer streaming from the disk tier with read-ahead 1. Each run logged "Pre-quantised
  bitsandbytes NF4 base: imported byte for byte, not re-quantised" and "Additional writes before training:
  0.00 GB". The shard stores were built before the window from the published checkpoints: for Mistral Large
  `import-summary-mistral123.json` shows 2,645 tensors, 64,449,162,720 B, `import_verified` and
  `standalone_verified` true; for Qwen3-32B the operator reports 2,053 tensors and 19,211,023,168 B with the
  verified flags true (summary not copied). The counts equal those of section 2, which come from the earlier
  importer.
- **Conditions at each start** (operator, with `wait-checks-mistral123-512.txt` for the 512-token run; the
  driver logs show `nvidia-smi before: 0 MiB` and the free RAM before each run): AC power online, 0 MiB used on
  the GPU, at least 10 GiB of free host RAM.

| | Qwen3-32B, 128 tokens | Mistral Large, 128 tokens | Mistral Large, 512 tokens |
|---|---|---|---|
| result | completed, exit 0 | completed, exit 0 | completed, exit 0 |
| wall time, s | 86 | 149 | 147 |
| `train_runtime` in the log, s | 37.82 | 99.96 | 102.1 |
| step losses | 3.6200623512268066 / 4.209296703338623 / 3.682640552520752 | 3.204108953475952 / 3.5063159465789795 / 2.904374599456787 | 1.4975941181182861 / 1.4210911989212036 / 1.4317288398742676 |
| **whole-device peak, MiB** (1 s samples; number of samples in the file) | **3,042** (84) | **4,388** (146) | **6,998** (140) |
| earlier record, section 5, MiB | 3,002 | 4,598 | 5,602 |
| difference, MiB | +40 | -210 | +1,396 |
| torch peak allocated, `GPU peak` in the log's final panel, GiB (printed as GB) | 2.5 | 3.3 | 6.2 |
| earlier record, torch peak allocated, section 5, GiB | 2.59 | 3.50 | 4.63 |
| the trainer's own forecast of peak VRAM in the log, GB | 3.35 | 5.92 | 6.97 |
| page-locked host arena in the log, GB (decimal; 4.00, 3.00 and 3.00 GiB) | 4.29 | 3.22 | 3.22 |
| lowest sampled free host RAM, GiB (a sample about every 2.4 s) | 5.58 | 5.80 | 5.19 |
| lowest sampled free commit, GiB, of a 71.73 GiB limit | 34.89 | 32.21 | 27.45 |

Sources: `RF/rerun-2026-11-11/train-*.log`, `driver-*.log`, `gpu-*.csv`, `host-*.csv`, `step-losses.json`. The
losses are the exact values of each run's `checkpoint-3/trainer_state.json`. The peak is the largest
`memory.used` reading of the whole device taken once per second by a separate `nvidia-smi` sampler before
the end-of-training logits capture; the maximum over the whole run including the capture forward pass is the
same number in all three runs, and the sample counts are for the whole file. The earlier record's peaks
came from the guard's sampler of section 5, one reading per 1.1 to 1.7 s on average, so the two sets come
from different samplers as well as different code. No tolerance was agreed before the runs; the differences
are stated and not judged. `GPU peak` is `torch.cuda.max_memory_allocated` divided by 1024^3
(`src/soup_cli/monitoring/callback.py` at the commit above). The free-memory figures are the sampler's: Win32
kilobyte counts divided by 1,048,576, printed as GB, so GiB, and minima of samples, not of the whole run.

The earlier Q32 and Mistral runs used the earlier worker with its own fixed data; these runs used `soup train`
on three rows of the wrapper's own data, so the step losses are not comparable with sections 5 and 6 and are
not claimed to be. The second Mistral run's end-of-training capture ran at 128 tokens as well, because the
wrapper passes `--seq 128` to every configuration; training itself ran at 512 tokens (`num_tokens` 512 per
step). No Windows commit pre-flight message (pull request #2658) appeared in any of the three logs.

**Mistral Large, 128 tokens: one earlier attempt discarded.** The first attempt started at 14:21:58 and its
process was stopped by hand at about 14:26 (driver log: end 14:26:12, wall 253 s, exit code -1). The operator
reports that another GPU job's process was created at 14:22:36, during the run's first step, and appeared in
the GPU's process list next to the run's own process. The GPU samples show whole-device use of 2,732 MiB at
14:22:40, 4,230 MiB at 14:22:46 to 14:22:49, 7,159 MiB at 14:22:50, 7,863 MiB at 14:22:51 and 7,875 MiB at the
end, and the host-memory samples a lowest sampled free RAM of 0.37 GiB (the operator reads 0.8). Only the run's
own process was stopped (operator). After the stop 5,952 MiB remained in use on the device (driver log). The
split of the 7,875 MiB between this run and the other job at the peak is not known, so 7,875 MiB is not a
Mistral Large peak. The attempt printed no loss line and saved no adapter, and its step times were taken
under that load and are not used (`discarded-attempt-driver-mistral123-128.log`,
`discarded-attempt-gpu-mistral123-128.csv`, `discarded-attempt-host-mistral123-128.csv`). The row above is
the one retry, started at 14:27:57; the operator reports that its pre-run check read AC online, 0 MiB used and
11.4 GiB free, and that a watcher printed no other GPU process during it.

**Mistral Large, 512 tokens: +1,396 MiB over the earlier record, cause not determined.** The run's own torch
peak in the log's final panel is `GPU peak: 6.2/7.9 GB`, about 6.2 GiB allocated, against 4,968,217,088 B (4.63
GiB) allocated in the earlier M512 worker (`RF/earlier-recorded/laptop/summary-m512.json`): the increase is in
the run's own allocations, of the same order as the 1,396 MiB (1.36 GiB) of the whole-device difference.
Nothing in this record attributes it to another GPU job. The sampler's GPU memory rose with the run's own
phases (2,448 MiB at model load, a plateau at 4,140 MiB, 6,838 to 6,998 MiB in training;
`gpu-mistral123-512.csv`), the device read 0 MiB before the launch (driver log), and the trainer's own forecast
in the log was `peak VRAM ~6.97 GB at batch 1 x seq 512 (logits 0.23 GB)`. This is the only `soup train`
measurement at 512 tokens. The earlier 5,602 MiB came from a trimmed variant of the earlier worker, not from
`soup train`, and is not the `soup train` footprint. It is a single run, to be repeated on an idle GPU before
it is relied on. As a separate fact, other GPU jobs were active on the device around this window: a first
launch at 14:34 was refused by the pre-run check (operator: 114 MiB in use and another job's process created
14:34:01), and the wait file shows other jobs holding 6,132 to 6,580 MiB at the checks from 14:34:54 to 14:38:37
except one reading of 0 MiB at 14:38:05, then 0 MiB at 14:39:09 and 14:39:42, after which one run was launched
(14:39:46). The operator reports that the watcher printed
no other GPU process after the launch, but a process shorter than its 10 s interval cannot be excluded by it,
and gives the other jobs' process starts as 14:22:36, 14:34:01, 14:37:57, 14:45:38, about 14:57 and about 15:07.

### 13.1 Adapter reload: identical across fresh processes, NOT identical to the training process

For each of the three saved adapters the comparison script of the wrapper loaded the base and the adapter into
a fresh streamed inference process, ran the same 128-token input and compared the logits with the ones
captured from the training process before it closed. It reports a verdict by exit code.

| | Qwen3-32B | Mistral Large 128 | Mistral Large 512 |
|---|---|---|---|
| compare against the training process's capture | exit 2, `exact: false`: shape/dtype differ, bfloat16 (reload) against float32 (capture) | the same | the same |
| elements of the reload output that differ from the base model's | 17,094,803 of 19,447,808, largest absolute difference 1.2421875 | 4,048,414 of 4,194,304, 4.984375 | 4,065,893 of 4,194,304, 6.84375 |
| two-process fallback (run once): a fresh process dumps the reload logits, another fresh process reloads and compares | exact, 0 of 19,447,808 elements differ, top-1 equal at 128 of 128 positions | exact, 0 of 4,194,304, 128 of 128 | exact, 0 of 4,194,304, 128 of 128 |
| SHA-256 of the reload logits (first 8 hex digits), the same in the compare step and in both fallback processes | 302dfa81 | f8af8a1c | a121e5e8 |
| CPU comparison afterwards, saved live logits against the saved reload logits: live values exactly representable in bfloat16 | 19,447,808 of 19,447,808 | 4,194,304 of 4,194,304 | 4,194,304 of 4,194,304 |
| elements that still differ after the live logits are cast to bfloat16 | 16,402,544 (84.3%) | 2,955,854 (70.5%) | 3,043,686 (72.6%) |
| largest / mean absolute difference | 1.3125 / 0.052843 | 0.875 / 0.013165 | 0.625 / 0.014650 |
| top-1 token equal at | 126 of 128 positions | 127 of 128 | 127 of 128 |

Sources: `RF/rerun-2026-11-11/reload-compare-vs-live-*.json`, `reload-two-process-*.log`,
`live-vs-reload-posthoc-*.log`.

The adapter loads and changes the output (the first comparison shows it: a silently lost adapter is
excluded), and reloading is repeatable: three separate reload processes per adapter gave one logits hash.
The comparison against the training process was made, and the result is negative. The comparison script
itself stopped at its dtype guard and gave no element-wise verdict; the CPU comparison afterwards, made by a
separate script on the saved files and not an output of the comparison script, found the training process's
logits and the reloaded model's logits **not bit-identical**: 70.5% to 84.3% of the elements differ after the
cast, the largest absolute difference is 1.3125 (Qwen3-32B), 0.875 and 0.625 (Mistral Large), and the top-1
token differs at 2, 1 and 1 positions. The cause was not determined. The dtype is not the cause: the live
values are exactly representable in bfloat16. A different forward path or a different LoRA precision between
the live training model and the inference model are possibilities that were not investigated; the config sets
`lora.dropout: 0.0`, and whether the capture ran in eval mode is not recorded. Sections 5 and 9 compare a
reload with the earlier runner's own trained held-out losses and logits slices; that is a different comparison
from this one, and neither replaces the other.

### 13.2 Qwen3-32B reload-only check with the earlier runner, run once

The reload-only wrapper of the earlier runner for Qwen3-32B (the check that gave no result in section 6) was
run once on 2026-11-11. The operator reports that a first attempt at 14:46:16 was refused by its pre-run check
(6,102 MiB used on the device by another GPU job) and that waiting began at about 14:48; the wait file
(`wait-checks-earlier-runner-q32-reload.txt`) runs from 14:49:55 to 15:00:07 and shows the device free at the
last two checks. The second launch started at 15:00:10 with 0 MiB used and 11,181 MiB of host RAM available.
Result: the worker finished (`worker_status: passed`, `fresh_process_exact: true`, wrapper exit 0, 182.27 s
guard time); the sampled whole-device peak was 2,440 MiB over 126 samples, equal to the recorded 2,440 MiB (the
wrapper file also prints the recorded value), torch peak allocated 2,329,868,288 B and reserved 2,384,461,824 B
(both equal to the Q32 reload column of section 5), lowest available host RAM 5,328,457,728 B. As the
operator's notes quote the worker's result file (not copied here): tolerance `bit-exact`, held-out loss after
the three updates 4.298623561859131 for example 1 (section 6), adapter SHA-256 printed as `88aa4279...eb614b048`
(equal at both printed ends to the adapter hash of sections 5 and 6). It checks the adapter of the earlier
runner's train run `q32-n1` against that run's own recorded held-out loss and logits slices; it is a different
adapter and a different comparison from 13.1 and does not resolve the gap in 13.1. File:
`RF/rerun-2026-11-11/earlier-harness-q32-reload-check.txt` (an excerpt of the wrapper log).

### 13.3 Not measured

- **A repeat of any peak.** Each peak is one run. The 128-token Mistral peak comes from the retry after the
  contaminated first attempt; none was repeated on an idle device; the 512-token difference is not explained.
- **Step time.** The wall times and `train_runtime` values above are 3-step wall figures of `soup train`
  including start-up, not the audit-step timings of the earlier workers; they were not compared with sections
  5 and 6 and nothing is claimed about them.
- **The earlier runner's remaining reproduction steps.** Qwen3-32B train repeats 2 and 3 and the Mistral 128
  and 512 runs through the earlier wrappers (operator: its reproduction notes estimate about 45 and 50
  minutes for the Mistral ones) were not run. The wait for repeat 2 ran from 15:05:48 to 15:15:48 and gave up
  (`wait-checks-earlier-runner-q32-repeat2.txt`): at each of the 12 checks from 15:09:27 to 15:15:18 another
  GPU job held 6,606 to 7,148 MiB of the device and free host RAM was 5.9 to 6.8 GiB, below the wrapper's
  8 GiB stop line (operator). Repeat 3 and the Mistral runs were not attempted. No exclusive window of that
  length was observed: the operator gives other GPU jobs' process starts as 14:22:36, 14:34:01, 14:37:57,
  14:45:38, about 14:57 and about 15:07, each holding about 6 to 7 GiB.
- **Host RAM was the tight resource in these runs.** The lowest sampled free host RAM was 5.19 to 5.80 GiB
  (`host-*.csv`), with page-locked arenas of 4.00 and 3.00 GiB (4.29 and 3.22 GB as logged); the operator reports
  that another process on the machine used 3 to 6 GiB between the runs. Host memory was not the only limit
  near its edge: the 512-token run's whole device reached 6,998 of 8,151 MiB.
