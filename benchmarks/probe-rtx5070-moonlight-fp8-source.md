# Probe: does LoRA on an FP8-block source of Moonlight-16B-A3B train like the bf16 source? RTX 5070 Laptop 8 GB

Status: **FAIL (T2_max_step, T6_step_time)**, runs 2026-10-20 11:51Z-12:25Z. Both arms
COMPLETE; T1, T3, T4 and T5 pass. Sections 1-3 and appendix A are the rule and the setup as
committed in `62bc0331b` (branch `feat/moe-fp8-loss-check`, from `25c876f65`, then the head of
#2061) before either arm ran, and are unchanged; after #2061 merged (as `32af390c`, with one
later review commit) that commit was rebased onto main as `2efb4ca99`. `62bc0331b` was committed
locally before either arm ran and was never pushed; the only public copy is `2efb4ca99`, whose
committer date is 2026-10-20 14:04Z and which sits in PR #2066 (opened 14:06Z), both after the
runs, so "written before any run" cannot be checked from public history. Section 4 is the run log and
section 5 the results. Sections 6 and 7 are analysis done AFTER the verdict and are not part of
the rule: section 6 gives the most likely cause of T6, the D: drive's sustained-read plateau, and
section 7 discusses T2. Neither changes the verdict. No threshold was changed and no arm was
re-run.

## 1. Question

Does a LoRA run on the FP8-block source train like the same run on the bf16 source?

#2061 lets layer streaming shard a DeepSeek-V3 FP8 block-quantised checkpoint by dequantising
it at shard time. Layer streaming is Soup's implementation of layer-to-layer streaming of a
frozen base (L2L, Pudipeddi et al. 2020). The CPU gates of
[`gate-fp8-shard.md`](gate-fp8-shard.md) show that the dequant is exact (G1), that a streamed
model built from FP8 matches a resident model built from the dequantised weights bit for bit
(G2), and that the real 27-layer cache holds exactly the reference-dequantised tensors (G3).
That record leaves the GPU loss check to the maintainer: this is it.

The FP8 source is `D:/soup-bench/models/Moonlight-16B-A3B-fp8block`, which G3 made from the
bf16 `moonshotai/Moonlight-16B-A3B` with `benchmarks/harness/fp8_block_quantize.py`: every
decoder linear in `float8_e4m3fn` with a `weight_scale_inv` per 128 x 128 block, DeepSeek-V3's
`quantization_config`. **It is a real-format conversion of a real model, not an official FP8
release.** Its weights differ from the bf16 original by 2.65% (relative Frobenius error, every
tensor class, G3). No claim about Kimi K2, about any 1T model, or about convergence quality on
real data follows from this probe: 9 training rows show whether the two sources train alike,
nothing more.

## 2. Decision rule (written before any run)

### 2.1 Arms

| | A | B |
|---|---|---|
| `base` | `D:/soup-bench/models/Moonlight-16B-A3B` (bf16) | `D:/soup-bench/models/Moonlight-16B-A3B-fp8block` |
| config | `D:/soup-bench/fp8-loss-check/bf16/soup.yaml` | `D:/soup-bench/fp8-loss-check/fp8/soup.yaml` |
| `SOUP_LAYER_STREAM_CACHE_DIR` | `D:/soup-bench/soup-cache` (section 4.1's cache, reused) | `D:/soup-bench/fp8-loss-check/soup-cache` (new) |
| cache | written 2026-10-15 by section 4.1's run | written by this run: FP8 dequantised to bf16, then the same storage rules |

Common to both:
- the code: `src/` of `25c876f65`, imported through `PYTHONPATH`;
- the config, apart from `base` and `output` (section 3.1);
- the cache layout: fused, NF4 with double quantisation on the decoder linears the meta probe converts (attention, shared expert, layer 0's dense MLP), quantised on CUDA; the routed experts stay bf16 (in B, the dequantised FP8 values);
- the box: sequential, A first; AC power, lid open, no other GPU work, a measurement hold announced. B starts at least 10 minutes after A ends, with the GPU no more than 5 °C warmer than at A's start.

The run script (appendix A) refuses to start when any of these does not hold.

### 2.2 COMPLETES

An arm COMPLETES iff:
- `soup train` exits 0;
- `output/rewind.jsonl` has a finite loss for each of the 27 steps;
- the tracker holds 3 finite evaluation losses (after epochs 1, 2 and 3);
- `output/adapter_model.safetensors` is written, every tensor in it is finite, and at least one `lora_B` tensor is non-zero (the adapter trained).

### 2.3 Thresholds

Every loss is in nats. ΔL is B's loss minus A's at the same step, on the same row (batch 1, so
a step's loss is one row's mean token loss, read at full precision from `rewind.jsonl`).

| | statistic | limit | matched reference (4.2 vs 4.1) | failure it must catch, and its size |
|---|---|---|---|---|
| T1 | \|ΔL\| at step 1, the first logged step (row 1, 14 supervised tokens) | **0.40** | 0.121 | a wrong base: step 1 is a pure base forward (LoRA B is 0), and a dequant fault puts the blocks it touches at O(1) relative error, against 0.03-0.09 between the A and B weights (section 3.3) |
| T2 | max \|ΔL\| over the 27 steps | **1.6** | 0.818 (step 10, a 14-token row) | a B that does not learn: ≈ 5.0 (row 0 falls 5.0 nats in 4.1, rows 1, 4 and 6 fall 2.3-3.3) |
| T3 | \|ΔL\| of the evaluation loss after epoch 3 (1 held-out row) | **0.30** | 0.139 (3 evaluations: +0.070, -0.088, +0.139; RMS 0.103) | a B that does not learn: ≥ 0.56 (4.1's evaluation loss fell 4.036 → 3.472 from epoch 1 to 3, and an untrained B sits at or above epoch 1) |
| T4 | \|ΔL\| of the mean loss over epoch 3 (steps 19-27, each row once) | **0.15** | 0.0007 (the 3 epochs: -0.072, -0.045, -0.0007; RMS 0.049) | a B that does not learn: ≈ 2.1 (4.1's epoch means: 5.05 → 2.99) |
| T5 | GPU peak, B / A | **≤ 1.10** | | the shards have the same shapes, so the VRAM plan is the same |
| T6 | median seconds per step, B / A | **≤ 1.10** | | the same bytes are read per step from the same drive |

**Where the reference comes from.** Section 4.1 (routed experts bf16) and section 4.2 (routed
experts NF4) of [`probe-rtx5070-moonlight-16b.md`](probe-rtx5070-moonlight-16b.md) ran the
same config, the same 9 rows in the same order, the same LoRA seed and the same step count, on
two numerically different bases. Section 3.3 measures the weight distance of that pair and of
this A/B pair on the matrices one token passes through: **they are the same size (ratio
1.00)**. The 4.2-vs-4.1 differences are therefore the expected scale of the A/B differences,
from the same data, measured rather than assumed. Raw:
`benchmarks/results/probe-rtx5070/fp8-loss-check/reference-41-vs-42.json`.

**Why the limits are not noise-tight.** Equal distance is not equal sensitivity: the A/B
distance sits 92% in the attention and shared-expert linears (where the LoRA trains), the
4.2/4.1 distance 100% in the routed experts. The reference is also one realisation. So T1-T3
are 2-3.3 times their reference value, and T3 and T4, which have three reference values each,
about 3 times their spread (T3: 2.9 x RMS; T4: 3.1 x RMS, and twice the largest of the three
epoch-mean differences, because its epoch-3 value, 0.0007, is near zero). The compare script,
run on the 4.1/4.2 pair itself as if it were A and B, returns PASS (T1 0.121, T2 0.818, T3
0.139, T4 0.0007). A rule the matched reference fails would be measuring noise. Run-to-run noise of one arm is not measured here.
Step 1 is a fixed computation on fixed bytes, and A's step 1 is also compared with section
4.1's (same cache; reported, section 2.5). Every loss limit is at least 0.15 nats; GPU
nondeterminism is not expected to come near that in 27 steps, but this record does not
measure it.

**Why they are not trivially loose.** Each limit sits below the failure it targets by at least
a factor of 1.8 (T3: 0.30 against 0.56; T2: 1.6 against 5.0; T4: 0.15 against 2.1). Run with
B's losses frozen at their epoch-1 values (a B that does not learn), the compare script FAILs
T2 (4.90) and T4 (1.98).

T1 is the base-identity check. To first order, a step-1 loss moves in proportion to the
distance between the bases. The matched reference moved it 0.12 nats, so T1 trips at roughly
three times the A/B distance. A fault spread over the model, such as a skipped scale, a
scale/inverse confusion or FP8 bytes read as integers, is far beyond that. A fault in a few
blocks (a mis-cropped ragged edge of the 576-row `kv_a_proj_with_mqa`) may or may not trip it,
depending on where it sits. That is why the dequant's test is the bit-exact CPU gates of
`gate-fp8-shard.md`, not T1. T1 checks that the GPU path trains on bytes that behave like A's.

**What 2.65% predicts, and why it is not the figure the limits are built on.** FP8's own error
on the routed experts (bf16 after the dequant) is 2.65% in B, against 9.2% for NF4 on the
same experts in 4.2. That alone would predict a smaller A/B effect than the reference. The NF4
linears make up the difference. NF4 rounds each 64-value block to 16 levels scaled by the
block's largest value. FP8 moves every value by up to half an e4m3 step (G3: at most 3.6% of a
tensor's largest value), which shifts most blocks' largest value and moves values across level
boundaries. So B's NF4 weights are a fresh rounding of nearly the same matrix: about as far
from A's (0.091-0.093) as A's are from the bf16 weights (0.092-0.095). B itself is only
slightly further from the bf16 original than A is (0.096-0.099 against 0.092-0.095, the two
errors adding in quadrature). Section 3.3 has the table.

**T5 and T6.** T5 reads Soup's own GPU peak (`torch.cuda.max_memory_allocated`, printed in GiB
to 0.1 and stored per step in the tracker; 0.1 GiB is 3% of section 4.1's 3.1). B's sharding
quantises on the GPU too, but one weight at a time: the largest is layer 0's dense MLP
(11,264 x 2,048, 46 MB at bf16), far below the training peak. T6 is the median of the 26
intervals between the tracker's per-step timestamps (`logging_steps: 1`); the median skips the
two intervals that hold an end-of-epoch evaluation. B's cache is written just before it trains, so part of it may be in the page
cache. That can only make B faster: a T6 pass is weaker than it looks, but a fail cannot come
from it. The thermal side is held by the cool-down rule of section 2.1. The two logs' stream
plan lines (tier, buffers, pinned staging, read-ahead) must match. If they differ, T5 and T6
are INCONCLUSIVE; the loss checks still stand.

### 2.4 INCONCLUSIVE

The pair has no verdict when any of these holds:
- the RAM watchdog fired;
- a gap of more than 30 s appears between its 5 s samples (the laptop slept), any sample is off AC, or another process is listed on the GPU;
- the dataset fingerprints or the step-to-row orders of the two `rewind.jsonl` differ;
- A does not complete, whatever the cause;
- B does not complete for a cause outside the code under test: a driver reset, a full disk, or memory taken by another process (read from `train.log`). A B error from the sharder, the FP8 dequant, the streaming runtime or the trainer is a FAIL, not this;
- a pre-check of the run script was bypassed.

One re-run of the pair is allowed. The first attempt's files are moved into a subfolder (the
script refuses a used folder). The re-run's result stands.

### 2.5 Verdict

- **PASS:** both arms COMPLETE, and T1-T6 hold.
- **FAIL:** B does not complete for a code cause, or any of T1-T6 is broken. Each broken check is named.

`benchmarks/harness/fp8_loss_check_compare.py` applies sections 2.2-2.5 to the two arm folders.

Reported, with no threshold:
- both arms' 27 per-step losses, the three epoch means, and the evaluation losses after epochs 1 and 2;
- A's step-1 loss minus section 4.1's (4.712586): the same cache and the same first row, different code;
- B's shard wall time and cache bytes (the estimate is 30,598,555,900 B);
- the new cache's `index.json` (`quant_device` must read `cuda`);
- peak working set, `nvidia-smi` maximum memory and temperature, and each arm's wall time.

## 3. Setup (prepared 2026-10-19 20:00-21:00Z, CPU only)

Software, as G3 recorded it: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0,
bitsandbytes 0.50.2. The configs were loaded with Soup's loader and the cache checks run with
`CUDA_VISIBLE_DEVICES=-1`; nothing used the GPU. The box was on battery while this was
prepared; the run script refuses to start off AC.

### 3.1 Configs

Section 4.1's `D:/soup-bench/moonlight-train/soup.yaml`, with two fields added:
- `training.seed: 42`. Unset, `TrainingArguments` used 42 and the streamed LoRA init used 0 (`stream_setup.py`). Pinned, both use 42. The data order is 4.1's (0-based rows 1, 6, 7, 4, 5, 3, 0, 8, 2 in epoch 1). Step 1 does not depend on the LoRA init, because LoRA B starts at 0.
- `training.logging_steps: 1`, so the tracker holds every step.

The 10-row data file (9 training rows, 1 held out by `val_split: 0.1`, the last row) is copied
once to `D:/soup-bench/fp8-loss-check/data.jsonl` (sha256 `17264667...57f1`, identical to
4.1's). Both configs point at it, so they differ in `base` and `output` only. Both loaded with
`soup_cli.config.loader.load_config` from the `25c876f65` tree. The two loaded `SoupConfig`
dumps differ in exactly `base` and `output`. Against 4.1's config they differ in `data.train`,
`output`, `training.seed` and `training.logging_steps`. Defaults that matter, as loaded: LoRA
dropout 0.05, `init_strategy: random`, double quantisation on, `stream_expert_layout: fused`,
`rewind_log: true`, `warmup_ratio: 0.03`, cosine schedule.

The two models' `config.json` differ only in `quantization_config`. Their tokenizer files
(`tokenizer_config.json`, `tiktoken.model`, `tokenization_moonshot.py`) and
`generation_config.json` are byte-identical.

### 3.2 Which cache `soup train` uses, and whether it re-shards

From `trainer/stream_setup.py` and `utils/layer_shard.py` at `25c876f65`:
- **Folder.** The cache folder is `<root>/<model_slug(base)>`. The root is `SOUP_LAYER_STREAM_CACHE_DIR` when that lies under `$HOME`, the cwd or `$TMPDIR`; otherwise Soup **silently** falls back to `~/.soup/layer-stream` on C:. Hence cwd `D:/soup-bench`. The slug is made from the `base` string: `D___soup-bench__models__Moonlight-16B-A3B` and `D___soup-bench__models__Moonlight-16B-A3B-fp8block`.
- **Reuse.** `inspect_shard_cache` reuses a folder only if every one of these matches:
  - dtype, quantisation, the double-quant flag and the quantisation device kind;
  - the source fingerprint: name, size and mtime_ns of each checkpoint file the index names, plus `config.json` for `deepseek_v3`;
  - the external-tensor mode, the expert layout and format version 5;
  - the stripe roots;
  - the presence of every file.

  It does not compare bytes, the NF4 suffix set or the Soup version that wrote the cache.
- **`soup train` quantises on the training device**, `quant_device=str(self.device)`, so on this box it asks for `cuda`.

`benchmarks/harness/fp8_loss_check_cache.py` asks the same question with the same arguments,
read-only. Its answers (JSON in `results/probe-rtx5070/fp8-loss-check/cache-*-before.json`):

- **A reuses `D:/soup-bench/soup-cache`.** Verdict "cache is reusable"; fingerprint `7a63224f…`, 28 source files, 31 cache files, 30,598,520,590 B, `quant_device: cuda`, format 5. Section 4.1's run wrote it on 2026-10-15 at `2846155470` (branch `feat/moe-moonlight` before its merge; not an ancestor of `25c876f65`). The FP8 diff since then changes nothing for a bf16 source: with no `quantization_config` the plan is `None`, every tensor goes through the same `_read_tensor` call, and the storage rule (`_layer_tensor_storage`) is the old expression moved into a function. Section 3.3 adds a check on the bytes: every tensor not derived from an FP8 weight is byte-identical between this cache and the one this branch wrote from the FP8 copy. The run script re-checks reuse before A and checks again after it that the cache is still reusable. A only reads the folder.
- **`D:/soup-bench/fp8-shard-27L-nf4` (G3's cache) cannot be reused by `soup train`, under any `SOUP_LAYER_STREAM_CACHE_DIR`.** Two reasons:
  - Its folder is not a `<root>/<slug>` path, so no root makes `soup train` look there.
  - Its index records `quant_device: cpu`, because G3 quantised on the CPU. With its fingerprint (`874138d7…`, the same as this run computes for the FP8 copy), `inspect_shard_cache` answers "quantization device changed ('cpu' -> 'cuda')" for a CUDA caller, and "reusable" only for a CPU one.

  Moving it to `<root>/D___soup-bench__models__Moonlight-16B-A3B-fp8block` would make B re-shard in place, adding about 0 new bytes, but it would overwrite G3's evidence. That is not done here; the call is the owner's.
- **B shards into the new root `D:/soup-bench/fp8-loss-check/soup-cache`.** The sharder's own estimate is 30,598,555,900 B; G3's CPU-quantised cache of the same copy came to 30,598,520,564 B. A re-shard writes one file at a time through a temporary file, so the transient overhead is at most one layer (1.12 GB).

**New bytes on D: for both runs: about 30.67 GB.** That is:
- B's cache, 30.60 GB;
- two output folders, about 34 MB each (section 4.1's was 34,322,735 B: adapter, `checkpoint-27` with the optimiser state, tokenizer files);
- logs, watchdog samples and tracker databases, under 1 MB.

D: had 101,162,680,320 B free at preparation, so about 70.5 GB stays free. The run script
refuses B unless the estimate plus 10 GB fits.

### 3.3 How far apart the two arms' weights are (CPU, read-only)

`benchmarks/harness/fp8_loss_check_weights.py` read decoder layers 1, 13 and 26, and routed
experts 0, 21, 42 and 63, from four sources:
- the bf16 source;
- A's cache;
- G3's FP8-sourced NF4 cache (CPU-quantised; CUDA quantisation agrees on the packed nibbles and absmax and differs only in the float32 nested statistic, `layer_shard._default_quant_device`);
- section 4.2's per-expert cache (E).

NF4 tensors were dequantised with bitsandbytes on the CPU. Relative Frobenius distances, mean
over the three layers (raw: `results/probe-rtx5070/fp8-loss-check/weights.json`):

| tensor class | A vs bf16 | B vs bf16 | A vs B |
|---|---:|---:|---:|
| `q_proj` (NF4) | 0.0923 | 0.0963 | 0.0919 |
| `kv_a_proj_with_mqa` (NF4) | 0.0927 | 0.0967 | 0.0920 |
| `kv_b_proj` (NF4) | 0.0926 | 0.0967 | 0.0922 |
| `o_proj` (NF4) | 0.0920 | 0.0961 | 0.0920 |
| shared `gate_proj` / `up_proj` / `down_proj` (NF4) | 0.0927 / 0.0929 / 0.0949 | 0.0967 / 0.0969 / 0.0988 | 0.0909 / 0.0911 / 0.0915 |
| routed `gate_up` / `down` (bf16) | 0 (A equals the source) | 0.0265 / 0.0265 | 0.0265 / 0.0265 |

- Section 4.2's NF4 routed experts sit at 0.0920 (`gate_up`) and 0.0921 (`down`) from the bf16 weights.
- E's 21 NF4 linears equal A's byte for byte, so 4.1 and 4.2 differed only in the routed experts.
- Every other tensor of the three layers equals its counterpart byte for byte between A's cache and the FP8-sourced one: 18 tensors (norms, router weight, correction bias), plus the extras (`model.norm.weight` and both NF4 code tables).
- **Per-token distance**: the squared Frobenius distance summed over what one token passes through in the three layers (4 attention, 3 shared-expert and, at the sampled experts' mean, 6 routed expert matrices). B vs A: 996.8, of which 914.9 (92%) is in the NF4 linears. E vs A: 990.5, all in the routed experts. **Amplitude ratio 1.003.** This weights every matrix equally per unit of squared error, which is the assumption section 2.3 does not lean on beyond the order of magnitude.

### 3.4 Checks of the harness itself

- The cache check on A answered "reusable" and on B "shard folder does not exist" (exit 0 with the expected answer each). It created nothing; `D:/soup-bench/fp8-loss-check` held only the two configs and the data file afterwards.
- The watchdog was run against two dummy processes named like a training run. It followed one to its exit (5 samples, summary written, exit 0). It stopped the other at a forced 1,000 GB floor (`fired: true`, the process gone, exit 3).
- The compare script was run on folders built from sections 4.1 and 4.2, with synthetic timestamps and GPU strings for T5 and T6. It returned PASS for the reference pair (section 2.3). With B's losses frozen at epoch 1, it returned FAIL on T2 and T4.
- The run script parses (PowerShell 7.6 parser, 0 errors). Its exit-code capture through `Tee-Object` was checked on a process exiting 3.

## 4. Run log (2026-10-20, UTC)

Both arms ran on AC power, under a measurement hold, through `fp8_loss_check_run.ps1` (appendix
A), A first. The script's pre-checks passed for both arms:
- `src/` of the `soup-wt/moe-fp8` worktree equal to `25c876f65`, nothing untracked under it;
- `soup_cli` imported from `C:\Users\user\projects\soup-wt\moe-fp8\src`;
- at least 10 GB of RAM available, on AC, no GPU compute process.

The cool-down rule held: B's process started 10 min 38 s after A's ended, with the GPU at 42 °C,
as at A's start. Raw: `benchmarks/results/probe-rtx5070/fp8-loss-check/`.

| | A (bf16 source) | B (FP8-block source) |
|---|---|---|
| process (watchdog window) | 11:51:00-11:59:42 | 12:10:20-12:24:47 |
| run id | `run_20261020_165102_5938d145` | `run_20261020_171022_5424b88c` |
| layer cache | reused `D:/soup-bench/soup-cache`: 31 files, 30,598,520,590 B, `quant_device: cuda`; the folder's listing digest (`d878adad…`) was the same before and after | re-sharded into `D:/soup-bench/fp8-loss-check/soup-cache`: "Re-sharding layer cache: cache index is missing or unreadable.", files written 12:10:41-12:12:17 (96.5 s), 31 files, 30,598,520,565 B (estimate 30,598,555,900 B), `quant_device: cuda`, reusable afterwards |
| FP8 notice in the log | none | "FP8 block-quantised base (128x128 blocks): dequantised to bfloat16 as the layer cache is written" |
| training began (`rewind.jsonl` header) | 11:51:49 | 12:12:34 |
| step 1 / step 27 logged (tracker) | 11:52:10.6 / 11:59:23.8 | 12:13:14.3 / 12:24:28.1 |
| `train_runtime` | 464.4 s | 725.5 s |
| stream plan line | "27 layers, streamed from DISK (30.60 GB on an NVMe volume) by an async reader, read_ahead=2, 3632 MB pinned host staging (4.29 GB page-locked), 2 x 1159 MB decoder buffers + 1 x 671 MB large-layer slot" | identical |
| D: free before / after | 101,162,680,320 / 101,128,101,888 B | 101,128,122,368 / 70,494,965,760 B |
| watchdog samples | 104 | 173 |
| minimum free RAM | 3.419 GB | 5.351 GB |
| largest tree RSS / peak working set | 7.146 / 7.174 GB | 7.166 / 7.194 GB |
| largest `nvidia-smi` memory used | 3,994 MiB | 3,994 MiB |
| hottest GPU temperature | 50 °C | 52 °C |
| watchdog fired / sampling gaps / samples off AC / other GPU processes | no / 0 / 0 / none | no / 0 / 0 / none |

**New bytes on D:** 30,667,714,560 B over both runs (101,162,680,320 B free before A,
70,494,965,760 B after B), against section 3.2's estimate of about 30.67 GB.

The console capture has one defect: PowerShell decoded `soup train`'s UTF-8 output through the
console's OEM code page (cp866), so `train.log` holds the box drawing as mojibake. The committed
`train-a.log` and `train-b.log` are re-decoded (cp866 → UTF-8), with box borders and progress
bars dropped. The `verdict.json` copy keeps only the plain "Layer streaming ready" line of
`stream_plan_lines`; everything else in it is as `fp8_loss_check_compare.py` wrote it.

## 5. Results

**Both arms COMPLETE.** Each exited 0 and logged a finite loss at all 27 steps and three
finite evaluation losses. Each wrote `adapter_model.safetensors` (7,134,912 B, 216 tensors, all
finite, `lora_B` non-zero). The dataset fingerprints (`84c087f8…`) and the step-to-row orders
are identical, and no condition of section 2.4 holds.

| | statistic | value | limit | |
|---|---|---:|---:|---|
| T1 | \|ΔL\| at step 1 | 0.0814 | 0.40 | pass |
| T2 | max \|ΔL\| over 27 steps | 3.2655 (step 7) | 1.6 | **FAIL** |
| T3 | \|ΔL\| of the evaluation after epoch 3 | 0.0721 (A 3.5660, B 3.4939) | 0.30 | pass |
| T4 | \|ΔL\| of the epoch-3 mean | 0.1010 | 0.15 | pass |
| T5 | GPU peak, B / A | 1.00 (3.1 / 3.1 GiB) | ≤ 1.10 | pass |
| T6 | median seconds per step, B / A | 1.909 (A 13.124 s, B 25.058 s) | ≤ 1.10 | **FAIL** |

The compare script printed the verdict line:

```
FAIL (T2_max_step, T6_step_time)
```

Reported, with no threshold (section 2.5):
- **A's step-1 loss minus section 4.1's: 0.0.** It is 4.712586402893066 in both.
- **B's shard:** 96.5 s; the cache is 30,598,520,565 B and its `index.json` reads `quant_device: cuda`.
- **Wall time:** A 8 min 42 s and B 14 min 27 s as watchdog windows; B's includes the shard.
- **Peaks:** see section 4.

Per step: losses from `rewind.jsonl`, and the interval since the previous step's log entry in
each arm's tracker. Steps 10 and 19 carry the end-of-epoch evaluation. Raw: `steps.json`.

| step | row | tokens | A loss | B loss | ΔL | A interval s | B interval s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 14 | 4.7126 | 4.6312 | -0.0814 | | |
| 2 | 6 | 11 | 6.1370 | 5.7089 | -0.4281 | 12.684 | 22.372 |
| 3 | 7 | 17 | 5.1476 | 4.7102 | -0.4374 | 12.667 | 21.756 |
| 4 | 4 | 11 | 6.4743 | 5.0898 | -1.3845 | 12.675 | 22.030 |
| 5 | 5 | 39 | 2.5789 | 2.4871 | -0.0918 | 12.837 | 22.184 |
| 6 | 3 | 54 | 1.9471 | 1.9478 | +0.0007 | 12.882 | 22.331 |
| 7 | 0 | 5 | 13.3655 | 10.1000 | -3.2655 | 13.015 | 22.954 |
| 8 | 8 | 50 | 1.4611 | 1.3732 | -0.0879 | 12.904 | 22.407 |
| 9 | 2 | 21 | 5.3738 | 5.3983 | +0.0245 | 12.872 | 22.244 |
| 10 | 1 | 14 | 4.0283 | 2.7735 | -1.2548 | 20.308 | 36.110 |
| 11 | 6 | 11 | 3.6503 | 3.4920 | -0.1583 | 12.842 | 24.303 |
| 12 | 0 | 5 | 8.2903 | 8.7763 | +0.4860 | 12.839 | 25.009 |
| 13 | 7 | 17 | 3.4764 | 3.6071 | +0.1306 | 12.939 | 25.108 |
| 14 | 5 | 39 | 1.7902 | 1.8890 | +0.0988 | 12.882 | 24.859 |
| 15 | 2 | 21 | 4.8216 | 4.6327 | -0.1889 | 12.799 | 24.980 |
| 16 | 4 | 11 | 3.4582 | 3.7394 | +0.2813 | 13.233 | 25.071 |
| 17 | 3 | 54 | 1.4136 | 1.5150 | +0.1014 | 19.828 | 25.282 |
| 18 | 8 | 50 | 1.0888 | 1.0700 | -0.0188 | 17.203 | 25.045 |
| 19 | 3 | 54 | 1.3469 | 1.5008 | +0.1539 | 27.086 | 39.939 |
| 20 | 5 | 39 | 1.6067 | 1.7244 | +0.1176 | 21.018 | 25.176 |
| 21 | 6 | 11 | 2.9913 | 2.8720 | -0.1193 | 21.149 | 25.618 |
| 22 | 0 | 5 | 6.5275 | 7.6556 | +1.1281 | 21.122 | 25.440 |
| 23 | 8 | 50 | 1.0300 | 1.0104 | -0.0196 | 21.066 | 33.691 |
| 24 | 2 | 21 | 4.1548 | 4.2905 | +0.1356 | 21.204 | 25.300 |
| 25 | 1 | 14 | 3.0285 | 2.3029 | -0.7256 | 21.013 | 25.499 |
| 26 | 4 | 11 | 3.1388 | 3.6456 | +0.5068 | 21.166 | 33.989 |
| 27 | 7 | 17 | 3.3620 | 3.0939 | -0.2682 | 20.986 | 25.133 |

| | epoch 1 mean | epoch 2 mean | epoch 3 mean | eval after epoch 1 | eval after epoch 2 | eval after epoch 3 |
|---|---:|---:|---:|---:|---:|---:|
| A | 5.2442 | 3.5575 | 3.0207 | 4.2493 | 3.5632 | 3.5660 |
| B | 4.6052 | 3.4994 | 3.1218 | 4.1399 | 3.5809 | 3.4939 |
| ΔL | -0.6391 | -0.0581 | +0.1010 | -0.1093 | +0.0177 | -0.0721 |

## 6. T6: what made B slower (analysis after the verdict, not part of the rule)

Done 2026-10-20 12:43Z-13:00Z, read-only and on the CPU: cache headers, the reader's code, and
a short read benchmark. Nothing on D: was modified.

**1. The cache layout is identical.** `benchmarks/harness/fp8_loss_check_layout.py` compared all
30 safetensors files of A's cache with B's (output: `t6-layout.txt`):
- header length, data start, file size, metadata, tensor order, and every tensor's dtype, shape, length and absolute offset: no difference in any file;
- for example `layer_010.safetensors`: a 3,752-byte header and 1,123,596,780 bytes in both, with the fused experts at offsets ≡ 3436 (mod 4096) and ≡ 364 (mod 512) in both;
- `index.json`: equal in every field except `source_files` and `source_fingerprint`, including `quant_specs`, `layer_keys`, format version 5 and `quant_device: cuda`.

`fsutil file queryextents` reports a single NTFS extent for every one of the 60 files
(`t6-extents.txt`), so neither cache is fragmented.

**2. The reader did not fall back.** `AsyncDiskSource` (`utils/async_disk_source.py`) opens
`FILE_FLAG_NO_BUFFERING` handles (`safetensors_reader.open_direct`). It decides once per drive
per run, by opening the first layer file, and the read spans are 4 KiB-aligned supersets of the
tensors, so tensor offsets do not matter to it. When the volume refuses direct I/O, or the
staging is unaligned, the console says so: "Layer streaming reads <root> through the page
cache: direct I/O is unavailable there (…)", or "the disk tier reads through the page cache
instead of direct I/O". Neither `train.log` has either line. A grep for direct, unbuffered,
buffered, aligned and fallback finds only the disk-tier notice and a transformers
token-alignment message, the same in both logs, and both print the identical plan line.

**3. On a rested drive, B's files read as fast as A's.** `fp8_loss_check_readbench.py` read
`layer_010` and `layer_020` (1,123,596,780 B each) through `open_direct` in 4 ranges, as the
reader does, alternating A and B. The box was quiet (no pytest and no GPU job; 10.4 GB free at
the start), 12:56-13:00Z, about 30 minutes after B ended; 11.2 GB read in all
(`t6-readbench.txt`):

| read | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | then, fresh process | 1 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arm, layer | A 10 | B 10 | A 10 | B 10 | A 10 | B 10 | B 20 | A 20 | | B 10 | A 10 |
| GB/s | 2.61 | 5.29 | 4.82 | 5.39 | 4.82 | 5.36 | 5.28 | 4.92 | | 2.70 | 4.52 |

The first read of each process pays the buffer's first touch, whichever arm it reads (2.61 A,
2.70 B). Leaving those out, A reads at 4.52-4.92 GB/s and B at 5.28-5.39 GB/s.

**The step is read-bound, and the drive has a sustained-read plateau.** During the steps the
GPU sat idle: utilisation 0% in 44% (A) and 57% (B) of the watchdog's 5-s samples, 45-52 °C.

The two drives on this box (Samsung PM9B1, `MZAL81T0HFLB-00BL2`) were seen to drop from about
4.0 GB/s to a flat 2.43-2.59 GB/s plateau about two minutes into sustained unbuffered reading,
and to hold it to the end of the arm. In an arm that followed another arm straight away, the
plateau began within 17 s. A third level, 1.86-2.04 GB/s, once held for a whole arm.
The sources:
- [`probe-rtx5070-two-drive-sustained.md`](probe-rtx5070-two-drive-sustained.md);
- [`probe-rtx5070-drive-throttle.md`](probe-rtx5070-drive-throttle.md). It holds the hypothesis that a drive thermally throttles, most likely D:, as not established; the drives' temperatures cannot be read without elevation.

The runs match that:
- **A slowed at step 17.** It took 12.67-13.23 s per step for steps 2-16 (the epoch-1 evaluation aside), then 21.0-21.2 s from step 20 to the end, with 19.8, 17.2 and 27.1 s (the epoch-2 evaluation) in between. The ratio is 1.64, about the 4.0 → 2.5 GB/s step.
- **B started on the plateau.** Its training began 17 s after its own shard ended: 16.6 GB read and 30.6 GB written on D: in 96.5 s. B took 21.8-23.0 s per step from step 2, and 24.3-25.6 s from step 11 onward (33.7 and 34.0 s at steps 23 and 26, the evaluations aside).
- The bytes per step were not measured, so these ratios only fit the recorded drive levels; they do not measure them.

**Conclusion.**
- **The most likely cause of the T6 gap is the drive's state, not the FP8 path.** The layout is identical, the reader took the direct path in both arms, and on a rested drive B's files read at A's rate or faster.
- **No product bug:** caches from the current sharder and from FP8 sources stream like the old ones.
- **The rule had a gap: its cool-down covered the GPU, not the drive.** It asked for 10 minutes after A and a GPU within 5 °C, but put B's 47 GB of shard I/O on D: inside B's own run, 17 s before its timed steps.

**What this does not show.** It does not show that B trains at A's speed. No training run on a
rested drive was compared: the benchmark shows that B's files read as fast as A's, not that B's
steps run as fast. T6 stays FAIL.

## 7. T2: per-step divergence on short rows (analysis after the verdict)

Step 7 is the one breach: 3.2655 against the limit of 1.6, on row 0, the shortest row (5
supervised tokens). Row 0 comes up at steps 7, 12 and 22; steps 12 and 22 are below the limit:

| step | A | B | ΔL |
|---:|---:|---:|---:|
| 7 | 13.3655 | 10.1000 | -3.2655 |
| 12 | 8.2903 | 8.7763 | +0.4860 |
| 22 | 6.5275 | 7.6556 | +1.1281 |

The next largest per-step gaps after step 7 also fall on short rows, all under the limit but
above the matched reference's maximum of 0.818: -1.3845 at step 4 (row 4, 11 tokens), -1.2548 at
step 10 (row 1, 14 tokens) and +1.1281 at step 22 (row 0, above). Step 12, at +0.4860, is below
0.818. Against that:
- step 1, a pure base forward, agrees to 0.081;
- the evaluation after epoch 3 agrees to 0.072;
- the epoch-3 mean agrees to 0.101.

All three are within their limits. The epoch-1 mean differs by 0.64 (reported; it has no limit).

Two facts noticed after the verdict:
- **Step 2 is also a pure base forward.** The scheduler's first step has learning rate 0, so A's step-2 loss equals section 4.1's exactly, like its step 1. The base-only differences between A and B are therefore -0.081 (row 1, 14 tokens) and -0.428 (row 6, 11 tokens), against +0.121 and -0.114 for the 4.1/4.2 reference.
- **For scale only:** A and section 4.1 stream the same cache bytes and differ in the LoRA-A seed (42 against 0) and the code. They still differ by 1.62 at step 7 (row 0), and by at most 0.60 elsewhere.

**Neither fact changes T2.** The record does not show that the two sources train alike step by
step: per-step losses on short rows diverge beyond the registered bound. No threshold was
changed after the fact and no arm was re-run.

## 8. What this shows, and what it does not

Shown:
- the `soup train` path from an FP8 block-quantised DeepSeek-V3 checkpoint runs end to end on an 8 GB card: dequantised at shard time, fused NF4, streamed from disk by layer streaming (Soup's implementation of layer-to-layer streaming, L2L, Pudipeddi et al. 2020);
- it uses the same GPU peak as the bf16 source (3.1 GiB);
- its final evaluation loss differs from the bf16 run's by 0.072, its step-1 loss by 0.081 and its epoch-3 mean by 0.101, each within its registered limit.

Not shown:
- speed parity (T6 failed, and the rested-drive comparison of section 9 has not been run);
- per-step parity (T2 failed);
- anything about an official FP8 release, Kimi K2 or any 1T model. The FP8 source is this project's own conversion of a real bf16 model into the real format, and 9 training rows say nothing about quality.

## 9. Future work (not done)

A re-run of T6 with the drive's state controlled, pre-registered as its own rule before it runs:
- **Order.** Interleave B, A, A, B. B's cache already exists, so no timed run shards anything.
- **Rest.** D: rests for at least 30 minutes before the first run and between runs.
- **Logging.** The watchdog logs D:'s per-second read rate (a per-disk read-bytes counter), so a step time can be split into bytes and rate.
- **Order effect.** The difference between the first and second run of each arm is a check of its own.

The run script should also set the console's output encoding to UTF-8, so `train.log` keeps
Rich's box drawing (section 4).

## Appendix A. Commands

From a PowerShell 7 window, AC on, the measurement hold announced. Run A first; B no sooner
than 10 minutes after A ends:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File C:\Users\user\projects\soup-wt\moe-fp8\benchmarks\harness\fp8_loss_check_run.ps1 -Arm bf16
pwsh -NoProfile -ExecutionPolicy Bypass -File C:\Users\user\projects\soup-wt\moe-fp8\benchmarks\harness\fp8_loss_check_run.ps1 -Arm fp8
```

Then the verdict (CPU):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:PYTHONPATH = 'C:\Users\user\projects\soup-wt\moe-fp8\src'
python C:\Users\user\projects\soup-wt\moe-fp8\benchmarks\harness\fp8_loss_check_compare.py --arm-a D:\soup-bench\fp8-loss-check\bf16 --arm-b D:\soup-bench\fp8-loss-check\fp8 --json D:\soup-bench\fp8-loss-check\verdict.json
```

What the script runs for arm A, after its pre-checks (arm B: `fp8` folder, the FP8 `base`,
`SOUP_LAYER_STREAM_CACHE_DIR=D:/soup-bench/fp8-loss-check/soup-cache`):

```powershell
Set-Location D:\soup-bench
$env:PYTHONPATH = 'C:\Users\user\projects\soup-wt\moe-fp8\src'
$env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:COLUMNS = '160'
$env:SOUP_LAYER_STREAM_STRIPE_DIRS = $null; $env:CUDA_VISIBLE_DEVICES = $null
$env:SOUP_LAYER_STREAM_CACHE_DIR = 'D:/soup-bench/soup-cache'
$env:SOUP_DB_PATH = 'D:\soup-bench\fp8-loss-check\bf16\experiments.db'
python -m soup_cli.cli train --config D:\soup-bench\fp8-loss-check\bf16\soup.yaml --trust-remote-code --yes 2>&1 | Tee-Object -FilePath D:\soup-bench\fp8-loss-check\bf16\train.log
"exit=$LASTEXITCODE" | Tee-Object -FilePath D:\soup-bench\fp8-loss-check\bf16\train.log -Append
```

The watchdog the script opens in its own window (for B, `fp8` in both paths). It polls
`psutil` and `nvidia-smi` every 5 s into `<arm>\watchdog.jsonl`, stops the training process
tree if available RAM falls below 1.5 GB, and writes `<arm>\watchdog.json` when the run ends:

```powershell
python C:\Users\user\projects\soup-wt\moe-fp8\benchmarks\harness\fp8_loss_check_watchdog.py --match 'D:\soup-bench\fp8-loss-check\bf16\soup.yaml' --out 'D:\soup-bench\fp8-loss-check\bf16' --min-free-gb 1.5
```

The pre-checks, in order:
- `src/` equals `25c876f65` (`git diff --quiet`, nothing untracked), and `soup_cli` imports from the worktree;
- at least 10 GB of RAM available, on AC, no GPU compute process;
- for B: at least 10 minutes since A's log was last written, the GPU at most 5 °C warmer than at A's start, and room on D: for the cache plus 10 GB;
- the cache check: A must reuse, B must re-shard.

After the run, the cache check runs again and must answer "reusable" for both arms.

**Expected wall time.**
- **A: about 7.5 min.** Section 4.1's run took 8 min 44 s from start to end: 15 s of setup, 75 s writing the cache on CUDA (08:37:10-08:38:25Z), then 425 s from the first step to the end, which includes 27 steps at about 15.7 s and three evaluations. A reuses the cache, so it skips the 75 s.
- **B: about 9 min.** The same, plus 75-95 s to shard (75 s for the bf16 source on CUDA in 4.1; 93 s for the FP8 copy with CPU quantisation in G3).
- **The whole session:** about 30 minutes, including the cool-down and the compare.
