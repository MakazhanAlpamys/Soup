<!--
Measurement record: all 20 cells ran. Published as measured; nothing here is rounded in the direction of a conclusion.

Hardware: 1x RTX 4090 24 GB (24,564 MiB), rented, 2026-11-08.
  AMD Ryzen 9 7900X (12 cores / 24 threads), 61.9 GB host RAM, Ubuntu container (glibc 2.39), kernel 6.8.0-147,
  driver 595.99.02 (CUDA 13.2), power limit 450 W (max 600 W).
Stack: Python 3.12.3, torch 2.11.0+cu128, transformers 5.17.0, trl 0.29.1, peft 0.20.0, bitsandbytes 0.50.2,
  accelerate 1.15.0, datasets 4.8.5, triton 3.6.0, numpy 2.5.4, liger-kernel 0.8.5 (arm D only); pip check clean.
Soup: soup-cli 0.77.2 from PyPI (not editable). No file under src/soup_cli was changed.
Harness: benchmarks/harness/kbit_fp32_ab/ (re-runnable; hashes in its README.md).
Rule: written before the first launch and not edited afterwards; sha256
  edb551d2a99070568724d6587ab32b1ae632f3b134b074618e844875f4ae0350 (section 4). The file is kept outside the repository.
Raw output: benchmarks/results/kbit_fp32_ab/ (this record's numbers). The eight Chrome traces and the analysis.json
  made from them are kept outside the repository.
-->

# Gate record: the fp32 upcast of resident 4-bit SFT, an A/B on one RTX 4090 (#2609, step 1)

**Status: MEASURED (2026-11-08, one rented RTX 4090; 4 arms x 3 rounds of 105-step timing cells and 4 arms x 2
rounds of 9-step profiled cells; all 20 cells ran, none ran out of memory, none was void).** **Verdict on
"`prepare_model_for_kbit_training`'s fp32 cast is the cause": PARTIAL, against the 70% bar fixed before the first run.
Casting `embed_tokens`, `lm_head` and the norms back to bf16 (arm B), or never casting them (arm C), removes 63.7% (B)
and 63.6% (C) of the issue's 818 ms elementwise gap, cuts the unprofiled step by 24.9% (2.2562 s to 1.6949 s) and the
allocator peak by 22.4% (17,425 to 13,520 MiB), and stays inside the pre-registered loss tolerance on all three seeds.
It does not touch the three fp32 logits tensors (7,128 MiB), and 395 ms of eager elementwise work is still there.** The
bar was not moved.

- **Arm A reproduces the issue's profile on this box.** Unprofiled median step 2.2562 s (1,815 tok/s; the issue:
  2.270 s, 1,804 tok/s), peak allocated 17,425 MiB (the issue: 17,425), elementwise bucket 915.8 ms (916),
  elementwise ops with an fp32 input 837.0 ms (837), cross-entropy and logits-side kernels 80.1 ms (80),
  `embed_tokens` and `lm_head` float32 at 2,079 MiB each, 57 norms float32. Profiler overhead +0.4% (the issue: +0.4%).
- **Arms B and C are the same result.** Median step 1.6949 s and 1.6950 s (2,417 and 2,416 tok/s, 1.33x A), identical
  across the three rounds to 0.1%; peak allocated 13,520 MiB in both; every profile category agrees to 0.1 ms. B and C
  differ only in the biases, which bitsandbytes casts to the input dtype at the first forward anyway (the end-of-run
  dtype census shows 84 bf16 biases in all three arms).
- **What the dtype change removed of the issue's numbers.** Of the 818 ms elementwise gap: 520.6 ms (63.7%). Of the 837 ms
  on ops with an fp32 input: 729.8 ms (87.2%; 107.2 ms is left, mostly on `[..,3584]` tensors). GPU busy time: -563.6 ms of
  2,236.2 ms. Of the 7,128 MiB of three fp32 `[8,512,152064]` logits tensors: **0 MiB** (they are 52.7% of B's and C's
  peak). Of the issue's 10,264 MiB gap to the reference: 3,905 MiB (38.0%).
- **What is left after B or C.** 395.2 ms of elementwise work still running as eager PyTorch ops (the SwiGLU `silu * up` on
  `[8,512,18944]` alone is 214.6 ms; norm and residual 123.1 ms; casts 23.5 ms; RoPE and q/k/v layout 21.6 ms; the
  reference spends 98 ms on the same work, the issue's figure), 76.6 ms in the logits-side kernels, and the three fp32
  logits tensors. B's step is 1.26x the issue's reference figure (1.345 s) where A's is 1.68x; that reference figure is the
  issue's, it was not re-run here.
- **Arm D (`training.use_liger: true`) is the only arm that touches the logits tensors, and on its own it is not a
  speed-up.** The Liger modules were verified in the model (28 `LigerSwiGLUMLP`, 57 `LigerRMSNorm`,
  `Qwen2ForCausalLM.forward` = `lce_forward`). Peak allocated 10,852 MiB (-6,573 MiB, -37.7%; the largest live block at
  the peak is 296 MiB, the three logits tensors are absent), but the step is only -1.8% (2.2161 s, 1,848 tok/s). Of the 818 ms
  it removes 210.9 ms (25.8%); the fp32 residual stream is still there (633.5 ms of elementwise on fp32 inputs), and 223.4 ms
  of `aten::copy_` sits in its cross-entropy bucket. B plus D was not run.
- **Power and clocks.** B and C ran at 444-445 W median and sat on the 450 W software power cap in 76-89% of their busy
  samples (SM clock 2,700 MHz); A and D ran at 410-414 W with the cap hit in at most 5% (2,715-2,730 MHz). The clock difference
  is 0.6-1.1%, so the speed-up is not a clock artefact, but it was measured under a 450 W cap; the issue's box had 400 W,
  where B and C would be capped harder. That was not measured.

Unit convention: peak memory is in MiB (`torch.cuda.max_memory_allocated / 2^20`; "GiB" in the text is 1,024 MiB). tok/s is
4,096 tokens per step divided by the median step time unless marked "wall". A spread is (max - min) / median over the rounds
of one arm. "ms" is GPU time per optimizer step. The issue gives the elementwise gap as 916 - 98 = 818 ms. Shares of the gap in this
record use 817.8 ms (915.6 - 97.8, the unrounded figures behind those two numbers, held in `ISSUE` in
`benchmarks/harness/kbit_fp32_ab/p6_post.py`); with the rounded 818 ms, B's share is 63.6% instead of 63.7% and no other figure
moves by more than 0.1.

---

## 1. Why this exists

Issue #2609 reads the 1.69x step-time gap and the 2.4x peak-memory gap between Soup and a reference run on a resident 4-bit
SFT task (Qwen2.5-7B NF4, LoRA r16, RTX 4090) as fp32 elementwise traffic plus fp32 full-vocabulary logits, and names PEFT's
`prepare_model_for_kbit_training` (which casts every non-quantised fp16/bf16 parameter to fp32 in peft 0.20.0) as the most
likely source. The issue says that cause was read from source and from tensor dtypes, and that no run had yet changed a
dtype. Step 1 of its proposal is this A/B. Steps 2 to 4 (a helper, a loss path, fusion) depend on how much of the 818 ms
and the 7,128 MiB it removes. This record is step 1 only; nothing under `src/soup_cli` was changed.

## 2. Setup

| item | value |
|---|---|
| card | RTX 4090, 24,564 MiB, compute capability 8.9 (`box/preflight.txt`) |
| driver | 595.99.02, CUDA 13.2 (`box/env.txt`) |
| power limit | 450 W, max 600 W (`box/env.txt`); the issue's box: 400 W |
| clocks | max SM clock 3,105 MHz; measured medians per arm in section 6 |
| PCIe | gen 4 x16 under load, 26.84 GB/s pinned host-to-device (`box/preflight.txt`; the idle read in `box/env.txt` is gen 1) |
| host | AMD Ryzen 9 7900X, 12 cores / 24 threads, 61.9 GB RAM, kernel 6.8.0-147, glibc 2.39, `ulimit -l` 8,192 KB |
| stack | Python 3.12.3, torch 2.11.0+cu128, transformers 5.17.0, trl 0.29.1, peft 0.20.0, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 4.8.5, triton 3.6.0, numpy 2.5.4, typer 0.27.3, click 8.4.2, soup-cli 0.77.2 (`box/freeze-with-liger.txt`); liger-kernel 0.8.5 for arm D only (`box/liger-install.txt`); `pip check` clean (`box/pip-check-with-liger.txt`); no unsloth, xformers or torchao in the environment (`requirements-tco.txt`) |
| model | `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28` (`box/model-download.txt`) |
| data | the 840-row UltraChat window (`HuggingFaceH4/ultrachat_200k`, `train_sft`, first rows in stored order whose templated length is at least 512 tokens and whose first user turn leaves at least 128 tokens for the answer), every row exactly 512 tokens, sha256 `eba8a5bc6074eced5adb255f3f8872d942779978171421e13f9c670494479c8a`, equal to the hash the job script expected (`box/data-manifest.json`, `box/data-sha-check.txt`) |
| task | `soup train --config soup.yaml --yes`, `task: sft`, `quantization: 4bit`, LoRA r16 alpha 32 dropout 0 on q, k, v, o, gate, up, down, batch 8 x 512 (exactly 4,096 tokens per step in all 1,260 steps of the 12 timing cells, no padding), gradient accumulation 1, bf16, gradient checkpointing on, `adamw_torch`, lr 2e-4 constant, `max_grad_norm` 1, loss on all tokens (`runs/A-r1-T/soup.yaml`, `runs/D-r1-T/soup.yaml`) |
| run window | launch 11:40:53Z, job 11:42:33Z to 12:42:35Z (`box/job_wrapper.log`), results pulled 12:46:59Z, machine destroyed 12:47:29Z (`launch-events.jsonl`) |
| set-up | 335 s for torch and the pinned stack (`box/setup.json`), 5 s for liger-kernel, 126 s for the 15 GB model download, 13.5 s to build the data window; 20 cells of 48-59 s (profiled) and 190-252 s (timing) (`box/cell_times.tsv`) |

All paths in the table are under `benchmarks/results/kbit_fp32_ab/`.

## 3. Arms

| arm | what differs from A | how (`benchmarks/harness/kbit_fp32_ab/p6_arms.py` and one yaml line) |
|---|---|---|
| A | nothing: soup-cli 0.77.2, default | not patched |
| B | `embed_tokens`, `lm_head` and every norm weight back to bfloat16 right after `prepare_model_for_kbit_training` | `peft.prepare_model_for_kbit_training` is wrapped: the real function runs, then 59 float32 parameters (1 `embed_tokens`, 1 `lm_head`, 57 norms) are cast to bf16 and the CUDA cache is emptied. `prepare_calls: 1` in every B cell (`runs/B-r1-P/arm_info.json`) |
| C | the function replaced by: freeze the base, `enable_input_require_grads()`, `gradient_checkpointing_enable()`; bf16 kept, no cast | the same attribute replaced; the gating (loaded in 4/8 bit, `use_gradient_checkpointing`, `use_reentrant` default) is copied from peft 0.20.0 `utils/other.py`. `prepare_calls: 1` in every C cell |
| D | A plus `training.use_liger: true` | one line in the generated yaml. Verified in the model: 28 `LigerSwiGLUMLP`, 57 `LigerRMSNorm`, `Qwen2ForCausalLM.forward` is `liger_kernel.transformers.model.qwen2.lce_forward` (`runs/D-r1-P/arm_info.json`) |

The batch order is Soup's own for a given seed and is the same in every arm of a seed: the per-step token fingerprint column
of `steps.jsonl` is equal across A, B, C and D in all three rounds.

## 4. The rule, written before the first run

The rule file was written on 2026-11-08 and begins "Written 2026-11-08, BEFORE the first launch." Its last write was at
11:34Z, its sha256 was taken at 11:37:33Z, and the first launch event is at 11:38:15Z (`launch-events.jsonl`). The sha256 of the
whole file is
`edb551d2a99070568724d6587ab32b1ae632f3b134b074618e844875f4ae0350`. The file is kept outside the repository (it also holds
notes that are not published); the parts that decide the outcome are quoted here word for word. The one sentence of
the loss section that is not quoted says that A's own profiled-cell versus timing-cell loss on steps 1-9 and the spread of
each arm over seeds are reported but are not part of pass or fail.

Loss equivalence, "compared against arm A of the SAME seed, from the T cells' `log.jsonl` (one logged loss per optimizer step)":

> 1. step-1 loss: |X - A| / A <= 0.5 % (LoRA B = 0 at step 1, so this isolates the numerics of the frozen forward);
> 2. mean of the last 16 logged steps (steps 90-105): |X - A| / A <= 1.0 %, for each seed AND for the mean over the three seeds.
>
> An arm that misses either bound is reported "outside the tolerance" with the number; the bound is not moved afterwards.

Verdict rule for "the cause", with "baseline numbers from the issue: elementwise/cast bucket 916 ms vs 98 ms for the
reference (gap 818 ms), step 2,270 ms unprofiled, peak 17,425 MiB":

> * confirmed: B or C removes >= 70 % of the 818 ms gap in the elementwise bucket (>= 573 ms) AND the unprofiled median step falls by >= 400 ms AND the arm is inside the loss tolerance;
> * refuted: B and C each remove < 25 % of the gap (< 205 ms) of the elementwise bucket;
> * partial: anything in between, or the bucket falls but the step does not (or the loss tolerance is missed).

Memory prediction: "the dtype change cannot touch the three fp32 `[8,512,152064]` logits tensors (7,128 MiB), so the prediction
written here is a peak reduction of at most about 3.9 GiB for B and C (fp32 embed+head 2,079 MiB, the bf16 copy of the fp32
head made by autocast 1,040 MiB, the 28 saved layer inputs 784 MiB), and D is the only arm expected to touch the logits
tensors."

## 5. Method

Cell P (profile, rounds 1 and 2): 5 warm-up steps; `torch.profiler` (CPU and CUDA, `record_shapes=True`) over steps 6 to 8;
one step (9) with `torch.cuda.memory._record_memory_history(stacks="python")`; stop. Every GPU event is joined by correlation
id to its CPU launch and to the aten op that launched it (op name, input dims, input dtypes) and bucketed by kernel name and
shape (`analyze.py`); no GPU event was left without a launch link in any of the eight cells (`missing_launch_link` is 0 in
`analysis.json`), and one event in one cell (A-r2-P) has no launching-op link. Cell T (timing, rounds 1 to 3): no profiler, 5 warm-up
and 100 measured steps over the 840 rows; the statistic is the median of `t_end - t_begin` over steps 6 to 105 (the meter
synchronises the GPU at both ends). The SM clock, power and temperature are the harness poller's samples (about two per second)
restricted to that window; a second sampler (`nvidia-smi -l 2`) recorded the throttle-reason bitmask for the whole job
(`throttle.txt`). Rounds are seeds 3407, 3408 and 3409. Inside a round the order is A, B, C, D, each as P then T, so the timing
cells are interleaved A B C D A B C D A B C D. Peak allocated and reserved are the maxima over the 105 steps of each T cell
(`steps.jsonl`); the P cells' own meter peak (7,710 MiB) is void because the profiler resets the peak counter at every phase
boundary, so the memory decomposition uses the profiler's per-phase peaks instead.

## 6. Timing

### Per arm (median of the three rounds' medians; range = (max - min) / median across rounds)

| arm | what | median step s (steps 6-105) | tok/s | range across rounds | vs A | SM clock median MHz (by round) | power median W (by round) | peak allocated MiB | peak reserved MiB (by round) |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| A | default | 2.2562 | 1,815 | 0.09% | | 2,715, 2,730, 2,730 | 413, 410, 411 | 17,425 | 21,592, 21,592, 21,592 |
| B | bf16 cast-back | 1.6949 | 2,417 | 0.11% | -24.88% (-561.3 ms) | 2,700, 2,700, 2,700 | 445, 445, 445 | 13,520 | 17,942, 17,942, 17,942 |
| C | no kbit prepare cast | 1.6950 | 2,416 | 0.13% | -24.87% (-561.2 ms) | 2,700, 2,700, 2,700 | 445, 444, 445 | 13,520 | 17,996, 17,940, 17,940 |
| D | `use_liger` | 2.2161 | 1,848 | 0.02% | -1.78% (-40.1 ms) | 2,730, 2,730, 2,730 | 410, 413, 414 | 10,852 | 11,944, 11,944, 11,944 |

Source: `timing.json` and `tables.md` (T2), from `runs/<cell>/{steps.jsonl,train_begin.json,gpu_samples.json,result.json}`; every
T cell has 105 steps, 4,096 tokens per step and status `complete`. I recomputed the medians, peaks and tok/s of all twelve
cells directly from `steps.jsonl` with a separate script before writing this table; they agree. The second half of the measured
window runs within 0.1% of the first in every cell. The spread of a median across the three rounds is at most 0.13% for every
arm, so differences of 1.8% and larger are outside the run-to-run spread of this box, and the difference between B and C
(0.13 ms, 0.008%, in the medians) is inside it.

### Per round

| arm | round (seed) | median step s | tok/s | tok/s wall | SM clock median / min MHz | power median / max W | temp max C | share of busy samples at the software power cap |
|---|---|---:|---:|---:|---|---|---:|---:|
| A | r1 (3407) | 2.2582 | 1,814 | 1,807 | 2,715 / 2,520 | 413 / 421 | 78 | 0.03 |
| B | r1 (3407) | 1.6965 | 2,414 | 2,403 | 2,700 / 2,520 | 445 / 452 | 79 | 0.84 |
| C | r1 (3407) | 1.6965 | 2,414 | 2,404 | 2,700 / 2,505 | 445 / 452 | 78 | 0.89 |
| D | r1 (3407) | 2.2163 | 1,848 | 1,840 | 2,730 / 2,715 | 410 / 429 | 74 | none |
| A | r2 (3408) | 2.2562 | 1,815 | 1,809 | 2,730 / 2,520 | 410 / 417 | 77 | 0.05 |
| B | r2 (3408) | 1.6946 | 2,417 | 2,406 | 2,700 / 2,535 | 445 / 452 | 79 | 0.85 |
| C | r2 (3408) | 1.6944 | 2,417 | 2,406 | 2,700 / 2,520 | 444 / 453 | 79 | 0.84 |
| D | r2 (3408) | 2.2161 | 1,848 | 1,840 | 2,730 / 2,715 | 413 / 430 | 74 | none |
| A | r3 (3409) | 2.2562 | 1,815 | 1,809 | 2,730 / 2,520 | 411 / 418 | 76 | 0.02 |
| B | r3 (3409) | 1.6949 | 2,417 | 2,405 | 2,700 / 2,520 | 445 / 452 | 78 | 0.87 |
| C | r3 (3409) | 1.6950 | 2,416 | 2,406 | 2,700 / 2,520 | 445 / 451 | 78 | 0.76 |
| D | r3 (3409) | 2.2158 | 1,849 | 1,840 | 2,730 / 2,715 | 414 / 430 | 74 | 0.01 |

Source: `tables.md` (T1) and `throttle.txt` (busy = power above 300 W inside the cell; the share is the fraction of busy
samples whose throttle-reason bitmask has the software power cap bit; the other reasons did not occur). The "min" clock is the
lowest sample in the window, not the median.

## 7. What each arm removed (profile cells, ms per step)

Mean of the two profiled cells of an arm (`profile.json`, from `analysis.json`). The two cells of an arm agree to 0.18% in wall
time and 0.41% in any category above 5 ms.

| category | A | B | C | D |
|---|---:|---:|---:|---:|
| base-weight GEMM | 929.6 | 935.9 | 935.8 | 928.3 |
| LoRA GEMM | 83.7 | 89.2 | 89.2 | 83.6 |
| lm_head GEMM | 50.7 | 51.0 | 50.9 | 84.9 |
| NF4 dequant | 52.5 | 52.8 | 52.8 | 52.5 |
| attention | 22.4 | 22.4 | 22.4 | 22.4 |
| cross-entropy + logits-side | 80.1 | 76.6 | 76.6 | 225.4 |
| RMSNorm / RoPE / SwiGLU / elementwise / casts | 915.8 | 395.2 | 395.3 | 704.8 |
| optimizer + clip | 2.0 | 2.0 | 2.0 | 2.0 |
| memcpy / memset (device to device) | 99.6 | 47.7 | 47.7 | 92.0 |
| **GPU busy** | **2,236.2** | **1,672.6** | **1,672.6** | **2,195.8** |
| step wall (profiled) | 2,269.5 | 1,705.0 | 1,704.9 | 2,227.5 |
| of which elementwise on an fp32 input | 837.0 | 107.2 | 107.3 | 633.5 |
| of which elementwise with no fp32 input | 78.8 | 288.0 | 288.0 | 71.4 |
| Soup rewind recorder (per-row loss) | 9.9 | 9.9 | 9.9 | 0.0 |

A here against the issue's Soup column (the issue's figures in brackets): base-weight GEMM 929.6 (933), LoRA GEMM 83.7 (83),
lm_head GEMM 50.7 (51), NF4 dequant 52.5 (53), attention 22.4 (22), cross-entropy and logits-side 80.1 (80), elementwise 915.8
(916), optimizer 2.0 (2), device-to-device copies 99.6 (99.5), GPU busy 2,236.2 (2,240), step wall 2,269.5 (2,278).

| arm | elementwise bucket ms | removed vs A | share of the issue's 818 ms gap | fp32-input elementwise ms | removed | GPU busy removed ms | peak allocated MiB | removed MiB | the three fp32 logits tensors (7,128 MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | 915.8 | 0.0 | 0% | 837.0 | 0.0 | 0.0 | 17,425 | 0 | present |
| B | 395.2 | 520.6 | 63.7% | 107.2 | 729.8 | 563.6 | 13,520 | 3,905 | present (0 MiB removed) |
| C | 395.3 | 520.5 | 63.6% | 107.3 | 729.7 | 563.7 | 13,520 | 3,905 | present (0 MiB removed) |
| D | 704.8 | 210.9 | 25.8% | 633.5 | 203.5 | 40.4 | 10,852 | 6,573 | absent from the peak snapshot |

Where B's and C's 563.6 ms of GPU busy time came from: elementwise -520.6 ms, device-to-device copies -51.9 ms, logits-side
-3.5 ms, and +6.3 ms (base-weight GEMM), +5.5 ms (LoRA GEMM), +0.3 ms (lm_head), +0.3 ms (NF4 dequant): the matmuls got very
slightly slower with bf16 inputs and everything else got faster.

What is left in B's 395.2 ms (`profile.json`, `elementwise_split` of `B-r1-P`; the other cell of B agrees to 0.1 ms):

| what | shape | input dtype | ms |
|---|---|---|---:|
| SwiGLU `silu * up` | `[..,18944]` | bf16 | 214.6 |
| norm and residual | `[..,3584]` | fp32 (HF's RMSNorm upcasts inside) | 81.4 |
| norm and residual | `[..,3584]` | bf16 | 41.7 |
| casts | `[..,3584]` | fp32 | 23.5 |
| RoPE and q/k/v layout | `[..,H,128]` | bf16 | 21.6 |
| everything else | | | 12.4 |

Arm D's cross-entropy bucket is 225.4 ms against 80.1 ms for A: `aten::copy_` 223.4 ms and the Liger kernel itself 2.0 ms
(`profile.json`, `ce_split` of `D-r1-P`). That the copies are the fp32 head being cast, twice per call, is read from op shapes
in `analysis.json` (not in the repository) and was not tested. The lm_head GEMM grows from 50.7 ms to 84.9 ms under D. Soup's
rewind recorder switches itself off under Liger ("Rewind recorder disabled: the model returned no logits",
`box/d-arm-worker-log-excerpt.txt`), which is the 9.9 ms row.

Profiler overhead and agreement with the timing cells (`tables.md`, P4): unprofiled steps 2-5 against profiled steps 6-8
differ by +0.32% to +0.55% in every profiled cell, and the unprofiled mean of steps 2-5 is within 0.23% of the matching timing
cell's median.

## 8. Memory

Parameter bytes by group at trainer construction, first profiled cell of each arm (`profile.json`, `param_groups`; MiB):

| group | A | B | C | D |
|---|---:|---:|---:|---:|
| `embed_tokens` | float32 2,079.0 | bfloat16 1,039.5 | bfloat16 1,039.5 | float32 2,079.0 |
| `lm_head` | float32 2,079.0 | bfloat16 1,039.5 | bfloat16 1,039.5 | float32 2,079.0 |
| norms (57) | float32 0.8 | bfloat16 0.4 | bfloat16 0.4 | float32 0.8 |
| biases (84) | float32 0.5 | float32 0.5 | bfloat16 0.2 | float32 0.5 |

Per-phase allocator peaks of the profiled cells (MiB, maximum over steps 3-9) and the allocator snapshot of step 9, replayed to
the allocated-bytes peak:

| arm (cell) | pre_step | fwd_body | head_ce | backward | step peak | blocks of 1,000 MiB or more at the snapshot peak |
|---|---:|---:|---:|---:|---:|---|
| A-r1-P, A-r2-P | 7,633 | 10,350 | 17,421 | 17,425 | 17,425 | 2,376, 2,376, 2,376, 1,040 |
| B-r1-P, B-r2-P | 5,552 | 6,984 | 13,516 | 13,520 | 13,520 | 2,376, 2,376, 2,376 |
| C-r1-P, C-r2-P | 5,552 | 6,984 | 13,516 | 13,520 | 13,520 | 2,376, 2,376, 2,376 |
| D-r1-P, D-r2-P | 7,633 | 10,350 | 10,447 | 10,852 | 10,852 | none |

Decomposition at the allocated-bytes peak (`profile.json`, `snapshot`): **A**: resident 7,633 + three fp32 logits-sized blocks
7,128 (one allocated in `F.cross_entropy`, two with no Python frame) + the bf16 copy of the fp32 head 1,040 (`linear.py:134`)
+ 28 saved layer inputs of 56 MiB 1,568 + the embedding output 56 = 17,425. **B and C**: resident 5,552 + the same three
logits blocks 7,128 + the saved layer inputs and the embedding output 840 (27 blocks of 28 MiB, one of 56, one of 28) = 13,520.
**D**: the three logits blocks are gone (the largest live block is 296 MiB; by allocating stack, 4 blocks with no Python frame
total 592.7 MiB, 2 blocks from `bnb.py:552` 592 MiB, 27 layer outputs 1,512 MiB, which are fp32 as in A).

The memory prediction of section 4 held: B and C came out 3,905 MiB (3.81 GiB) lower, against "at most about 3.9 GiB", and only D
touched the logits tensors. Reserved memory fell by 3,650 MiB (B, 16.9%) and 9,648 MiB (D, 44.7%).

Against the issue's reference peak of 7,161 MiB (not re-run here): A is 2.43x, B 1.89x, D 1.52x.

## 9. Loss equivalence

Per seed (step-1 loss / mean of logged steps 90-105), from `runs/<cell>/log.jsonl` of the timing cells:

| arm | seed 3407 | seed 3408 | seed 3409 |
|---|---|---|---|
| A | 1.5053 / 1.0931 | 1.3934 / 1.0187 | 1.4804 / 1.0690 |
| B | 1.5050 / 1.0939 | 1.3908 / 1.0191 | 1.4790 / 1.0688 |
| C | 1.5050 / 1.0936 | 1.3908 / 1.0193 | 1.4790 / 1.0689 |
| D | 1.5051 / 1.0932 | 1.3922 / 1.0190 | 1.4785 / 1.0687 |

Differences against A of the same seed (the tolerance of section 4: step-1 0.5%; last-16 1.0% per seed and in the mean of the
three seeds):

| arm | step-1 delta by seed | largest abs step-1 delta | last-16 delta by seed | mean last-16 delta | inside tolerance |
|---|---|---:|---|---:|---|
| B | -0.017%, -0.185%, -0.093% | 0.185% | +0.075%, +0.033%, -0.017% | +0.030% | yes |
| C | -0.017%, -0.185%, -0.093% | 0.185% | +0.046%, +0.052%, -0.013% | +0.029% | yes |
| D | -0.011%, -0.086%, -0.123% | 0.123% | +0.017%, +0.023%, -0.028% | +0.004% | yes |

Source: `loss.json` and `tables.md` (L1), from `runs/<cell>/{log.jsonl,steps.jsonl}`. The batches are identical to A's in every
seed and arm (the `fp` column of `steps.jsonl`). The non-determinism floor (the same arm and seed, profiled cell against timing
cell, steps 1-9) is 0.000% at step 1 and at most 0.198% over the first nine steps (`tables.md`), so the step-1 differences of B and
C (up to 0.185%) are of the size of that floor, and the last-16 differences (up to 0.075%) are well inside the tolerance. Held-out
evaluation was not run; this is the training loss on the same batches.

## 10. Verdict

| condition (section 4) | threshold | B | C | met |
|---|---|---|---|---|
| elementwise gap removed | at least 573 ms (70%) | 520.6 ms (63.7%) | 520.5 ms (63.6%) | **no**, short by 52 ms |
| unprofiled median step falls | at least 400 ms | 561.3 ms | 561.2 ms | yes |
| inside the loss tolerance | step-1 0.5%, last-16 1.0% | 0.185%, 0.075% | 0.185%, 0.052% | yes |
| refuted (each removes under 205 ms) | | no | no | not refuted |

**PARTIAL.** The first condition fails and the refutation bound is far away. The bar is not moved after the fact.

What the numbers support, in plain terms: the fp32 cast in `prepare_model_for_kbit_training` is the dominant source of the fp32
elementwise traffic (837 ms to 107 ms on ops with an fp32 input; what is left of it sits on `[..,3584]` tensors, which fits
RMSNorm's own upcast, not tested), of 3.8 GiB of the peak, and of a 24.9% step-time loss, and the loss stays inside the tolerance
without it. It is not the whole gap. 395 ms of eager elementwise work, 77 ms in the logits-side kernels and 7,128 MiB of fp32
logits tensors are outside the reach of a dtype change. Arm D shows the logits tensors can be taken out of the peak (-6,573
MiB) but that, with the fp32 head still in place, it gains 1.8% in time. The ceiling from the issue (2,278 - 818 - 77 = 1,383 ms) was
not reached: B and C stand at 1,705 ms in the profile (1,695 ms unprofiled), which is 564.5 ms of the 886.5 ms (63.7%) that
separate A's 2,269.5 ms from that ceiling.

Inference, not measured here: the untested combination of the bf16 parameters with `use_liger` may avoid the head casts; fused
SwiGLU, RMSNorm and RoPE kernels are what a reference would run for the remaining 395 ms.

## 11. Incidents and deviations

1. **The first rented machine failed and was destroyed before any job ran.** It was launched at 11:38:15Z, was
   armed at 11:38:51Z, and was destroyed at 11:39:13Z with the reason "payload delivery failed" (`launch-events.jsonl`, instance 1).
   A second machine was then rented (instance 2 in the same file); its payload
   was delivered at 11:42:11Z and the job started at 11:42:33Z.
2. **Stack against the issue's box.** The pins are the same (torch, transformers, trl, peft, bitsandbytes, accelerate, soup-cli,
   driver), but this environment has no unsloth, xformers or torchao (`requirements-tco.txt`), the card's power limit is 450 W
   (the issue's: 400 W), and the host CPU of the issue's box is not stated in the issue.
3. Three timing rounds were run instead of two; profiled cells only in rounds 1 and 2 (20 cells in `box/cell_times.tsv`).
4. `prepare-data` exited with code 134 (a pyarrow abort at interpreter exit, `box/data-prepare.txt`) after writing the data file
   and its manifest; the file's sha256 matches the expected one and every cell used it.
5. The harness pre-flight printed `disk_free_gb_ok: false` (66.1 GB free against its 80 GB bar) and so `valid_host: false`
   (`box/preflight.txt`); no cell was affected (the largest write was the 15 GB model cache).
6. The wall-clock tok/s of the profiled cells (about 512-612) includes the profiler, the allocator-history step and the stop; it is
   not comparable and is not used. The meter's peak in those cells (7,710 MiB) is void (section 5).
7. liger-kernel prints two `ImportError` lines at import for optional backends (`_cutedsl`, `_ascend`); the default Triton
   backend was used and the module swap was verified (section 3).

## 12. Not measured

Against the first two acceptance criteria of the issue: the method was followed (same model, config and data in every arm, 5
warm-up steps, a 3-step profile, an unprofiled median over 100 steps, three interleaved repeats, SM clock beside every tok/s,
peak allocated and reserved, the category table) and the loss tolerance was written down before the run and checked over three
seeds. Three of the four listed arms were run; the arm "bf16 plus the chosen loss path" was not, because no loss path has been
chosen. The rest of the list follows.

- The reference itself. The reference was not re-run on this box; every comparison with it uses the issue's numbers (1.345 s,
  7,161 MiB, 98 ms elementwise).
- **The bf16 parameters together with a loss path** (the issue's third arm): B or C plus `use_liger`, and a token-chunked head and
  loss on the resident path. Only D alone was run.
- Steps 2 to 4 of the proposal: a shared helper and a scanner test for the 15 direct calls, a default change, the pre-flight
  estimate (`utils/vram_estimate.py`: `KBIT_UNQUANTIZED_BYTES`, `KBIT_LOGITS_EXTRA_BYTES`) against the measured 13,520 MiB, the
  scope note in `docs/performance-and-quantization.md`, resuming from a checkpoint written before a change, a CPU test of the helper.
- A power limit of 400 W (this box: 450 W), any other card (this one is compute capability 8.9, like the issue's), another host CPU,
  locked clocks.
- 8-bit; tied-embedding models; other model families and sizes; other batch sizes and sequence lengths; gradient checkpointing off;
  fp16-only devices (where the fp32 cast is needed); multi-GPU; the 14 other call sites of the same function (the vision and audio
  SFT paths and the other trainers); `fast_lora`; layer streaming.
- Held-out evaluation loss; more than three seeds.
- The fusion work of step 4 (SwiGLU, RMSNorm plus residual, RoPE): only the size of what is left, 395 ms, was measured.
- Whether the 223 ms of `copy_` under arm D are the head casts (read from op shapes, not tested).

## 13. Files

Under `benchmarks/results/kbit_fp32_ab/`:

| path | holds |
|---|---|
| `tables.md` | the tables above as generated by `p6_post.py` (T1, T2, L1, P1-P4) |
| `timing.json`, `loss.json`, `profile.json` | the numbers behind sections 6 to 9 |
| `throttle.txt` | throttle-reason share per timing cell (`throttle_summary.py` over `box/smi_throttle.csv`) |
| `launch-events.jsonl` | the launch, destroy and pull events of the two rented machines (numbered 1 and 2; owner, offer and price fields removed) |
| `runs/<cell>/` | per cell (`A-r1-T`, `A-r1-P`, ...): `steps.jsonl`, `log.jsonl`, `train_begin.json`, `arm_info.json`, `soup.yaml`; timing cells also `result.json` and `gpu_samples.json`; profiled cells also `prof/prof_meta.json` |
| `box/` | job-level files: `env.txt`, `preflight.txt`, `setup.json`, `model-download.txt`, `data-*`, `cell_times.tsv`, `freeze-with-liger.txt`, `liger-install.txt`, `smi_throttle.csv`, `job_wrapper.log`, `PULLED.json` (sha256 of the result archive), `d-arm-worker-log-excerpt.txt` |

Kept outside the repository: the eight Chrome traces (`prof/trace.json.gz`), `analysis.json` (475,886 bytes, sha256
`edd5b2a42a7e95414e85e257160aff79ca2e7c4b4591f6046d9f5ca5edb170cc`), the per-cell worker logs and the rule file. With
`analysis.json` and the files above, `p6_post.py` reproduces `timing.json`, `loss.json`, `profile.json` and `tables.md` byte for
byte, except that the repository's end-of-file hook added a final newline to each JSON file (and trimmed trailing spaces in
`box/env.txt`).
