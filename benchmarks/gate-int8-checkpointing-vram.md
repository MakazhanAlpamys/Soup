# Gate record: 8-bit loading with gradient checkpointing, resident peak VRAM

**Hardware.** NVIDIA RTX PRO 5000 Blackwell (sm_120; torch reports 50.75 GB), driver
580.105.08, Linux. torch 2.11.0+cu130, transformers 5.16.1, trl 0.29.1, peft 0.20.0,
bitsandbytes 0.50.2, accelerate 1.14.0, Python 3.13. The card was shared with other
processes, so every row ran under `torch.cuda.set_per_process_memory_fraction(0.24)`
(12.18 GB); a row that needed more ran out of memory instead of spilling.

**Two sets of rows, on two main commits.** Each row of the evidence file names the Soup
commit it ran on (`main`) and how its k-bit model held the embeddings, norms and untied
output head (`setting`):

| set | Soup main | rows | `setting` | what the k-bit preparation did |
|---|---|---|---|---|
| A | `527314a4` (before #2683) | 1 to 26 | `kbit_fp32` | PEFT's `prepare_model_for_kbit_training` alone: fp32 embeddings, norms and head, bf16 compute |
| B | `3f9ba5ce` (#2683 in) | 27 to 44 | `bf16` | `prepare_kbit_model` on a bf16 card: bf16 embeddings, norms and head, the #2683 default (`RunSpec.kbit_fp32` False) |
| B | `3f9ba5ce` (#2683 in) | 45 to 47 | `kbit_fp32` | `auto_mixed_precision: true` on a Qwen2.5 base, which picks fp16: `prepare_kbit_model` keeps PEFT's fp32 (`RunSpec.kbit_fp32` True) |

The estimate prices the two settings with two constants: `INT8_CHECKPOINT_KEPT = (3.0, 1.0)`
for `bf16` and `INT8_CHECKPOINT_KEPT_FP32 = (5.0, 2.0)` for `kbit_fp32`.

**What is measured.** The #2049 harness, `benchmarks/harness/vram_estimate_calibration.py`,
unchanged (sha256 `cd4687bf327086dd5efdd0442a8cd7315ceb37da013dfb3dc35455c95ce898fb`, the
file [`gate-2049-resident-vram.md`](gate-2049-resident-vram.md) names; the same file on
both commits). Each row runs its `run_row` in a process of its own, through a driver that
sets the memory fraction first and adds `memory_fraction_cap` to the result line:
`soup train`'s own setup (`SFTTrainerWrapper.setup`), two training steps,
`torch.cuda.max_memory_allocated()` over them. Every row fills `max_length`, LoRA r8 on the
default targets, alpha 16, AdamW. For rows 45 to 47 the driver also adds
`auto_mixed_precision: true` to the harness's training block and runs the
`align_trainable_dtype_for_fp16` call that `SFTTrainerWrapper.train` makes before
`trainer.train()`, since the harness calls `trainer.train()` directly; without it the fp16
grad scaler meets the bf16 adapter TRL casts a k-bit model to and stops at step 1.

**Models.** Snapshots at the revisions #2049 measured: Qwen2.5-1.5B-Instruct `989aa798`,
Qwen2.5-0.5B-Instruct `7ae55760`, SmolLM2-360M-Instruct `a10cc151`. Rows marked `random` (set
A only) load the `config.json` of Qwen2.5-0.5B-Instruct `7ae55760` or Qwen2.5-3B-Instruct
`aa8e7253` initialised with `torch.manual_seed(0)` and saved in bf16; they are kept for
completeness and are not evidence about a checkpoint (below).

Every result line is in [`evidence/vram-int8-checkpointing.json`](evidence/vram-int8-checkpointing.json),
in the format of `evidence/vram-2049-resident.json` plus the `weights`, `main` and `setting`
fields.

---

## Why 8-bit and checkpointing did not add up

bitsandbytes' int8 matmul, `MatMul8bitLt`, keeps the tensors its backward needs on the
context object (`bitsandbytes/autograd/_functions.py:187`, `ctx.tensors = (CAt, subA, A)`)
instead of passing them to `save_for_backward`. Non-reentrant checkpointing frees a layer's
activations by intercepting `save_for_backward`, so it never sees these: every int8 linear
layer's input stays on the card from the forward pass until that layer's backward. The
backward does not even read `A` (`_functions.py:208`, `CAt, subA, _A = ctx.tensors`).
4-bit's `MatMul4Bit` keeps only the weight there (`_functions.py:358`).

Isolated on Qwen2.5-1.5B-Instruct on set A's commit, 8-bit, LoRA on q/k/v/o, checkpointing
on, 4 x 512 tokens, one forward and backward outside the harness. With bitsandbytes as
shipped, then with `MatMul8bitLt.forward` patched to drop `A` from `ctx.tensors`:

| | kept after the forward pass | peak |
|---|---|---|
| as shipped | 4.415 GiB | 8.840 GiB |
| `A` dropped | 2.603 GiB | 7.028 GiB |

The 1.812 GiB difference is the size of the inputs the int8 layers saw in that forward
pass, measured separately with forward pre-hooks (1.812 GiB).

## What the kept inputs are, per setting

Per layer and token, four inputs are kept: the q/k/v input (the attention norm's output,
one tensor for the three projections), the o_proj input (the attention output), the
gate/up input (the MLP norm's output) and the down_proj input. Read off layer 5 with
forward pre-hooks:

- **`bf16`, main `3f9ba5ce`.** `prepare_kbit_model(compute_dtype=torch.bfloat16)`, autocast
  bf16: all seven int8 linear layers of Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct and
  SmolLM2-360M-Instruct get bf16 inputs. That is 1 + 1 + 1 bf16 values per hidden unit and
  1 per intermediate unit: `INT8_CHECKPOINT_KEPT = (3.0, 1.0)`.
- **`kbit_fp32`.** The q/k/v and gate/up inputs are fp32 (the norms are fp32), the o_proj
  input is half precision, and the down_proj input is fp32 when gate/up met no outlier
  column and half precision when they did, because bitsandbytes adds the outlier columns
  back with an `addmm` (`backends/default/ops.py:98`) that autocast runs in half precision.
  On set A's commit Qwen2.5-0.5B and Qwen2.5-1.5B gave a bf16 down_proj input and
  SmolLM2-360M an fp32 one. The estimate charges the fp32 case: 4 + 2 + 4 bytes per hidden
  unit and 4 per intermediate unit, `INT8_CHECKPOINT_KEPT_FP32 = (5.0, 2.0)`.

Both are charged on every checkpointed layer.

## The rows

"before step" is `memory_allocated()` before the first step. "peak" is
`max_memory_allocated()` over the two steps. "estimate" is every term except `overhead`,
with the row's own setting (`RunSpec.kbit_fp32`), before the change (the int8 term off,
which is main) and after it. GB is 1e9 bytes.

### Set A: main `527314a4`, `kbit_fp32` (PEFT's fp32 norms)

| # | model | weights | load | seq x batch | ckpt | before step | peak | estimate before | est/peak | estimate after | est/peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen2.5-1.5B-Instruct | pretrained | 8-bit | 512 x 4 | on | 2.254 | 8.811 | 7.172 | 0.814 | 10.108 | 1.147 |
| 2 | Qwen2.5-1.5B-Instruct | pretrained | 8-bit | 1024 x 2 | on | 2.254 | 8.812 | 7.172 | 0.814 | 10.108 | 1.147 |
| 3 | Qwen2.5-1.5B-Instruct | pretrained | 8-bit | 2048 x 1 | on | 2.254 | 8.813 | 7.172 | 0.814 | 10.108 | 1.147 |
| 4 | Qwen2.5-1.5B-Instruct | pretrained | 8-bit | 512 x 2 | on | 2.254 | 5.792 | 4.730 | 0.817 | 6.198 | 1.070 |
| 5 | Qwen2.5-0.5B-Instruct | random | 8-bit | 512 x 4 | on | 0.906 | 6.515 | 5.556 | 0.853 | 6.953 | 1.067 |
| 6 | Qwen2.5-0.5B-Instruct | random | 8-bit | 1024 x 2 | on | 0.906 | 6.515 | 5.556 | 0.853 | 6.953 | 1.067 |
| 7 | Qwen2.5-0.5B-Instruct | random | 8-bit | 2048 x 1 | on | 0.906 | 6.516 | 5.556 | 0.853 | 6.953 | 1.067 |
| 8 | Qwen2.5-3B-Instruct | random | 8-bit | 512 x 1 | on | 4.032 | 6.969 | 5.379 | 0.772 | 6.568 | 0.942 |
| 9 | Qwen2.5-3B-Instruct | random | 8-bit | 512 x 2 | on | 4.032 | 9.267 | 6.653 | 0.718 | 9.031 | 0.975 |
| 10 | Qwen2.5-3B-Instruct | random | 8-bit | 1024 x 1 | on | 4.032 | 9.268 | 6.653 | 0.718 | 9.031 | 0.974 |
| 11 | Qwen2.5-1.5B-Instruct | pretrained | 8-bit | 1024 x 2 | medium | 2.254 | out of memory | 12.014 | - | 13.482 | - |
| 12 | Qwen2.5-1.5B-Instruct | pretrained | 8-bit | 512 x 1 | off | 2.254 | 5.361 | 5.841 | 1.090 | 5.841 | 1.090 |
| 13 | Qwen2.5-1.5B-Instruct | pretrained | 4-bit | 1024 x 2 | on | 1.623 | 6.216 | 6.524 | 1.050 | 6.524 | 1.050 |
| 14 | Qwen2.5-1.5B-Instruct | pretrained | bf16 | 1024 x 2 | on | 3.092 | 7.046 | 7.202 | 1.022 | 7.202 | 1.022 |
| 15 | Qwen2.5-3B-Instruct | random | 4-bit | 512 x 1 | on | 2.693 | 4.432 | 4.054 | 0.915 | 4.054 | 0.915 |
| 16 | Qwen2.5-3B-Instruct | random | 4-bit | 512 x 2 | on | 2.693 | 5.522 | 5.318 | 0.963 | 5.318 | 0.963 |
| 17 | Qwen2.5-0.5B-Instruct | pretrained | 8-bit | 512 x 4 | on | 0.906 | 6.065 | 5.556 | 0.916 | 6.953 | 1.146 |
| 18 | Qwen2.5-0.5B-Instruct | pretrained | 8-bit | 1024 x 2 | on | 0.906 | 6.065 | 5.556 | 0.916 | 6.953 | 1.146 |
| 19 | Qwen2.5-0.5B-Instruct | pretrained | 8-bit | 2048 x 1 | on | 0.906 | 6.066 | 5.556 | 0.916 | 6.953 | 1.146 |
| 20 | SmolLM2-360M-Instruct | pretrained | 8-bit | 1024 x 2 | on | 0.506 | 3.319 | 2.178 | 0.656 | 3.479 | 1.048 |
| 21 | SmolLM2-360M-Instruct | pretrained | 8-bit | 2048 x 1 | on | 0.506 | 3.320 | 2.178 | 0.656 | 3.479 | 1.048 |
| 22 | SmolLM2-360M-Instruct | pretrained | 8-bit | 4096 x 1 | on | 0.506 | 6.019 | 3.834 | 0.637 | 6.434 | 1.069 |
| 23 | SmolLM2-360M-Instruct | pretrained | 8-bit | 2048 x 1 | medium | 0.506 | 4.645 | 4.043 | 0.870 | 4.693 | 1.010 |
| 24 | Qwen2.5-0.5B-Instruct | pretrained | 8-bit | 1024 x 1 | medium | 0.906 | 4.162 | 4.376 | 1.052 | 4.725 | 1.135 |
| 25 | SmolLM2-360M-Instruct | pretrained | 8-bit | 2048 x 1 | off | 0.506 | 5.970 | 5.787 | 0.969 | 5.787 | 0.969 |
| 26 | SmolLM2-360M-Instruct | pretrained | 4-bit | 2048 x 1 | on | 0.354 | 1.937 | 2.020 | 1.043 | 2.020 | 1.043 |

### Set B: main `3f9ba5ce`, pretrained weights

| # | model | setting | load | seq x batch | ckpt | before step | peak | estimate before | est/peak | estimate after | est/peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 27 | Qwen2.5-1.5B-Instruct | bf16 | 8-bit | 512 x 4 | on | 1.787 | 7.333 | 6.705 | 0.914 | 8.261 | 1.127 |
| 28 | Qwen2.5-1.5B-Instruct | bf16 | 8-bit | 1024 x 2 | on | 1.787 | 7.333 | 6.705 | 0.914 | 8.261 | 1.127 |
| 29 | Qwen2.5-1.5B-Instruct | bf16 | 8-bit | 2048 x 1 | on | 1.787 | 7.334 | 6.705 | 0.914 | 8.261 | 1.127 |
| 30 | Qwen2.5-1.5B-Instruct | bf16 | 8-bit | 512 x 2 | on | 1.787 | 4.571 | 4.263 | 0.933 | 5.041 | 1.103 |
| 31 | Qwen2.5-1.5B-Instruct | bf16 | 8-bit | 1024 x 2 | medium | 1.787 | 9.568 | 11.547 | 1.207 | 12.325 | 1.288 |
| 32 | Qwen2.5-1.5B-Instruct | bf16 | 8-bit | 512 x 1 | off | 1.787 | 4.333 | 5.375 | 1.240 | 5.375 | 1.240 |
| 33 | Qwen2.5-1.5B-Instruct | bf16 | 4-bit | 1024 x 2 | on | 1.156 | 5.107 | 6.058 | 1.186 | 6.058 | 1.186 |
| 34 | Qwen2.5-1.5B-Instruct | bf16 | bf16 | 1024 x 2 | on | 3.092 | 7.046 | 7.202 | 1.022 | 7.202 | 1.022 |
| 35 | Qwen2.5-0.5B-Instruct | bf16 | 8-bit | 512 x 4 | on | 0.633 | 5.266 | 5.284 | 1.003 | 6.026 | 1.144 |
| 36 | Qwen2.5-0.5B-Instruct | bf16 | 8-bit | 1024 x 2 | on | 0.633 | 5.266 | 5.284 | 1.003 | 6.026 | 1.144 |
| 37 | Qwen2.5-0.5B-Instruct | bf16 | 8-bit | 2048 x 1 | on | 0.633 | 5.266 | 5.284 | 1.003 | 6.026 | 1.144 |
| 38 | Qwen2.5-0.5B-Instruct | bf16 | 8-bit | 1024 x 1 | medium | 0.633 | 3.496 | 4.104 | 1.174 | 4.290 | 1.227 |
| 39 | SmolLM2-360M-Instruct | bf16 | 8-bit | 1024 x 2 | on | 0.412 | 2.489 | 2.084 | 0.837 | 2.797 | 1.124 |
| 40 | SmolLM2-360M-Instruct | bf16 | 8-bit | 2048 x 1 | on | 0.412 | 2.490 | 2.084 | 0.837 | 2.797 | 1.123 |
| 41 | SmolLM2-360M-Instruct | bf16 | 8-bit | 4096 x 1 | on | 0.412 | 4.548 | 3.740 | 0.822 | 5.166 | 1.136 |
| 42 | SmolLM2-360M-Instruct | bf16 | 8-bit | 2048 x 1 | medium | 0.412 | 3.500 | 3.948 | 1.128 | 4.305 | 1.230 |
| 43 | SmolLM2-360M-Instruct | bf16 | 8-bit | 2048 x 1 | off | 0.412 | 4.485 | 5.692 | 1.269 | 5.692 | 1.269 |
| 44 | SmolLM2-360M-Instruct | bf16 | 4-bit | 2048 x 1 | on | 0.259 | 1.623 | 1.925 | 1.187 | 1.925 | 1.187 |
| 45 | Qwen2.5-1.5B-Instruct | kbit_fp32 (`auto_mixed_precision`, fp16) | 8-bit | 1024 x 2 | on | 2.256 | 8.818 | 7.172 | 0.813 | 10.108 | 1.146 |
| 46 | Qwen2.5-0.5B-Instruct | kbit_fp32 (`auto_mixed_precision`, fp16) | 8-bit | 1024 x 2 | on | 0.907 | 6.068 | 5.556 | 0.916 | 6.953 | 1.146 |
| 47 | Qwen2.5-1.5B-Instruct | kbit_fp32 (`auto_mixed_precision`, fp16) | 8-bit | 512 x 2 | on | 2.256 | 5.798 | 4.730 | 0.816 | 6.198 | 1.069 |

## What the rows show

- **Every pretrained 8-bit row with full checkpointing was under-priced on both
  settings**, except the `bf16` Qwen2.5-0.5B rows. Set A (`kbit_fp32`): 0.637 to 0.916 of
  the measured peak, below the 0.95 the 8-bit band of `tests/test_vram_estimate_calibration.py`
  allows; with the term 1.048 to 1.147. Set B, `bf16`: Qwen2.5-1.5B 0.914 and 0.933,
  SmolLM2-360M 0.822 to 0.837; with the term 1.103 to 1.136. Qwen2.5-0.5B read 1.003
  before (its 150k-vocabulary logits are most of its estimate) and 1.144 after, inside the
  band. Set B, `kbit_fp32`: 0.813 to 0.916 before and 1.069 to 1.146 after, the same as set
  A's matching rows (1.147, 1.146 and 1.070), so `INT8_CHECKPOINT_KEPT_FP32` still holds
  with #2683 in.
- **The controls do not move:** 8-bit without checkpointing (rows 12, 25, 32 and 43), 4-bit
  (13, 15, 16, 26, 33 and 44) and bf16 (14 and 34) give the same estimate before and after.
- **`medium`:** a `medium` layer that is not checkpointed already keeps these inputs in its
  activations, so the term is charged on half the layers. Set A: 0.870 and 1.052 before,
  1.010 and 1.135 after. Set B, `bf16`: 1.128 to 1.207 before and 1.227 to 1.288 after.
- **The harness agrees with the RTX 3060 record across cards.** Row 25 is #2049's row 35
  on another card: 5.970 GB here against 5.970 GB there. Row 13's `before step` is #2049's
  row 28's (1.623 GB), and rows 15 and 16, randomly initialised, read 4.432 and 5.522 GB
  against 4.430 and 5.521 GB for #2049's rows 32 and 33 on the pretrained weights. Row 34,
  bf16 LoRA on set B, reads row 14's peak (7.046 GB): #2683 changes only k-bit runs.
- **`tests/test_vram_estimate_int8_checkpointing.py`** holds every pretrained `kbit_fp32`
  row and, on `bf16`, the 8-bit rows with full checkpointing and the bf16 LoRA control to the
  bands of the #2049 test; checks that without the term every pretrained 8-bit row with full
  checkpointing falls below the band (the `bf16` Qwen2.5-0.5B rows excepted) and no other
  row moves; and pins the term per layer for each setting.

## Which #2049 rows the term leaves untouched

All 42. The term applies only to 8-bit loading with gradient checkpointing (`full` or
`medium`), and the three 8-bit rows of
[`evidence/vram-2049-resident.json`](evidence/vram-2049-resident.json) all ran without
checkpointing; the other 39 are bf16 LoRA, 4-bit and full fine-tuning. So
`tests/test_vram_estimate_calibration.py` passes unchanged. Outside the evidence, `soup
train`'s pre-flight, `soup profile`, the MCP `profile` tool and `soup plan` give a higher
estimate for every 8-bit run with checkpointing, and the same for every other run.

## What the rows do not show

- **The `bf16` rows the estimate already over-prices on main.** On set B the `bf16` 8-bit
  rows without checkpointing read 1.240 and 1.269, the 4-bit rows 1.186 and 1.187, and the
  `medium` rows 1.128 to 1.207, before this change. The per-layer activation coefficients
  (`ACT_EXTRA`) were fitted on #2049's fp32-norm rows, and #2683 measured no 8-bit run, so
  a whole layer in the `bf16` setting keeps less than the estimate charges. That is outside
  this record: the term leaves the first two groups unchanged and adds its bytes to the
  checkpointed half of a `medium` run, which takes those rows from 1.128 to 1.207 up to
  1.227 to 1.288. The test does not hold these `bf16` rows to the band; it checks that the
  term does not move the first two.
- **Randomly initialised rows are not evidence about a checkpoint.** Random weights meet
  no outlier column, so with fp32 norms bitsandbytes returns the gate/up outputs in fp32 and
  the down_proj input is fp32 (rows 5 to 10, read off with forward hooks). They are in the
  file and the table, and the test leaves them out. With the term, the random Qwen2.5-3B rows
  read 0.942 to 0.975: under-priced at 512 tokens in the same way as the 4-bit rows of the
  same shape (row 15, and #2049's row 32 at 0.915), so that part is not the int8 term.
- **A small part with bf16 norms that the term does not charge.** On the GPU test's tiny
  random Llama (16 layers, hidden 256, LoRA on q/v, 512 tokens), 8-bit keeps more than 4-bit
  after a checkpointed forward pass: 3,328 bf16 values a token per layer with fp32 norms,
  exactly `5 * 256 + 2 * 1024`, but 2,312 with bf16 norms against the `3 * 256 + 1 * 1024`
  = 1,792 charged. The 520 above it stay at 520 with an intermediate size of 4,096 (5,384
  against 4,864), so they are hidden-sized. `MatMul8bitLt`'s context holds the four inputs
  (`A` shares their storage) and an empty `subA`, so they are kept elsewhere; this record
  does not trace them. The pretrained `bf16` rows read 1.103 to 1.144 with the term, inside
  the band, and the test uses the 4,096-wide MLP, where the term is 0.95 of the gap.
- One card, one software stack, one memory cap, three pretrained models of 1.5B parameters
  and below, at up to 4,096 tokens a step.
- Not the cause, but noted: the `A` that bitsandbytes keeps is never read in its backward,
  so a bitsandbytes release that stops keeping it would make the term an over-estimate.
