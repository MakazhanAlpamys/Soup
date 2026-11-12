# Gate record: 8-bit loading with gradient checkpointing, resident peak VRAM

**Hardware.** NVIDIA RTX PRO 5000 Blackwell (sm_120; torch reports 50.75 GB), driver
580.105.08, Linux. torch 2.11.0+cu130, transformers 5.16.1, trl 0.29.1, peft 0.20.0,
bitsandbytes 0.50.2, accelerate 1.14.0, Python 3.13. Soup main `527314a4`. The card was
shared with other processes, so every row ran under
`torch.cuda.set_per_process_memory_fraction(0.24)` (12.18 GB); a row that needed more ran
out of memory instead of spilling.

**What is measured.** The #2049 harness, `benchmarks/harness/vram_estimate_calibration.py`,
unchanged (sha256 `cd4687bf327086dd5efdd0442a8cd7315ceb37da013dfb3dc35455c95ce898fb`, the
file [`gate-2049-resident-vram.md`](gate-2049-resident-vram.md) names). Each row runs its
`run_row` in a process of its own, through a driver that only sets the memory fraction
first and adds `memory_fraction_cap` to the result line: `soup train`'s own setup
(`SFTTrainerWrapper.setup`), two training steps, `torch.cuda.max_memory_allocated()` over
them. Every row fills `max_length`, LoRA r8 on the default targets, alpha 16, AdamW.

**Models.** Local snapshots at the revisions #2049 measured: Qwen2.5-1.5B-Instruct
`989aa798`, Qwen2.5-0.5B-Instruct `7ae55760`, SmolLM2-360M-Instruct `a10cc151`. Rows marked
`random` load the `config.json` of Qwen2.5-0.5B-Instruct `7ae55760` or Qwen2.5-3B-Instruct
`aa8e7253` initialised with `torch.manual_seed(0)` and saved in bf16; they are kept for
completeness and are not evidence about a checkpoint (below).

Every result line is in [`evidence/vram-int8-checkpointing.json`](evidence/vram-int8-checkpointing.json),
in the format of `evidence/vram-2049-resident.json` plus a `weights` field.

---

## Why 8-bit and checkpointing did not add up

bitsandbytes' int8 matmul, `MatMul8bitLt`, keeps the tensors its backward needs on the
context object (`bitsandbytes/autograd/_functions.py:187`, `ctx.tensors = (CAt, subA, A)`)
instead of passing them to `save_for_backward`. Non-reentrant checkpointing frees a layer's
activations by intercepting `save_for_backward`, so it never sees these: every int8 linear
layer's input stays on the card from the forward pass until that layer's backward. The
backward does not even read `A` (`_functions.py:208`, `CAt, subA, _A = ctx.tensors`).
4-bit's `MatMul4Bit` keeps only the weight there (`_functions.py:358`).

Isolated on Qwen2.5-1.5B-Instruct, 8-bit, LoRA on q/k/v/o, checkpointing on, 4 x 512
tokens, one forward and backward outside the harness. With bitsandbytes as shipped, then
with `MatMul8bitLt.forward` patched to drop `A` from `ctx.tensors`:

| | kept after the forward pass | peak |
|---|---|---|
| as shipped | 4.415 GiB | 8.840 GiB |
| `A` dropped | 2.603 GiB | 7.028 GiB |

The 1.812 GiB difference is the size of the inputs the int8 layers saw in that forward
pass, measured separately with forward pre-hooks (1.812 GiB).

What those inputs are, per layer and token: the q/k/v input and the gate/up input are the
norm outputs, fp32, because k-bit preparation casts the norms to fp32; the o_proj input is
the attention output, bf16; the down_proj input is fp32 when gate/up met no outlier column
and bf16 when they did, because bitsandbytes adds the outlier columns back with an `addmm`
(`backends/default/ops.py:98`) that autocast runs in bf16. Read off layer 5 with forward
hooks: Qwen2.5-0.5B and Qwen2.5-1.5B gave a bf16 down_proj input, SmolLM2-360M an fp32 one.
The estimate charges the fp32 case: 4 + 2 + 4 bytes per hidden unit and 4 per intermediate
unit, `INT8_CHECKPOINT_KEPT = (5.0, 2.0)` in bf16 values, on every checkpointed layer.

## The rows

"before step" is `memory_allocated()` before the first step. "peak" is
`max_memory_allocated()` over the two steps. "estimate" is every term except `overhead`,
before the change (the int8 term off) and after it. GB is 1e9 bytes.

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

## What the rows show

- **Every pretrained 8-bit row with full checkpointing was under-priced:** 0.637 to 0.916
  of the measured peak, below the 0.95 the 8-bit band of
  `tests/test_vram_estimate_calibration.py` allows. With the term they read 1.048 to 1.147.
  The two `medium` rows read 0.870 and 1.052 before and 1.010 and 1.135 after: a `medium`
  layer that is not checkpointed already keeps these inputs in its activations.
- **The controls do not move:** 8-bit without checkpointing (rows 12 and 25), 4-bit (13,
  15, 16 and 26) and bf16 (14) give the same estimate before and after.
- **The harness agrees with the RTX 3060 record across cards.** Row 25 is #2049's row 35
  on another card: 5.970 GB here against 5.970 GB there. Row 13's `before step` is #2049's
  row 28's (1.623 GB), and rows 15 and 16, randomly initialised, read 4.432 and 5.522 GB
  against 4.430 and 5.521 GB for #2049's rows 32 and 33 on the pretrained weights.
- **`tests/test_vram_estimate_int8_checkpointing.py`** holds the pretrained rows to the
  bands of the #2049 test, checks that without the term every pretrained 8-bit row with
  full checkpointing falls below the band and no other row moves, and pins the term per
  layer.

## What the rows do not show

- **Randomly initialised rows are not evidence about a checkpoint.** Random weights meet
  no outlier column, so bitsandbytes returns the gate/up outputs in fp32 and the down_proj
  input is fp32 (rows 5 to 10, read off with forward hooks). They are in the file and the
  table, and the test leaves them out. With the term, the random Qwen2.5-3B rows read 0.942
  to 0.975: under-priced at 512 tokens in the same way as the 4-bit rows of the same shape
  (row 15, and #2049's row 32 at 0.915), so that part is not the int8 term.
- One card, one software stack, one memory cap, three pretrained models of 1.5B parameters
  and below, at up to 4,096 tokens a step.
- The term follows the current k-bit preparation, which casts the norms to fp32. A change
  that keeps them in bf16 (#2683) halves the q/k/v and gate/up part of it.
- Not the cause, but noted: the `A` that bitsandbytes keeps is never read in its backward,
  so a bitsandbytes release that stops keeping it would make the term an over-estimate.
