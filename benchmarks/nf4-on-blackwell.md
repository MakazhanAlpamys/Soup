# NF4 on Blackwell (sm_120): the two problems, where each stands, how to re-measure

Status record, 2026-11-07. Not a gate: nothing here is a pass/fail criterion for a release.
It collects what was found about 4-bit NormalFloat (NF4) layer streaming on an RTX 5070
Laptop GPU (Blackwell, `sm_120`) in one place, with today's re-measurement and the scripts
that produce every number.

Two different things were found on this card. They are easy to confuse because both
involve `bitsandbytes` and both showed up as "streamed NF4 does not match resident NF4":

| | Problem 1: two kernels round differently | Problem 2: wrong gradients with recycled weight buffers |
|---|---|---|
| Issue | [#776](https://github.com/MakazhanAlpamys/Soup/issues/776) (closed 2026-09-14), pinned by [#968](https://github.com/MakazhanAlpamys/Soup/issues/968) | [#331](https://github.com/MakazhanAlpamys/Soup/issues/331) (closed 2026-08-07); upstream [bitsandbytes#2034](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2034), open |
| What | `bitsandbytes`' fused 4-bit GEMM and "dequantize, then `F.linear`" give results that differ by **one unit in the last place (ulp)** of the output dtype, for small row counts | `MatMul4Bit` keeps the packed weight on the autograd context outside `save_for_backward`; with gradient checkpointing and a recycled buffer the backward reads the wrong layer. Forward stays exact, gradients do not |
| Size | At most 1 ulp of the output dtype at the largest output of the row: 0.03 to 4 absolute on outputs of about 235 to 580 (unit-variance weights), up to 0.06 on outputs of about 5 to 12 (weights of std 0.02); both kernels are within about 1.5 ulp of an fp32 reference | Wrong gradients on every layer except the last few; large (the loss curve looked healthy) |
| Specific to Blackwell? | Measured on `sm_120` only; other architectures were not measured (no card here) | No: measured first on 8x H100 |
| Soup status | Not a Soup defect. Since v0.76.0 streamed NF4 follows `bitsandbytes`' own kernel choice, so streamed equals resident (see "What Soup does") | **Fixed in v0.73.0**: a streamed weight never enters `MatMul4Bit` |
| Upstream status | Filed on 2026-11-06 as [bitsandbytes#2121](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2121) (open, no reply as of 2026-11-07). `bitsandbytes` main is 3 commits past 0.50.2, none touching the CUDA 4-bit kernels, and behaves identically (below) | Reported as [#2034](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2034) on 2026-08-07 (open, no reply as of 2026-11-07) |

## Problem 1 - fused GEMM versus dequantize + `F.linear` (#776)

### What it is

`bitsandbytes` multiplies by a packed NF4 weight in one of two ways, and picks per call by
shape (`bitsandbytes/backends/cuda/ops.py`, `gemm_4bit`):

- a custom fused kernel that dequantizes inside the multiply, used for small row counts
  `M = batch x sequence` (up to 1536 rows, and by a per-architecture heuristic, which has a
  dedicated `sm_120` branch), or
- the fallback: dequantize the whole weight, then an ordinary cuBLAS GEMM (`F.linear`).

`bnb.matmul_4bit` calls the same `gemm_4bit` op whether or not a gradient is needed
(`MatMul4Bit.apply` in the autograd case), so "the `MatMul4Bit` path" and "the fused kernel"
are the same numbers in every row measured below (`autograd vs fused` was exactly 0 in all
72 rows of the 0.50.2 run).

The two ways accumulate in a different order, so they are not bit-identical. The size of
the difference is the interesting part: it is at most one ulp of the output dtype, and each
kernel is within 1.5 ulp of the same product computed in fp32.

### Measured today

Card: RTX 5070 Laptop GPU, `sm_120`, 8123 MiB as torch reports it; driver 616.92; torch
2.14.0+cu130 (CUDA 13.0); Python 3.12.10; Windows 11. Protocol: weight `N(0, 1)` of shape
`N x K`, quantized with `blocksize=64`, `nf4`, double quantization on; inputs `N(0, 1)`;
`M` in 1, 4, 8, 12, 32, 33, 64, 256, 512, 1536, 1537, 2048. "ulp" is the spacing of the
output dtype at the largest output magnitude of that row. Command and raw JSON below.

**Table 1 - bitsandbytes 0.50.2 (installed release), weights `N(0, 1)`** (outputs of magnitude
about 235 to 690; on the rows that differ, 235 to 580; absolute differences 0.03 to 4)

| K x N | dtype | rows where bnb picks the fused kernel | rows where fused != dequant | worst max abs diff (rows) | in output ulps | fused vs fp32, worst (ulp) | dequant vs fp32, worst (ulp) | diff on the fallback rows |
|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | float16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 4, 8, 12, 32, 33, 64 | 0.125 (M=1) | 1.00 | 0.89 | 0.51 | 0 (M = 256, 512, 1536, 1537, 2048) |
| 4096 x 4096 | bfloat16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1 | 1 (M=1) | 1.00 | 0.99 | 0.50 | 0 (M = 256, 512, 1536, 1537, 2048) |
| 4096 x 14336 | float16 | M = 1, 4, 8, 12, 32, 33, 64, 256 | M = 1, 4, 8, 12, 32, 33, 64, 256 | 0.25 (M=64) | 1.00 | 0.56 | 0.92 | 0 (M = 512, 1536, 1537, 2048) |
| 4096 x 14336 | bfloat16 | M = 1, 4, 8, 12, 32, 33, 64, 256 | M = 1, 64 | 2 (M=64) | 1.00 | 0.51 | 0.92 | 0 (M = 512, 1536, 1537, 2048) |
| 14336 x 4096 | float16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 4, 8, 12, 32, 33, 64 | 0.5 (M=32) | 1.00 | 0.90 | 0.95 | 0 (M = 256, 512, 1536, 1537, 2048) |
| 14336 x 4096 | bfloat16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 32, 33, 64 | 4 (M=32) | 1.00 | 0.84 | 1.00 | 0 (M = 256, 512, 1536, 1537, 2048) |

**Table 2 - bitsandbytes 0.50.2, weights `N(0, 0.02^2)`** (closer to a trained layer; outputs
of magnitude 4.7 to 13.8; absolute differences 1.5e-05 to 0.06)

| K x N | dtype | rows where bnb picks the fused kernel | rows where fused != dequant | worst max abs diff (rows) | in output ulps | fused vs fp32, worst (ulp) | dequant vs fp32, worst (ulp) | diff on the fallback rows |
|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | float16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 4, 8, 12, 32, 33, 64 | 0.003906 (M=1) | 1.00 | 0.69 | 0.50 | 0 (M = 256, 512, 1536, 1537, 2048) |
| 4096 x 4096 | bfloat16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 8, 12, 32, 33, 64 | 0.03125 (M=1) | 1.00 | 0.78 | 0.50 | 0 (M = 256, 512, 1536, 1537, 2048) |
| 4096 x 14336 | float16 | M = 1, 4, 8, 12, 32, 33, 64, 256 | M = 1, 4, 8, 12, 32, 33, 64, 256 | 0.003906 (M=1) | 1.00 | 0.65 | 1.17 | 0 (M = 512, 1536, 1537, 2048) |
| 4096 x 14336 | bfloat16 | M = 1, 4, 8, 12, 32, 33, 64, 256 | M = 1, 64 | 0.03125 (M=1) | 1.00 | 0.68 | 1.21 | 0 (M = 512, 1536, 1537, 2048) |
| 14336 x 4096 | float16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 4, 8, 12, 32, 33, 64 | 0.007812 (M=32) | 1.00 | 0.58 | 1.29 | 0 (M = 256, 512, 1536, 1537, 2048) |
| 14336 x 4096 | bfloat16 | M = 1, 4, 8, 12, 32, 33, 64 | M = 1, 32, 33, 64 | 0.0625 (M=1) | 1.00 | 0.58 | 1.45 | 0 (M = 256, 512, 1536, 1537, 2048) |

**bitsandbytes 0.50.3.dev0 (current `main`, pre-release wheel built 2026-10-27 from commit
`a616886`, loaded from a separate directory; the recorded module path is the `--target` copy):**
all 72 records of both runs are identical, field by field, to the 0.50.2 records above. `main`
is three commits past 0.50.2 (a version bump, a documentation URL, a ROCm-only cache), so this
is what the commit list predicts.

How to read it:

- **Where bitsandbytes uses its fused kernel, the two ways differ by at most 1 ulp (an
  absolute difference of one output-dtype spacing at the row's largest output; relative to an
  output near zero it is larger).** Where it falls back to dequantize + GEMM the difference is
  exactly 0, so "bit-exact" is true on the fallback rows and false on the fused rows.
- **Neither way is the accurate one.** Against an fp32 product of the same dequantized
  weights each is within 1.5 ulp on every fused row (columns "fused vs fp32" and "dequant vs
  fp32"), and which one is closer depends on the shape: the dequantize path is the further one
  for both shapes that involve 14336 at the realistic weight scale (1.17 to 1.45 ulp against
  0.58 to 0.68 for the fused kernel), the fused kernel for 4096 x 4096 at unit-variance weights
  (0.89 ulp against 0.51 in fp16).
- **Nothing here shows the effect is specific to Blackwell.** The mechanism (two kernels, two
  accumulation orders) would apply wherever the fused kernel is selected; `sm_120` is where it
  was measured, because no other card was available.
- Rows with `M` of 512 or more (a batch-1 sequence of 512 tokens already is) use the
  fallback on these shapes, so ordinary LoRA training at those token counts sees no difference
  at all. The difference lives in short sequences and in decode.

### What Soup does about it

- From v0.73.0 (the #331 repair below) until v0.75.x a streamed NF4 layer always took
  dequantize + `F.linear`, while a resident NF4 model took the fused kernel at small `M`. The streaming
  bit-exactness test therefore compared two kernels and failed with 4.88e-04 (fp16) and
  3.91e-03 (bf16) logit differences on this card, while `quantization: none` stayed exact.
- Since v0.76.0 ([#842](https://github.com/MakazhanAlpamys/Soup/issues/842), PR #1143) a
  streamed NF4 layer asks `bitsandbytes`' own dispatch which kernel it would use and runs
  that same operation through a Soup autograd Function that keeps its state checkpoint-visible.
  Streamed and resident NF4 now run the same kernel for the same shape.
  `tests/test_issue968_bnb_nf4_path_divergence.py` pins this at `== 0.0`, and pins the
  underlying `bitsandbytes` property (fused differs from dequantize inside the fused range,
  equal outside) so a version bump reports "unverified" instead of passing silently.
- `soup doctor` prints a short note on `sm_120` cards pointing at #776.
- No tolerance was introduced anywhere: an exact-equality check is the only thing that can
  detect a real streaming defect.

### Streamed versus resident, re-measured today

`harness/bitexact.py --quant nf4 --seq 64 --steps 20` on Qwen2.5-0.5B-Instruct (24 layers,
bf16, 2 stream buffers, pinned host store, batch 1), through the shipped streaming path of
this tree (`soup_cli.__file__` checked), with bitsandbytes 0.50.2. The resident reference is a
plain NF4 model running bitsandbytes' own forward.

| Run | Logits | LoRA gradients | 20-step loss curve | Verdict |
|---|---|---|---|---|
| `--quant nf4` | `max_abs_logit_diff` 0.0 | exact (`max_abs_diff` 0.0) | equal (`curve_max_rel` 0.0) | bit-exact gate passed |
| `--quant none` | `max_abs_logit_diff` 0.0 | exact (0.0) | equal (0.0) | bit-exact gate passed |

On 2026-09-10 the same protocol on the same card, driver and bitsandbytes (before #842
landed) gave `max_abs_logit_diff` 3.320312e-01 for `nf4` and 0.0 for `none`
([#776 comment](https://github.com/MakazhanAlpamys/Soup/issues/776)); that run diverged
because the two sides ran different bitsandbytes kernels (root cause in the same thread), and
this one does not. This harness does not itself prove that the fused kernel ran for a given
layer; the counter in `tests/test_issue968_bnb_nf4_path_divergence.py` does.

### Status, and when it "gets fixed"

There is nothing left to fix inside Soup for problem 1: both kernels are within 1.5 ulp of the
fp32 product and Soup now runs the kernel `bitsandbytes` itself chooses. Any further change
would be in the `bitsandbytes` kernel, which Soup does not control, so no date is given. Whether the
fused kernel should round the same way as dequantize + GEMM is a question for
`bitsandbytes`; it was filed on 2026-11-06 as
[bitsandbytes#2121](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2121)
(environment, a standalone reproducer, the numbers above); it is open and has no reply as of
2026-11-07. `bitsandbytes` main does not change the
result, and an open upstream PR
([bitsandbytes#2098](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/2098),
an `sm_120` single-row speed-up of the same kernel) would move the `M = 1` numbers if it
lands, so the pinned test records the version it was measured on.

## Problem 2 - wrong gradients with recycled weight buffers (#331, bnb#2034)

`MatMul4Bit.forward` stores `ctx.state = quant_state` and `ctx.tensors = (None, B)` as plain
attributes. `torch.utils.checkpoint` discards and recomputes saved tensors, and these are not
saved tensors, so the reference taken in the forward survives and aliases the layer-streaming
buffer pool. After the pool refills that slot with another layer, the backward reads the wrong
weight. Measured on 8x H100 against a resident NF4 reference: bit-exact forward, healthy loss
curve, gradients wrong on every layer except the last `stream_buffers`.

- Soup fixed it in v0.73.0 by keeping streamed weights out of `MatMul4Bit`
  (`install_dequant_forward`, later extended by #842 to keep the fused kernel without the
  alias). Ordinary QLoRA is not affected (`harness/issue331_qlora_scope.py`: 0.0 against a
  positive control that diverges by 3.77e-01).
- Upstream the mechanism is [bitsandbytes#2034](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2034)
  (open, no replies as of 2026-11-07). `harness/bnb_repro.py` is the standalone reproducer.

### Re-run today

`harness/bnb_repro.py` (8 layers of 4096 x 4096, 256 tokens, bf16, checkpointed, 2 recycled
slots) on bitsandbytes 0.50.2 and on 0.50.3.dev0 (`main`, separate directory): the output is
identical on both. The NF4 recycled-versus-private gradient differs in 6 of 8 layers (max abs
5.78e+01 to 6.57e+01, layers 6 and 7 equal), and the bf16 control matches on 8 of 8
(`RESULT: NF4 mismatch detected across 6/8 layers; bf16 control matched across 8/8 layers`).
So the upstream mechanism is still present on current `main`; what keeps streamed training
correct is Soup's workaround, not a bitsandbytes fix.

## Reproduce

All commands run from the repository root with the same interpreter that has CUDA torch and
`bitsandbytes`. Check `python -c "import soup_cli; print(soup_cli.__file__)"` points at the tree
under test before trusting a Soup-side number.

```bash
# Problem 1, no Soup involved: fused vs dequantize + F.linear, JSON out
python benchmarks/harness/bnb_nf4_path_divergence.py --json divergence-0.50.2.json

# the same against another bitsandbytes build installed with
#   pip install --no-deps --target ./bnb-main <wheel>
python benchmarks/harness/bnb_nf4_path_divergence.py --bnb-path ./bnb-main --json divergence-main.json

# Problem 1, Soup side: streamed NF4 vs resident NF4 (needs a local checkpoint)
python benchmarks/harness/bitexact.py --weights Qwen/Qwen2.5-0.5B-Instruct \
    --shards ./shards-nf4 --quant nf4 --seq 64 --steps 20

# Problem 2: the upstream reproducer
python benchmarks/harness/bnb_repro.py
```

`bitsandbytes` wheels for current `main` are published as the pre-release
`continuous-release_main` on the `bitsandbytes` GitHub releases page (a Windows wheel
included); install into a separate directory so the base environment is not touched.

## Scripts

| Script | What it answers here | Needs |
|---|---|---|
| [`harness/bnb_nf4_path_divergence.py`](harness/bnb_nf4_path_divergence.py) | Problem 1 with no Soup code: fused `gemm_4bit`, `MatMul4Bit.apply` and dequantize + `F.linear` against an fp32 reference, per shape, dtype and row count; records which `bitsandbytes` build it measured | one CUDA GPU, no download |
| [`harness/bitexact.py`](harness/bitexact.py) | Streamed vs resident logits, LoRA gradients and loss curve, NF4 or none, through the shipped streaming path; `--skip-lora-sync` is the negative control | CUDA GPU, a local checkpoint |
| [`harness/stream_probe.py`](harness/stream_probe.py) | What bounds a streamed step on this card (GEMM ceiling, step time, host-to-device copies, NF4 dequantization) in one process, so a fraction-of-ceiling is not compared across sessions | CUDA GPU, a 7B-class checkpoint for the published protocol |
| [`harness/issue361_nf4_throughput.py`](harness/issue361_nf4_throughput.py) | The v0.72.2 streamed-NF4 throughput protocol (batch 1, S=512, 10 warm-up + 50 steps) re-run on the code after the #331 repair, with a same-session GEMM ceiling | CUDA GPU, Llama-3.1-8B weights, pinned host memory |
| [`harness/variant2_gate.py`](harness/variant2_gate.py) | Problem 2: a defect-present control and the repaired arm in one process against one resident NF4 reference; the control must reproduce a wrong gradient | an H100-class card for the recorded 32B/72B runs |
| [`harness/bnb_repro.py`](harness/bnb_repro.py) | Problem 2 as a standalone upstream reproducer (recycled vs private buffers, bf16 control, `--bypass-pool` negative control) | one CUDA GPU, about a minute |
| [`harness/issue331_qlora_scope.py`](harness/issue331_qlora_scope.py) | Whether problem 2 reaches ordinary QLoRA (it does not), with a positive control | one CUDA GPU |

## What this record does not claim

- One card, one driver, one torch build. Other Blackwell parts (`sm_100`, `sm_121`) and older
  architectures choose `bitsandbytes` kernels by different rules and were not measured.
- Nothing here measures the accuracy cost of 4-bit quantization itself (NF4 against bf16);
  that is unrelated to which kernel multiplies and is the same on every card.
- The streamed-vs-resident comparison covers one 0.5B model at sequence 64; it is not a claim
  about every model or every row count.
- Raw outputs for the 2026-11-07 run are kept outside the repository; the tables above are
  copied from them, and every number can be regenerated with the commands above.
