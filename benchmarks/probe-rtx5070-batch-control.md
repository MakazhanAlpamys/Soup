<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any arm ran, which is the point of it: a rule
written after the numbers are in is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (MZAL81T0HFLB-00BL2, 954 GB each):
C: on a CPU root port (PEG0, 00:06.0), D: behind the chipset (RP13, 00:1D.4,
over the DMI 4.0 x8 link). Windows 11 Pro 10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2 (the two-drive gate's stack, unchanged on this box).
Soup: branch probe/batch-control, cut at 2c6ed087 (the head of PR #1536,
feat/two-drive-striping), worktree C:\Users\user\projects\Soup-r2.
src/ is byte-identical to 2c6ed087; only benchmarks/ changed.
soup_cli.__file__ resolves under Soup-r2\src (checked on 2026-10-01, and
stamped before and after every arm).
Harness: benchmarks/harness/stream_probe.py, the gate's cold 70B step, with
three changes committed before this rule, none inside the timed region: each
timed step records its wall-clock start (started_unix; the same four lines as
6ee60ce0 on probe/two-drive-sustained, so the two branches merge cleanly); an
out-of-memory error inside a block is written as a point of kind "oom" and the
run exits 3; every step point carries the per-row SHA-256 of its token ids and
the allocator's counters. The driver is
benchmarks/harness/batch_control_probe.py, with the rule in
batch_control_rule.py and the GPU samplers in batch_control_sampler.py; it
imports the gate's driver two_drive_stripe_gate.py unchanged. One process per
arm.
-->

# Probe — does a bigger batch ride free under the cold disk-tier read? (R2, the batch control)

**Status: rule committed 2026-10-01, BEFORE the run. Attempt 1 (2026-10-01 23:35) ended in NO
VERDICT, see §5; no round arm was judged, so the rule's rows 3-6 have not been applied to any
number.** **Attempt 2 (2026-10-07 10:15Z, `--run-prefix batch2`) completed; rows 1-3
did not fire, and the verdict is in §6.** Before the run the only executions were two `--dry-run`s of the driver into a scratch
folder (stamps, the cache check, a 4-s sampler self-test, the plan; no arm), a synthetic check
of the rule code on fake arm files, and a mocked run of the driver's arm flow (fake arm
processes that only write JSON). None of them wrote anything under `results/`.

---

## 0. The question

The cold 70B-shaped NF4 step at seq 512, batch 1 is bound by the read: 17.0-17.6 s on one
drive, 9.93-10.05 s on two striped drives (the gate's sequence 4), against a read-free compute
floor of 6.43 s. The record that measured the floor drew the consequence and said it was not a
measurement (`probe-rtx5070-what-bounds-streaming.md` §21, Finding 21):

> Because the step is read-bound and the read is per-step not per-token, **tokens per step are
> free until the compute floor meets the read**: from the two B points, B(S) = 1.24 s + 0.0101 s
> x S, which reaches 17.0 s at about **1,560 tokens per step** — batch 3 x 512 would roughly
> balance the two on this fixture, if it fits, and tok/s scales with the batch until then. That
> is arithmetic from two shapes, not a measurement at batch 3.

So: **does tok/s scale with `--batch` 1 -> 2 -> 3 at seq 512 on the cold disk tier as that
model predicts, and what peak VRAM does each batch need on this 8 GB card?**

Two drive layouts, because the second drive moves the crossover. The model's step is
`max(R, B(512 b))`, with R the layout's own batch-1 step (arithmetic, not a prediction of R):

| layout | R (gate sequence 4 median) | B(1024) | B(1536) | crossover (R - 1.242) / 0.010124 | where it falls |
|---|---|---|---|---|---|
| SINGLE | 17.438 s | 11.609 s | 16.792 s | 1,600 tokens | just past batch 3 |
| STRIPED | 9.971 s | 11.609 s | 16.792 s | 862 tokens | between batch 1 and batch 2 |

On one drive the model says batch 2 and batch 3 are both near-free (tok/s x2.00 and x3.00). On
two drives the step is already compute-bound at batch 2 (tok/s x1.72, then x1.78 at batch 3).
At batch 3 the two layouts' predicted steps are 17.44 s and 16.79 s: **the second drive, worth
1.75x at batch 1, would be worth about 4% at batch 3.** Whether that holds decides whether
striping and a bigger batch add up or substitute for each other.

This is also the control arm of the planned layer-major (L2L) experiment: on the 70B shape the
question for L2L is what it adds over the largest batch that fits, not over batch 1. §1a lists
what that experiment needs from here, and §1 says which arms it may cite.

## 1. The fixture, the arms and the order

The gate's fixture, unchanged: weights `D:/synth/llama-70b-shape-v32k-f4` (synthetic
Llama-3.1-70B shape, bf16 source, 32k head, random values, untied); NF4, disk tier, seq 512;
LoRA r 8 on q/k/v/o with dropout 0.0, adapter seed 3, input seed 17, `PagedAdamW8bit` at lr
1e-4, 2 layer buffers, `read_ranges` 4; both layouts pinned.

- **SINGLE — the L2L control.** No `--shards`: the harness resolves the default one-root
  cache, `C:\Users\user\.soup\layer-stream\D___synth__llama-70b-shape-v32k-f4`. Not the
  `D___synth__llama-70b-shape-v32k-f4.refit` copy that sits beside it, and not a cache under
  `SOUP_LAYER_STREAM_CACHE_DIR`: row 2 checks every SINGLE arm's `meta.shard_dir` against that
  exact folder. The whole store is on C:, read through direct I/O, `--read-ahead 2`.
- **STRIPED — an extra, not the L2L control.** `--shards
  C:/Users/user/.soup/layer-stream-striped/D___synth__llama-70b-shape-v32k-f4 --stripe-dir
  D:/soup-stripe --read-ahead 3`. Under the code at 2c6ed087 root 1's folder is
  `D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80` (the driver prints it from
  `stripe_folder_name`). It exists for the crossover question of §0; an L2L run on
  `origin/main` has no striping to compare with it.

**The code, and what the SINGLE arms are evidence for.** Every arm runs 2c6ed087, the head of
PR #1536. Its disk reader is the per-drive dispatcher that PR introduced, so the SINGLE arms run
that dispatcher with one drive: not byte-for-byte the reader on `origin/main`, where an L2L run
would start. The evidence that this one-drive path reproduces the known step is the gate's
sequence 4, whose SINGLE arms read 17.552, 17.431 and 17.438 s (gate §5.18) inside the band
gate-974 set for the pre-PR reader. That was measured on f798c46e. Between it and 2c6ed087,
`src/` gained the PR's review fixes and a merge of `origin/main` (a5dac3bc) that brought, among
others, #1143 (#842: a checkpoint-visible fused NF4 forward where bitsandbytes takes its fused
GEMM, and #841's cached per-buffer views). At this fixture's M (512 tokens and up) bitsandbytes
0.50.2 does not take that GEMM: for a weight of 4 M elements or more its heuristic reads "custom
wins up to M=512; dequant+F.linear wins above that", and its sm120 branch caps the custom kernel
at `M <= 256` for weights of three waves or more (`bitsandbytes/backends/cuda/ops.py`,
`_gemm_4bit_use_custom_cuda`). So the forward stays dequantise plus `F.linear`, as in §21; what
#841 removes is at most §21's 0.65 ms per visit (Finding 22), about 0.1 s per step. That is
reading code, not a measurement: row 1 re-establishes the known step on 2c6ed087 in this session,
and the B blocks below measure the compute floor on it.

Each layout runs at `--batch 1`, `2` and `3`. Every arm is `--steps 3 --warmup 1 --step
--skip-events`: the gate's judged block exactly (three timed steps after one warm-up, the
uninstrumented `step_plain` point), so the SINGLE batch-1 arm is the quantity the gate's band
was written for. **A SINGLE arm then runs the read-free compute floor at its own batch**
(`--ablate --arms B_nocopy --rounds 1`: arm B of §21, no source read and no host-to-device copy,
three timed steps after one warm-up), after its step block, in the same process. B is not
judged; it measures the compute half of the model on this code at 512, 1024 and 1536 tokens.
It runs in SINGLE arms only because without a read the layout does not matter.

The command per arm, as the driver builds it (`<b>` the batch, `<arm flags>` from above,
`<floor>` = `--ablate --arms B_nocopy --rounds 1` for SINGLE arms and nothing for STRIPED):

```
python benchmarks/harness/stream_probe.py --weights D:/synth/llama-70b-shape-v32k-f4 --quant nf4 --tier disk --seq 512 --batch <b> --steps 3 --warmup 1 --step --skip-events <floor> <arm flags> --label "R2 batch control round <r> arm <LAYOUT> batch <b>" --out benchmarks/results/probe-rtx5070/batch-control/batch_r<r>_<layout>_b<b>.json
```

with `PYTHONPATH=C:/Users/user/projects/Soup-r2/src` set by the driver.

**The order: two rounds, the second the exact reverse of the first.**

| position | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| round 0 | SINGLE b1 | STRIPED b1 | SINGLE b2 | STRIPED b2 | SINGLE b3 | STRIPED b3 |
| round 1 | STRIPED b3 | SINGLE b3 | STRIPED b2 | SINGLE b2 | STRIPED b1 | SINGLE b1 |

The first arm alternates (SINGLE, then STRIPED), the layouts interleave, and every one of the
six arms sits at mean position 2.5 over the two rounds, so a drift that is linear over a round
(the drives or the card warming) shifts round 0's batch-b ratio one way and round 1's the
other, and cancels in their mean. Three rounds (the gate's number) would not balance
(forward, reverse, forward), and a fourth would take the sequence past the time this probe
shares an AC window with the sustained probe (§4).

**Before the rounds: the build arm(s), not timed, never judged.** The driver first runs the
sharder's own cache check on the CPU (index and `stat` reads only: no torch, no shard opened)
with the arguments the harness passes (bfloat16, nf4, double quant, quantised on cuda, no
external tensors). A STRIPED build arm always runs (`--batch 1 --steps 1 --warmup 0 --step
--skip-events`); a SINGLE one runs only when the check does not report a hit. On 2026-10-01 the
check read: SINGLE **hit** at 20:12:56 and 20:16:35; STRIPED **miss** at 20:12:56 ("stripe
marker D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80\stripe.json is missing
or unreadable") and **hit** at 20:16:35, after the sustained probe's own build ran in between.
So a STRIPED build arm on a hit is one cold step (about 40 s); on a miss it re-shards (the
gate's took 275.9 s, §5.1 there). Either way the driver then waits 300 s before the first round
arm.

### 1a. What the L2L experiment needs from this record

The planned L2L step 0 runs k micro-batches of 1 x 512 through each loaded layer, on one drive.
It may cite the **SINGLE** rows of this record as its control and not the STRIPED ones, with
the code difference of §1 stated beside them. What it gets:

- **tok/s at batch 1, 2 and 3, and the largest batch that fits (row 4).** L2L's own
  contribution on this shape starts beyond that batch.
- **The read-free floor B at 512, 1024 and 1536 tokens, in session.** A rule that compares an
  L2L arm with "the read-removed arm at the same shape" has it here.
- **The loss convention.** The harness calls `model(input_ids=ids, labels=ids)` with no
  `num_items_in_batch`, so the step's loss is the mean over all b x 511 label tokens of the
  batch. An L2L arm over k micro-batches matches it only when each micro-batch's summed loss is
  divided by that same total, HF's `num_items_in_batch` over all k micro-batches; otherwise
  the gradients differ by a scale.
- **No RNG in the forward on this fixture.** LoRA dropout is 0.0 (`make_lora`) and the
  fixture's `config.json` sets no `attention_dropout` (LlamaConfig's default is 0.0), so the
  forward draws no random numbers, and an L2L arm needs no per-micro-batch RNG state here. A
  real model trained with the default `lora.dropout` 0.05 still does.
- **The token ids, row by row.** Every step point carries a SHA-256 per row of its `(b, 512)`
  ids, and row 2 requires every arm of one batch to carry the same list, so the losses of two
  arms of one batch are losses on the same tokens. An L2L arm can show it trains on the same
  rows. Whether row 0 of a `(b, 512)` draw equals the `(1, 512)` draw of the batch-1 arm is
  recorded, not assumed.
- **What a loss difference means.** Losses are not bit-identical across processes even within
  one arm (the gate's §5.20: same-arm differences reach 0.017391 on a first timed step), so an
  L2L-against-batch comparison of losses is read against the same-arm spread, as the gate did.
- **Peak VRAM per batch.** L2L moves the checkpointed boundary activations
  (2 x 80 layers x 8192 x 2 B = 1.31 MB per token on this shape) to host RAM; the growth of the
  peak with the batch says how much of it is that term.

## 2. The decision rule, written before the run

Nothing in this section is edited once a number exists.

**Inputs.** For layout `l`, batch `b` and round `r`:

- `step(l, b, r)`: the arm's `step_s_mean` (the JSON names are below).
- `e(l, b, r) = step(l, 1, r) / step(l, b, r)`, and `ē(l, b)`, its mean over the two rounds.
- `r̄(l, b) = b x ē(l, b)`: the tok/s ratio over batch 1, since tok/s is 512 b / step.
- `R(l)`: the mean of `step(l, 1, r)` over the two rounds.
- The model: `B(S) = 1.241971 s + 0.0101240 s x S`, and `e_pred(l, b) = R(l) / max(R(l), B(512 b))`.
- `in_use(l, b, r)`: the largest `nvidia-smi` memory-in-use sample (MiB) the driver took inside
  the arm's three timed steps (2-s sampling; each step's window runs from its `started_unix`
  for `step_s`), or over the whole arm if no sample fell inside them.
- The class of an e at batch b: **NEAR-FREE** when e >= **0.95**; else **NONE** when
  b x e <= **1.05**; else **PARTIAL**.

| row | measured | verdict |
|---|---|---|
| 1 | any SINGLE batch-1 arm whose `step` is outside **15.0-21.5 s**, or that has none | **NO VERDICT** for both layouts: the session does not reproduce the known step; record and stop |
| 2 | any arm off the path under test: `direct_io` or `pinned` not true; a source other than `AsyncDiskSource`; read-ahead not 2 (SINGLE) or 3 (STRIPED); `shard_dir` not the layout's cache of §1; a STRIPED arm whose stripe roots are not `D:\soup-stripe` or whose 80 layers are not placed i mod 2, or a SINGLE arm with any stripe root; `batch` or `seq` not the arm's; `tokens_per_step` not 512 b; not 3 timed steps; a timed step that did not load 157 + 2; not b row hashes; `shard_seconds` above **30 s** (it re-sharded). Two arms of one batch whose row hashes differ. Or the protocol stopped (a position void twice, a pre-arm wait expired), so a position that row 4 did not skip has no arm | **NO VERDICT** for both layouts: the arms did not measure what the rule reads |
| 3 | any STRIPED batch-1 arm whose `step` is outside **7.69-12.0 s**, or that has none | **NO VERDICT for STRIPED**; rows 4-6 are applied to SINGLE alone |
| 4 | per layout, for b = 2 then 3, in either round: (i) an out-of-memory point or line; (ii) no `step_plain` point within **600 s**; (iii) `peak_alloc_gb` above the arm's own `vram_total_gb`; (iv) **a spill**: `in_use` >= **7,620 MiB** AND `step(l, b, r)` > `step(l, 1, r) + B(512 b)` | **DOES NOT FIT at b**: a result, not a void. Quote which test fired, the peak, the error or the cap, and for (iv) the power, SM clock and utilisation of the timed steps and the shared GPU memory. Larger batches of that layout are not run and not judged |
| 5 | per layout, each b that fits: the class of `ē(l, b)` | **NEAR-FREE**, **PARTIAL** or **NONE** at b. Quote `ē`, both rounds' `e`, `r̄`, tok/s and the peak VRAM. Two labels, each quoted beside the class wherever it appears: **AT THE CARD'S EDGE** when `in_use` >= 7,620 MiB in either round; **SLOWER THAN READ + COMPUTE IN SERIES** when `step(l, b, r)` > `step(l, 1, r) + B(512 b)` in either round |
| 6 | per layout, each b classed in row 5: its class against the class of `e_pred(l, b)` | **AGREES** or **DISAGREES AT b**, with any label of row 5. Quote both classes, the measured crossover bracket and the model's crossover (R(l) - 1.241971) / 0.0101240 tokens |

The rows are applied in order. Rows 1 and 2 are validity conditions for the whole probe; row 3
is one for STRIPED. Rows 4-6 are the verdict, per layout and per batch: there is no single
pass or fail. **The measured crossover bracket** of a layout runs from 512 x the largest b for
which every batch up to it is NEAR-FREE (batch 1 counts as NEAR-FREE by definition) to 512 x
the first b judged PARTIAL or NONE, and is open-ended ("not reached") when no such b exists
among the batches that fit.

**The model's prediction, computed now** from R at the gate's sequence-4 medians (the rule
uses this session's R, which moves these numbers a little):

| layout | b | B(512 b) | predicted step | `e_pred` | class | predicted tok/s | `r̄` |
|---|---|---|---|---|---|---|---|
| SINGLE | 1 | 6.425 s | 17.438 s | — | — | 29.36 | 1 |
| SINGLE | 2 | 11.609 s | 17.438 s | 1.000 | NEAR-FREE | 58.72 | 2.000 |
| SINGLE | 3 | 16.792 s | 17.438 s | 1.000 | NEAR-FREE | 88.08 | 3.000 |
| STRIPED | 1 | 6.425 s | 9.971 s | — | — | 51.35 | 1 |
| STRIPED | 2 | 11.609 s | 11.609 s | 0.859 | PARTIAL | 88.21 | 1.718 |
| STRIPED | 3 | 16.792 s | 16.792 s | 0.594 | PARTIAL | 91.47 | 1.781 |

Crossovers: SINGLE 1,600 tokens (the bracket the model predicts is "beyond 1536, not
reached"); STRIPED 862 tokens (bracket (512, 1024]). Row 4 (iv)'s bound, read plus compute in
series, is 29.05 s and 34.23 s for SINGLE at batch 2 and 3, 21.58 s and 26.76 s for STRIPED.

**Peak VRAM, as arithmetic, judged by nothing.** §21's round-0 baselines peaked at 3.894 GB
allocated at 256 tokens and 4.378 GB at 512 (`i974_ablate_cold70b_seq{256,512}.json`), i.e.
1.889 MB per token on top of 3.411 GB. Extended to the batch: **5.345 GB at batch 2 and 6.312 GB
at batch 3**, against the 8.518 GB card (`vram_total_gb`, gate sequence 4). The boundary
activations alone (1.31 MB per token) would give 5.72 GB at batch 3. Neither counts the CUDA
context or reserved-but-free blocks (reserved ran 4.798 GB at 4.378 GB allocated in every gate
arm), so batch 3 should fit with about 2 GB to spare in `allocated` terms and much less in
what `nvidia-smi` sees: it may well read near 7,620 MiB without spilling, which is why row 4
(iv) asks for the slow step as well and the memory alone is only a label.

**Where every threshold comes from.** Each is read from a committed record; the source line is
quoted.

| threshold | value | source |
|---|---|---|
| SINGLE band (row 1) | 15.0-21.5 s | `gate-two-drive-striping.md` §2, row 1: "any SINGLE arm outside **15.0-21.5 s** — gate-974 §8's own cold rows are 16.175 s (15.77-17.06) and 17.311 s (16.28-20.42), widened ~5% each side". The gate's sequence 4 read 17.552, 17.431 and 17.438 s inside it |
| re-shard (row 2) | 30 s | gate §5.17: "Every arm's `shard_seconds` is 0.003-0.158 s, a cache hit."; gate §5.1: "Sharding: **231.812 s**". 30 s is far from both |
| STRIPED band, upper (row 3) | 12.0 s | gate §2: "STRIPED <= **12.0 s** in every round" (its SHIP line), met in every round of sequence 4 with "**The median of the three STRIPED means is 9.971 s.** The largest is 10.050 s." (§5.18) |
| STRIPED band, lower (row 3) | 7.69 s | gate §5.18: "Every arm moved 70,378,258,732 bytes per step"; gate §0: "Two drives read 7.65-9.15 GB/s together". 70,378,258,732 B / 9.15 GB/s = 7.692 s: a STRIPED step faster than that would read faster than any two-drive read on record, and is a measuring fault |
| out of memory (row 4 i) | exit 3 and a point of kind `oom`, or an out-of-memory line in the log | `stream_probe.py` since this probe's harness commit |
| wall cap (row 4 ii) | 600 s | a protocol bound, not a measurement: about 4x the longest arm the plan expects (SINGLE batch 3 with its B block, about 152 s; §4). A spill that slows the step 3x still finishes under it and meets (iii) or (iv) |
| allocated above the card (row 4 iii) | `peak_alloc_gb` > `vram_total_gb` | `gate-836-bench-train-contract.md` §8: "On this platform, a `memory` figure above the card's capacity is the signal that it did not." Allocated, not reserved: `gate-v0.73.1-measured-vram-fit.md` §3: "Reserved overshoots what must fit because it holds freed blocks that are returned under pressure. This is the reason the shipped gate reads `max_memory_allocated`." |
| the card as full as a spill (row 4 iv, row 5's first label) | 7,620 MiB in use | `probe-rtx5070-what-bounds-streaming.md` §18: "`nvidia-smi` reported **7,620 of 8,151 MiB in use, 100% utilisation, 1.8 GHz, 40 W** — torch's own peak was 4.38 GB allocated / 4.83 GB reserved", the reading that record took as the WDDM spill signature on this card. A spill need not show in torch's own counters, which is why (iii) alone is not enough |
| slower than read plus compute in series (row 4 iv, row 5's second label) | `step(l, 1, r)` + B(512 b) | The model's own worst case: the step is never longer than reading and computing one after the other with no overlap at all, since both are measured or modelled whole (R in this round; B from §21). A step past it has a cost the model has no term for. §18's spilled step is the scale of such a cost: "\| B no read, no copy \| **39.8 s** \|" at 1 x 512, where §21 measured "**6.450 / 6.401 s**" for the same arm and shape without a spill, 6.2x. The bound is 1.59-1.96x the predicted step here (STRIPED batch 3 to SINGLE batch 3), so a spill of that kind clears it and an imperfect overlap does not |
| NEAR-FREE (row 5) | e >= 0.95 | Two allowances, each from a record. What Finding 21 itself calls free: the baseline step grew from 17.005 to 17.139 s and from 17.001 to 17.208 s when the tokens doubled under the read (§21 table: "\| A baseline \| **17.139 / 17.208 s** \| 29.9 / 29.8 \| **17.005 / 17.001 s** \| 15.1 / 15.1 \|"), +0.79% and +1.21% per doubling, so +1.92% at batch 3 (log2 3 = 1.585 doublings). And two arm means of one session: gate §5.18's STRIPED means span 9.929-10.050 s (1.22%) and its SINGLE means "span 17.431-17.552 s" (0.70%); two arms anywhere in the wider span add up to 2 x 1.22%. Together 4.36%, i.e. e >= 0.958. 0.95 (a step at most 5.3% longer) is the round number just below, and is therefore slightly lenient |
| NONE (row 5) | b x e <= 1.05 | The model's own floor for a fully compute-bound step: with R far below B, `r̄` = b x B(512) / B(512 b) = **1.107 at batch 2 and 1.148 at batch 3**, above 1 because B carries a 1.242 s per-step cost (§21: "B(S) = 1.24 s + 0.0101 s x S"). A tok/s gain of 5% or less is below that floor, so no regime of the model produces it; 5% mirrors the NEAR-FREE band |
| model constants (rows 4 iv, 5, 6) | 1.241971 s, 0.0101240 s/token | §21 table: "\| B no read, no copy \| **6.450 / 6.401 s** \| 79.4 / 80.0 \| **3.822 / 3.846 s** \| 67.0 / 66.6 \|", unrounded from the JSON: 6.425469 s at 512 tokens and 3.833720 s at 256 (round means); slope (6.425469 - 3.833720) / 256, intercept 3.833720 - 256 x slope. Finding 21 rounds them to "1.24 s + 0.0101 s x S" |
| foreign reader (a void, §4) | 1 GB per arm | gate §5.21: "the largest foreign reader in any arm was `MoUsoCoreWorker.exe` (Windows Update's orchestrator, started during the arm), which read 58.2 MB during `gate4_r0_single`." 1 GB is 17x that; this probe's arms (55-152 s expected) are no longer than the gate's (94-154 s). The same condition and number as the sustained probe's §4 |

**Why power and the SM clock are recorded and not judged.** A spill is often described as "full
memory, low power, falling clocks", and §18's matched it ("1.8 GHz, 40 W"; in its arm D "the
reported clock falling 1965 -> 1462 MHz"). The other spill on record on this card did not:
`gate-836-bench-train-contract.md` §8, "Shape B completed while allocating 1.51x the card's
VRAM", ran at "100% utilisation whenever busy, 71–78% of samples busy, 2782 MHz median clock
while busy (2797 peak), 32.7–33.3 W median and 47.7 W peak against the 50 W cap". And no
committed record gives this step's healthy power or clock on mains to set a "low" against; the
gate's own SM samples ranged 907-2040 MHz across its valid arms (§5.18). A condition on either
would miss a spill like gate-836's, or fire on a healthy arm. So the driver samples power,
clock and utilisation every 2 s and the rule quotes them (median and minimum over the timed
steps) beside every row-4 spill and every batch-3 arm, without setting anything on them.

**What the rule deliberately does, stated now so none of it is read in afterwards:**

1. **The mean of the two rounds' ratios, not "in every round".** The reversed second round is
   what cancels a linear drift, and it cancels only in the mean; requiring each round alone
   to clear a band would let that drift decide the class. Both rounds' `e` are quoted beside
   every class, so a split is visible.
2. **The model is applied to this session's own R.** Row 6 does not test whether today's read
   matches §21's (rows 1 and 3 bound that); it tests the model's other two parts: that compute
   grows as B(S) says, and that the step is the larger of the read and the compute, with the
   other hidden.
3. **The model's B(S) was measured on older code.** §21 ran PR #991's tree (3d2be4d0,
   2026-09-15); this probe runs 2c6ed087, which carries #1143. At M >= 512 the forward is the
   same dequantise plus `F.linear` (§1), so B(S) is expected to hold within about the 0.1 s
   #841 can remove. If it does not, the in-session B blocks show it, and a DISAGREES that the
   B line explains is the code moving, not the model's structure failing.
4. **SINGLE batch 3 sits on the model's edge.** B(1536) is 16.79 s against a 17.44 s read, 3.7%
   under the crossover. The max() model has no term for imperfect overlap as the two meet, so
   a PARTIAL there is a DISAGREES that the record reports as such. It is not a fault in the
   protocol, and nothing in the rule moves to absorb it.
5. **A spill is a fit result, never a slow step.** Row 4 (iv) takes it out of rows 5 and 6
   before they can read it as compute catching up with the read. What (iv) cannot see is a
   spill too small to push the step past read plus compute in series; the edge label marks
   every batch whose card was that full, and §5 will quote, for every batch-2 and batch-3 arm,
   the arm process's shared GPU memory (the Windows counter that rises when allocations land in
   host memory), its dedicated memory, `peak_reserved_gb` against the card and
   `num_alloc_retries`.
6. **An out-of-memory error at a larger batch is a result.** The gate voided an arm with no
   `step_plain` point and re-ran it; here an arm that stopped on out of memory (the harness's
   `oom` point, exit 3, or an out-of-memory line in its log) or ran past the 600-s cap is not
   re-run. It is row 4's input, and the sequence goes on without that layout's larger batches.
7. **The read-free floor is reported, not judged.** The crossover from the in-session B line
   (the least-squares line through B at 512, 1024 and 1536 tokens, where R meets it) is quoted
   beside row 6's model crossover, so a reader can see whether a DISAGREES came from the
   compute growing differently from B(S) or from the overlap.
8. **Losses are recorded, not judged.** Row 2 only makes sure the arms of one batch saw the
   same tokens.

**The inputs, as the JSON names them** (fixed with the rule):

- `step`: `step_s_mean` of the `kind: "step"`, `label: "step_plain"` record of
  `batch_r<r>_<layout>_b<b>.json`; its `tok_per_s`, `peak_alloc_gb`, `peak_reserved_gb`,
  `tokens_per_step`, `ids.row_sha256` and `memory` (`num_alloc_retries`, `num_ooms`, `free_gb`)
  from the same record, and the per-step `step_s`, `started_unix`, `loss`, `layer_loads` and
  `large_loads` from its `records`.
- `vram_total_gb`, `shard_dir`, `direct_io`, `pinned`, `source_class`, `read_ahead`,
  `stripe_roots`, `layer_roots`, `shard_seconds`, `batch`, `seq`: the arm JSON's `meta`.
- The out-of-memory point: the `kind: "oom"` record (its `label`, `error_type`, `error`,
  `peak_alloc_gb`, `peak_reserved_gb`), or an out-of-memory line in the arm's `.log`.
- The read-free floor: `step_s_mean` of the `kind: "ablate"`, `label: "B_nocopy_r0"` record
  of a SINGLE arm.
- In `batch_driver.json`, per arm: `runs[*].timed_out`; the 2-s `nvidia-smi` rows
  `runs[*].during_samples.gpu` as `[unix time, used MiB, total MiB, SM MHz, temperature C,
  power W, utilisation %, throttle reasons]`, from which `in_use` and the timed-step power and
  clock are computed; the 1-s GPU memory rows `runs[*].during_samples.gpu_memory`; the foreign
  readers from `runs[*].after.top_readers`.
- `python benchmarks/harness/batch_control_probe.py --summarize` recomputes every input and
  applies the rows from those committed files alone (the rule's code is
  `batch_control_rule.py`). Comparisons are made on unrounded values against the thresholds
  as written; numbers are quoted at three decimals, and none is rounded in a way that moves a
  row.

**Also recorded, not part of the verdict:** the read-free floor per batch and its line; A / B
per batch (how far above the floor the step sits); `implied_h2d_gb_per_s`; each drive's read
rate per step, from the per-step `started_unix` and the driver's 1-s PDH counters; the SM
clock at the start and end of every block, and every 2 s with temperature, power, utilisation
and throttle reasons; dedicated and shared GPU memory every second, per adapter and for the
arm's process; the allocator's counters and the device's free memory after each block;
per-step losses; the per-row SHA-256 of the ids.

## 3. What this does NOT measure

- **A real 70B checkpoint.** Shape only, random values; no quality or correctness claim. The
  losses at batch 2 and 3 are not compared with anything.
- **Batch above 3, any seq other than 512, packing, or gradient accumulation.** Accumulation
  re-reads the model for every micro-batch; nothing here measures it.
- **L2L itself.** This is its control arm (§1a), not a test of it.
- **`origin/main`'s reader.** The SINGLE arms run the PR #1536 per-drive dispatcher with one
  drive (§1).
- **The setup path.** `soup train` reaches a batch through the stream setup and its VRAM
  pre-flight (the fitted formula, or `stream_vram_probe`); the harness builds the model
  directly and checks nothing before allocating. "Fits" here means it ran on the card, not
  that `soup train` would accept the config.
- **A spill too small to slow the step past read plus compute in series**, as a judged
  quantity (§2, note 5).
- **Sustained load.** Each arm reads for at most about a minute; the sustained probe on
  `probe/two-drive-sustained` asks that question.
- **Warm (page-cached) reads, the RAM tier, other drives or boxes.** Both layouts read
  through direct I/O.
- **The read-ahead difference between the layouts** (2 against 3), unchanged from the gate
  (its §7). It does not enter a within-layout ratio, which is all the rule reads.
- **Drive temperature** (unreadable without elevation on this box, as the sustained probe's §3
  records) and ambient conditions. The GPU's temperature and clocks are sampled.

## 4. Box state, per arm, and the commands

The gate's §4 protocol, run by `batch_control_probe.py`, which imports the gate driver's
functions rather than copying them. The same stamp is taken right before and right after every
arm: AC and battery (and whether it is charging), available RAM, commit used and limit, the
Python process count, GPU memory in use and compute apps, the AC ASPM index (read only),
`soup_cli.__file__`, and every process's cumulative read and write bytes, with the eight
processes whose reads grew most named in the after stamp. While an arm runs, the gate's sampler
takes AC and commit every 2 s and the same PDH counters every 1 s (per-volume read and write
bytes for C: and D:, paging, commit, CPU).

What differs from the gate's §4:

- **Files.** Everything goes to `results/probe-rtx5070/batch-control/`: the stamps to
  `batch_box_state.log`, every sample to `batch_driver.json`, each arm's JSON and log to
  `batch_r<round>_<layout>_b<batch>.json` / `.log`, a build arm's to
  `batch_build_<layout>.json` / `.log`. The gate's committed logs are never appended to.
- **Two more samplers** (`batch_control_sampler.py`). `nvidia-smi` every 2 s (memory in use and
  total, SM clock, temperature, power, utilisation, throttle reasons), and the Windows GPU
  memory counters every 1 s (`\GPU Adapter Memory(*)` and `\GPU Process Memory(*)`, Dedicated
  and Shared Usage; the per-process rows are kept for the arm's own pid, and the query is
  reopened every 5 s until that pid appears, so they do not depend on PDH picking up a process
  that started after it).
- **The GPU must be empty before an arm**: 0 MiB in use, where the gate allowed 1024. Every
  stamp of the gate's four sequences read 0 MiB (its §5.6, §5.12, §5.21), and at this probe's
  first dry run a peer's CUDA process held 752 MiB, which the gate's bar would have let
  through. A fit measurement cannot share the card.
- **A fourth void condition**, as in the sustained probe: a process other than the driver that
  read at least 1 GB during the arm (§2's threshold table). If the per-process capture fails in
  either stamp the check has no input for that arm; that is recorded and does not void it.
- **Out of memory, the wall cap and a spill are results, not voids** (§2, notes 5 and 6).
  Every other arm with no `step_plain` point, and every arm that saw battery power, a suspend
  (two power samples more than 30 s apart) or a foreign reader, is void: kept under a `_void`
  name and re-run once in the same position when the pre-arm checks hold again. A second void
  in the same position stops the sequence, and row 2 gives no verdict.
- **Skips.** After an arm that does not fit (row 4; the driver applies (iv) live against the
  latest batch-1 step of that layout), every later position of that layout at that batch or
  above is not run; it is listed as skipped, not as missing. The verdict recomputes row 4 from
  the files with each round's own batch-1 step.
- **Stopping early.** After each round arm the driver applies rows 1 and 2, the two a single
  arm can settle, and stops if either fires. Row 3 does not stop the sequence: SINGLE can still
  be judged.
- **Settle: 300 s** between the build arm(s) and the first round arm, whichever order this
  probe and the sustained one run in: either the sustained probe has just read both drives for
  about 40 minutes, or this probe's build has just written the striped cache (about 36 GB over
  four minutes, the gate's §5.1). A judgement, not a measurement, and the same figure the
  sustained probe chose after its build.

Unchanged from the gate: AC on and commit headroom at least 8 GiB before every arm, polled every
30 s for at most 15 minutes, after which the sequence stops and the arms not run are listed.
One process per arm, never two at once.

**The run order between this probe and the sustained one.** The commands below assume nothing:
the driver's cache check decides the build. As of 20:16:35 on 2026-10-01 the striped cache under
the PR's folder naming exists (the sustained probe's build made it), so the STRIPED build arm is
one cold step. If the cache were missing, the same command would re-shard it first; the cache it
writes is the one the sustained probe reads, from the same code and paths, and that probe's own
build arm would then find a hit.

**Estimated wall time**, from the gate's own figures and §21's model, without voids or pre-arm
waits (the driver prints the same estimate):

| part | estimate | from |
|---|---|---|
| STRIPED build arm | 0.7 min (hit) / 4.6 min (re-shard) | process start, build and one cold step; the gate's build, 275.9 s (§5.1) |
| settle | 5.0 min | above |
| 6 SINGLE arms | 110.5 / 131.2 / 151.9 s at batch 1 / 2 / 3, twice: 13.1 min | 15 s of process start (the gate's arms' wall time minus their steps, 13-15 s) + 4 steps at max(17.44 s, B) + 4 B steps |
| 6 STRIPED arms | 54.9 / 61.4 / 82.2 s, twice: 6.6 min | 15 s + 4 steps at max(9.97 s, B) |
| **total** | **about 25 min, 29 with a re-shard** | each void re-run adds 1-2.5 min, each pre-arm wait up to 15 |

**The commands, in order** — complete for whoever runs them (Git Bash, from the worktree, with
the base interpreter `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`, not a
venv; the driver sets each arm's `PYTHONPATH` to `Soup-r2/src` itself):

```
# 0. Before anything: the laptop on AC with the lid open; the other Claude sessions asked by
#    message to run no CUDA work, no test suite and no large copy on C: or D: until this probe
#    reports done; the sustained probe, if it is running, finished.
cd C:/Users/user/projects/Soup-r2
git status --porcelain -- src benchmarks/harness      # must print nothing
git log -1 --format='%h %s'                           # the rule commit (or a later records-only one)
PYTHONPATH="C:/Users/user/projects/Soup-r2/src" python -c "import soup_cli; print(soup_cli.__file__)"
                                                      # must print a path under Soup-r2\src
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must print 0 MiB

# 1. Optional: the dry run, into a scratch folder only (stamps, the cache check, the estimate,
#    a 4-s sampler self-test; no arm).
python benchmarks/harness/batch_control_probe.py --dry-run --out-dir "C:/Users/user/AppData/Local/Temp/r2-dry"

# 2. The probe: the cache check, the build arm(s), the 300 s settle, the twelve round arms; it
#    prints --summarize's table at the end. About 25 minutes (29 with a re-shard).
python benchmarks/harness/batch_control_probe.py --plan all

# 3. Recompute §2 from the committed files alone, at any time after.
python benchmarks/harness/batch_control_probe.py --summarize
```

Step 2 exits 0 when the sequence completes, 4 when a row-1 or row-2 stop or a second void ended
it, and 5 when a pre-arm wait expired; in every case `batch_driver.json` says why and which
positions did not run. It has no resume: a stopped sequence is re-run whole with a new
`--run-prefix` (`batch` plus up to eight lowercase letters or digits), so the first one's files
stay as measured. The results to commit are everything under
`benchmarks/results/probe-rtx5070/batch-control/`, `_void` files included.

## 5. Results

### 5.1 Attempt 1 — 2026-10-01 23:35, `batch_control_probe.py --plan all`, NO VERDICT

Run from `5a4df9be` (the rule commit; `src/` and `benchmarks/harness/` clean, `soup_cli.__file__`
under `Soup-r2\src`), with `--run-prefix batch`. AC online and charging at the first stamp
(73%, 19.15 GB available RAM, commit 29.91 of 60.91 GB, GPU 0 MiB, no compute apps); the four
other Claude sessions on the box had acknowledged a hold for 23:35-00:20 and were idle. Both
caches were a hit (SINGLE and STRIPED), so the build arm was one cold step.

| position | arm | wall | `step_s` (3 timed steps) | judged |
|---|---|---|---|---|
| build | STRIPED b1, 1 step, 0 warm-up | 52.4 s | 23.44 | never (build arm) |
| 0, try 1 | SINGLE b1 | 173.2 s | 31.44 / 32.77 / 33.95 (mean 32.722) | VOID: foreign reader `SearchIndexer.exe` 4.94 GB |
| 0, try 2 | SINGLE b1 | 52,762.8 s | 18.22 / 30.08 / 31.42 (mean 26.575) | VOID: AC values seen `[0, 1]`; suspended, 52,599 s between two power samples |

The second void is a stop by the rule's own text (§4: "A second void in the same position stops
the sequence, and row 2 gives no verdict"). The 11 later positions did not run. `--summarize`
reads: `NO VERDICT (row 2): the arms did not measure the rule` (no valid round arm on record; row 1
does not fire on the two void arms' own steps, which the rule does not read). Every file of the
attempt is kept under `results/probe-rtx5070/batch-control/` as measured, the two void arms under
their `_void` / `_void2` names. The sequence has no resume; attempt 2 runs whole with a new
`--run-prefix`.

**What happened, as far as the files say.**

- *The suspend.* The laptop went to sleep seconds after try 2 started (23:44:32) and woke at
  14:23:55 on 2026-10-02, on battery (AC 0, 79%). The Python process survived the sleep, so
  the arm finished and the driver stamped it. The cause of the sleep is not in the files; the box
  state log shows only the AC flip. Nothing of the driver can prevent it.
- *The slow first arm.* Try 1's three steps were 31.4-34.0 s, each with 157 + 2 loads, against
  17.4 s for the same arm in the gate (§1). Compute was not the cause: in the same process the
  read-free floor (`B_nocopy_r0`) ran 6.658 s (mean of 3; §21: 6.425 s). The step is the read:
  70,378,258,732 B in 31-34 s, and the 1-s PDH counters put C:'s read rate at **a steady
  2.0-2.2 GB/s through all three steps** (per-step means 2.17 / 2.11 / 2.03 GB/s; the C: bytes
  read inside each step window, 69.6 / 67.4 / 69.1 GB, are the step's own), against ~4.1 GB/s
  averaged in §21. D: read 0.
- *The indexer is not shown to be the cause.* `SearchIndexer.exe` read 4,940 MB and wrote 643 MB
  in the arm (it read 258 MB during the build arm and 523 MB in the whole of try 2). Its share
  of C:'s traffic in the arm is about 1.8% (4.94 GB of the 275.0 GB the 1-s PDH counters summed
  for C:), and the C: rate did not move in bursts. Its small reads could still have cost latency; the files cannot say. What its reads
  were of (the layer cache, the many worktrees on this profile, anything else) is not recorded
  anywhere: the per-process counters carry bytes, not paths.
- *A pattern inside try 2.* 18.22 s for its first step, which is inside the gate's band for this
  arm, then 30.08 and 31.42 s: the rate halved during the arm, and the arm began after a 14.6-h
  idle. One arm is not a pattern; it is the reason attempt 2 reads C:'s per-step rate before
  anything else. The two void arms' power and clock samples (32.5 W / 795 MHz median in try 1) are
  in `batch_driver.json`, and are not judged.

**Not concluded:** why C: read at half rate; that the indexer, the drive's temperature (not
readable without elevation on this box), a power state or something else is the cause. The
rule's row 1 (a SINGLE batch-1 step outside 15.0-21.5 s) is exactly the check that would fire on
this, and attempt 2 is bound by it as written.

### 5.2 Attempt 2 — 2026-10-07 10:15:46Z, `batch_control_probe.py --plan all --run-prefix batch2`, COMPLETE

Run from `aef2d31a` (attempt 1's records commit; `src/` and `benchmarks/harness/` clean,
`soup_cli.__file__` under `Soup-r2\src`). `src/` is byte-identical to `2c6ed087` and also to the
head PR #1536 merged at (`0130e025`, 2026-10-04): the PR's later commits touch only benchmarks and
docs (`git diff --stat 2c6ed087 0130e025 -- src/` is empty). Run by session measure-28 for soup-28
inside the owner's machine window (10:07Z), with a hold listed in `.claude/SESSIONS.md`. Times
below are UTC; the box-state log stamps local time (UTC+5).

**Box.** AC online and battery 100% at every stamp; GPU 0 MiB before every arm; commit headroom
21.3-28.1 GiB. Two preconditions of the runbook were not met, and the run went ahead under the
ruling recorded on 2026-10-04 (handoff, R2 section, UPDATE item 3: run once anyway after
SearchIndexer.exe has read ~0 MB/s for 5 minutes in a row):

- **Windows Search was RUNNING** (no session is elevated; the owner had not stopped it).
  SearchIndexer.exe's cumulative read count stayed flat at 1,150.789 GB from 10:10:10 to
  10:15:26Z (10-s samples), and the run started at 10:15:46Z. SearchIndexer.exe is not among the
  eight largest readers of any arm of this attempt.
- **ChatGPT.exe was open.** Its on-device-model process (PID 17064) is listed as a GPU compute app
  at every stamp, with 0 MiB in use, so the driver's pre-arm check (0 MiB) held.

Both caches hit; the STRIPED build arm was one cold step (wall 46.1 s, step 26.949 s,
`shard_seconds` 0.025 s), then the 300-s settle. The sequence ran 30.1 min against the 25.4-min
estimate (one void re-run, below).

**One void, re-run once in the same position, as §4 prescribes.** Round 0, position 4 (SINGLE b3),
try 1: foreign reader `BackgroundDownload.exe` (PID 3716, started during the arm) read 3.10 GB;
`MoUsoCoreWorker.exe` (Windows Update's orchestrator) also started inside it (45.4 MB). Its steps,
20.455 / 20.791 / 20.736 s (mean 20.661), are kept under `batch2_r0_single_b3_void.*` and are not
read by the rule. The re-run was valid. The largest foreign reader in any valid arm was
`svchost.exe` (PID 16716) at 376.3 MB, in round 1 SINGLE b1.

**Every valid arm:** `direct_io` and `pinned` true, `AsyncDiskSource`, read-ahead 2 (SINGLE) or 3
(STRIPED), 157 + 2 loads per timed step, 0 allocator retries, `shard_seconds` 0.004-0.008 s (cache
hits).

| round | pos | arm | timed steps (s) | `step` | tok/s | peak alloc / reserved (GB) | in use (MiB) | B read-free (s) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | SINGLE b1 | 18.712 / 19.179 / 19.547 | 19.146 | 26.74 | 4.378 / 4.798 | 4,750 | 6.511 |
| 0 | 1 | STRIPED b1 | 9.149 / 9.144 / 9.030 | 9.108 | 56.22 | 4.378 / 4.798 | 4,750 | — |
| 0 | 2 | SINGLE b2 | 17.781 / 18.077 / 17.910 | 17.923 | 57.13 | 5.345 / 5.545 | 5,462 | 12.074 |
| 0 | 3 | STRIPED b2 | 12.391 / 12.588 / 12.413 | 12.464 | 82.16 | 5.345 / 5.545 | 5,462 | — |
| 0 | 4 | SINGLE b3 (re-run) | 20.487 / 20.615 / 20.162 | 20.422 | 75.21 | 6.312 / 7.028 | 6,876 | 17.567 |
| 0 | 5 | STRIPED b3 | 17.748 / 17.842 / 17.760 | 17.784 | 86.37 | 6.312 / 7.028 | 6,876 | — |
| 1 | 0 | STRIPED b3 | 17.793 / 17.763 / 17.767 | 17.774 | 86.42 | 6.312 / 7.028 | 6,876 | — |
| 1 | 1 | SINGLE b3 | 20.322 / 20.274 / 20.405 | 20.334 | 75.54 | 6.312 / 7.028 | 6,876 | 17.503 |
| 1 | 2 | STRIPED b2 | 12.268 / 12.436 / 12.232 | 12.312 | 83.17 | 5.345 / 5.545 | 5,462 | — |
| 1 | 3 | SINGLE b2 | 17.640 / 17.948 / 17.743 | 17.777 | 57.60 | 5.345 / 5.545 | 5,462 | 11.913 |
| 1 | 4 | STRIPED b1 | 8.515 / 8.650 / 8.548 | 8.571 | 59.74 | 4.378 / 4.798 | 4,750 | — |
| 1 | 5 | SINGLE b1 | 17.491 / 17.626 / 17.408 | 17.508 | 29.24 | 4.378 / 4.798 | 4,750 | 6.490 |

**What §2 note 5 asks for, for every batch-2 and batch-3 arm** (the card is 8.518 GB; both rounds
read the same): process dedicated / shared GPU memory 5.731 / 2.229 GB (SINGLE b2), 7.214 / 2.229
(SINGLE b3), 5.733 / 3.303 (STRIPED b2), 7.216 / 3.303 (STRIPED b3); `peak_reserved_gb` 5.545 at
batch 2 and 7.028 at batch 3; `num_alloc_retries` 0 everywhere. The process's shared memory is the
same at batch 1 (2.229 GB SINGLE, 3.303 GB STRIPED), so it does not grow with the batch.

**Power and SM clock over the timed steps of every batch-3 arm** (median / minimum; utilisation
100% in all four): SINGLE b3 round 0 96.6 / 76.5 W, 2141 / 1620 MHz; round 1 106.7 / 11.3 W,
2362 / 1230 MHz. STRIPED b3 round 0 110.1 / 82.0 W, 2415 / 2340 MHz; round 1 112.8 / 79.3 W,
2422 / 2130 MHz.

**The read-free floor, in session** (mean of the two rounds' B): 6.500 s at 512 tokens (model
6.425 s), 11.993 s at 1024 (model 11.609 s), 17.535 s at 1536 (model 16.792 s). The least-squares
line through them: 0.975 s + 0.010776 s x tokens (largest residual 0.016 s). Its crossover with
this session's R: SINGLE 1,610 tokens, STRIPED 730 tokens.

**The ids:** row 0 of the batch-2 draw and row 0 of the batch-3 draw both equal the batch-1 draw.

**The peak VRAM arithmetic of §2** (judged by nothing) predicted 5.345 GB allocated at batch 2 and
6.312 GB at batch 3; the arms read 5.345 and 6.312.

**The rule's output**, `python benchmarks/harness/batch_control_probe.py --summarize --run-prefix
batch2`, recomputed from the committed files. It is identical to the table the driver printed at
the end of the run:

```
row 1 SINGLE b1 outside 15.0-21.5 s: no fire
row 2 off the path, or the protocol stopped: no fire
row 3 STRIPED b1 outside 7.69-12.0 s: no fire
row 4 SINGLE b2: fits
row 4 SINGLE b3: fits
row 5 SINGLE b2: NEAR-FREE: e 1.027 (r0 1.068, r1 0.985), tok/s ratio 2.053
row 5 SINGLE b3: PARTIAL: e 0.899 (r0 0.938, r1 0.861), tok/s ratio 2.698
row 6 SINGLE b2: AGREES: measured NEAR-FREE, model NEAR-FREE (e_pred 1.000)
row 6 SINGLE b3: DISAGREES AT b3: measured PARTIAL, model NEAR-FREE (e_pred 1.000)
row 6 SINGLE crossover: measured (1024, 1536] tokens; model 1688 tokens at R 18.327 s
row 4 STRIPED b2: fits
row 4 STRIPED b3: fits
row 5 STRIPED b2: PARTIAL: e 0.713 (r0 0.731, r1 0.696), tok/s ratio 1.427
row 5 STRIPED b3: PARTIAL: e 0.497 (r0 0.512, r1 0.482), tok/s ratio 1.492
row 6 STRIPED b2: AGREES: measured PARTIAL, model PARTIAL (e_pred 0.761)
row 6 STRIPED b3: AGREES: measured PARTIAL, model PARTIAL (e_pred 0.526)
row 6 STRIPED crossover: measured (512, 1024] tokens; model 750 tokens at R 8.839 s
row VERDICT: per layout: rows 4-6 above
```

**Also recorded, not judged.**

- Round 0's SINGLE b1 (19.146 s, steps rising 18.71 -> 19.55 s) is 9.4% slower than round 1's
  (17.508 s). The reversed second round cancels a linear drift only in the mean, which is what row
  5 reads (§2, note 1); both rounds' `e` are quoted beside each class above. Attempt 1's half-rate
  C: read did not recur: every SINGLE batch-1 step of this attempt was 17.41-19.55 s, against
  30.1-34.0 s for five of attempt 1's six (the sixth, try 2's first step, read 18.22 s). Why
  round 0's first arm was slower is not established.
- The second drive at batch 3: SINGLE b3 20.378 s against STRIPED b3 17.779 s (two-round means),
  1.146x, against 2.073x at batch 1 (18.327 / 8.839 s). §0's arithmetic expected about 4% at
  batch 3 (17.44 / 16.79 s). The STRIPED batch-3 step sits 1.4% above the in-session floor at 1536
  tokens.

## 6. Verdict

From attempt 2 (§5.2), by the rule of §2 as committed, applied by `batch_control_rule.py`:

**Rows 1-3 do not fire.** SINGLE batch 1 read 19.146 and 17.508 s (band 15.0-21.5 s), no arm was
off the path and the protocol completed, and STRIPED batch 1 read 9.108 and 8.571 s (band
7.69-12.0 s). Both layouts are judged.

**SINGLE, the L2L control** (R = 18.327 s; the code caveat of §1 applies):

| b | row 4 | row 5 class | `ē` (r0, r1) | `r̄` | tok/s r0 / r1 | peak alloc | in use | labels | row 6 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | (NEAR-FREE by definition) | — | 1 | 26.74 / 29.24 | 4.378 GB | 4,750 MiB | — | — |
| 2 | fits | **NEAR-FREE** | 1.027 (1.068, 0.985) | 2.053 | 57.13 / 57.60 | 5.345 GB | 5,462 MiB | none | **AGREES** (model NEAR-FREE, `e_pred` 1.000) |
| 3 | fits | **PARTIAL** | 0.899 (0.938, 0.861) | 2.698 | 75.21 / 75.54 | 6.312 GB | 6,876 MiB | none | **DISAGREES AT b3** (model NEAR-FREE, `e_pred` 1.000) |

Measured crossover bracket **(1024, 1536] tokens**; the model's crossover at this session's R is
1,688 tokens, and the in-session floor line's is 1,610 tokens (§2 note 7). Neither label fires at
batch 3: 6,876 MiB in use is below 7,620 MiB, and the steps (20.422 / 20.334 s) are below read plus
compute in series (35.938 / 34.300 s). **The largest batch that fits is 3**, the largest one run.
§2 note 4 named this outcome before the run: batch 3 sits on the model's edge, and a PARTIAL there
is a DISAGREES that this record reports as such. In this session the floor at 1536 tokens (17.535 s)
is 4.3% under R, and the step exceeds both.

**STRIPED, an extra** (R = 8.839 s):

| b | row 4 | row 5 class | `ē` (r0, r1) | `r̄` | tok/s r0 / r1 | peak alloc | in use | labels | row 6 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | (NEAR-FREE by definition) | — | 1 | 56.22 / 59.74 | 4.378 GB | 4,750 MiB | — | — |
| 2 | fits | **PARTIAL** | 0.713 (0.731, 0.696) | 1.427 | 82.16 / 83.17 | 5.345 GB | 5,462 MiB | none | **AGREES** (model PARTIAL, `e_pred` 0.761) |
| 3 | fits | **PARTIAL** | 0.497 (0.512, 0.482) | 1.492 | 86.37 / 86.42 | 6.312 GB | 6,876 MiB | none | **AGREES** (model PARTIAL, `e_pred` 0.526) |

Measured crossover bracket **(512, 1024] tokens**; the model's is 750 tokens at this session's R,
and the in-session line's is 730 tokens.

Nothing in §2 was edited. The records are local, on `probe/batch-control`, and not pushed.
