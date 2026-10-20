<!--
Working measurement record, published verbatim. The decision rule in §2 and the
protocol in §3 were written in issue #1702 on 2026-10-07, before any run of the
speed series, and are copied here unchanged: a rule written after the numbers
are in is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB, driver 616.92),
Intel i9-14900HX, 31.7 GB DDR5-5600, Samsung PM9B1 NVMe (C:). Windows 11 Pro
10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2, accelerate 1.15.0.
Soup: branch perf/pinned-arena-packing-1702 on main 7016df2b; soup_cli.__file__
is stamped into every run's result.
Harness: benchmarks/harness/issue1702_arena_pack_gate.py, one process per run,
through stream_probe.build (the shipped shard_checkpoint -> build_streamed_model
path). Its schedule and rule are held by tests/test_issue1702_arena_pack_gate.py.
-->

# Gate record — largest-first packing of the pinned arenas (#1702)

**Status (2026-10-07): memory saving confirmed, strict control passed, speed series run:
NO DIFFERENCE measurable at +/-15.1%.**
On `main` plus the change, the Qwen3-8B NF4 RAM tier page-locks 6,442,450,944 bytes
where the in-order plan page-locks 8,589,934,592, and three deterministic optimizer
steps are byte-identical between the two plans (§4). The speed series (§5) completed 40
of 40 runs: the RAM tier's mean difference is +1.96%, inside the band of +/-15.13% the
control set, so by the rule written beforehand no speed-up and no slow-down is claimed.
The band is wide; the series cannot exclude a change smaller than that.

---

## 0. The question

`plan_pinned_arenas` now also plans stable first-fit decreasing and returns it when it
page-locks strictly fewer bytes than the in-order walk. On the RAM tier that changes
where every tensor of the store lives: a layer's tensors are no longer neighbours in
one arena. The host-to-device copies are per tensor and each tensor is still one
contiguous page-locked view, so no effect on the step time is expected. Expected is
not measured. Does the packed layout change the training step time?

## 1. What is settled without a timing

| | in-order plan | packed plan |
|---|---:|---:|
| Qwen3-8B NF4 RAM tier, arenas | 4 x 2 GiB | 3 x 2 GiB |
| page-locked bytes | 8,589,934,592 | 6,442,450,944 |
| store bytes (1,154 tensors) | 6,073,035,760 | 6,073,035,760 |
| peak process RSS, 3-step run | 10.85 GB | 8.71 GB |
| peak VRAM allocated / reserved | 2.807 / 3.047 GB | 2.807 / 3.047 GB |
| Qwen3-8B NF4 disk tier, page-locked staging | 4,294,967,296 | 4,294,967,296 (same plan) |

The first three rows are also arithmetic, and are tested without a GPU
(`tests/test_issue1702_pinned_arena_packing.py`). The other rows are from the four
runs of §4.

## 2. The decision rule (from #1702, written before any run)

1. Failed runs are never dropped. Each is listed with its error and counted against
   its arm. A seed/mode block with a failed run has no `d`. A mode with fewer than 4 of
   5 `d` is INCONCLUSIVE.
2. Noise band `b` = the larger of 3% and the largest `|d|` seen in the control.
3. If the control's mean `d` is outside +/-3%, the series is INCONCLUSIVE: the box was
   not quiet or the order did not cancel. No speed statement is made. One repeat in
   another window is allowed, and both series are published.
4. Otherwise, with `m` = the mean `d` on the RAM tier:
   - `m < -b`: slower. The number goes in the PR and the change is decided as a trade of
     speed for 2 GiB of page-locked memory.
   - `m > +b`: faster, reported with the number.
   - anything else: no difference measurable at +/-`b`; the PR claims memory only.
5. Nothing is tuned after the results are seen: no dropped seed, no extra run beyond the
   one declared repeat.

`d = mean(packed) / mean(in-order) - 1` per seed and mode, over that block's two runs
per arm, on input tokens per second across the 10 measured steps (5,120 positions
divided by the sum of the 10 CUDA-synchronized step times).

## 3. The protocol (from #1702, written before any run)

- **Recipe.** Qwen/Qwen3-8B (revision `b968826d9c46dd6066d109eabc6255188de91218`), NF4
  with double quantization, bf16 compute, LoRA r16 (alpha 32, dropout 0) on
  `q_proj`/`v_proj`, `PagedAdamW8bit` at 2e-4, batch 1, 512-token blocks, 5 warm-up and
  10 measured optimizer steps, two GPU layer buffers, read-ahead 2, a
  4,000,000,000-byte per-process allocator cap. Seeds 3 / 17 / 29 / 43 / 59.
- **Data.** 32 blocks of 512 tokens from WikiText-2 raw v1, tokenized without special
  tokens; a run takes the first 15. The file is not in the repository (sha256
  `663f28bf9bf8307a9018c3bf4281d58d053fa55752de01f160101bc83cb0a8ee`). Without
  `--batches` the harness draws seeded random token ids instead and says so in the
  result.
- **Arms.** `in-order`: the planner's largest-first walk is pointed at its in-order walk
  inside the process, so the candidate can never win and the plan is the one from
  before #1702. `packed`: the shipped planner. No config field exists for this.
- **Modes.** RAM tier: the subject. Disk tier with direct I/O: the negative control.
  Its plan is the same in both arms, so a difference between its arms is noise or run
  order.
- **Order.** One series, every run a fresh process. Per seed and mode the four runs go
  A B B A; the next seed starts with the other arm and the other mode.
  5 seeds x 2 modes x 4 runs = 40 runs
  (`python benchmarks/harness/issue1702_arena_pack_gate.py` prints the list).
- **Validity.** A RAM run that does not show 8,589,934,592 page-locked bytes in the
  in-order arm or 6,442,450,944 in the packed arm, or a control whose arms page-lock
  different amounts, is a harness fault and voids the series. A run whose staging is
  not page-locked, a control run that is not reading with direct I/O, and a run with
  no result after 15 minutes (it is stopped) are failed runs.
- **Power** (added 2026-10-07, before any run of the series). A rehearsal of the driver
  on Qwen2.5-1.5B found the laptop on battery: a step took 0.83 s there against 0.31 s
  on AC. The driver does not start a series on battery power and stops one when the
  box goes on battery; a series with any run stamped on battery, before or after, is
  INCONCLUSIVE and counts as the one allowed repeat.
- **The box.** Nothing else runs for the whole series: no test suite, no other GPU job.
  RAM in use, commit charge and AC power are stamped before and after every run.

```bash
python benchmarks/harness/issue1702_arena_pack_gate.py --mode series \
    --weights $MODEL --shards $SHARDS --batches $BATCHES --out $OUT
```

## 4. Strict control: byte-identical (2026-10-07)

Three optimizer steps, one seed (3), `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, one fresh process per arm and mode. After every step
the run takes the SHA-256 of the full logits, the loss, all 144 LoRA gradients and all
144 updated adapter tensors: 290 digests per step, 870 per run.

| mode | in-order | packed | digests equal |
|---|---:|---:|---|
| RAM tier, page-locked bytes | 8,589,934,592 | 6,442,450,944 | 870 of 870 |
| disk tier (direct I/O), page-locked bytes | 4,294,967,296 | 4,294,967,296 | 870 of 870 |

All four runs give the same digest of digests,
`1b127f4880e4f35d07609a1c503728e17ac22e2f0ea0f69f81ba62f023e04140`: the two plans agree
with each other, and the RAM tier agrees with the disk tier. Losses over the three
steps: 2.699491, 3.197587, 3.101483 in every run.
Summary file: `benchmarks/results/probe-rtx5070/issue1702/strict.json`.

```bash
python benchmarks/harness/issue1702_arena_pack_gate.py --mode strict \
    --weights $MODEL --shards $SHARDS --batches $BATCHES --seed 3 --out $OUT
```

This is a statement about three steps under deterministic execution. It shows the two
layouts hand the same bytes to the same computation. It is not a claim about default
(non-deterministic) execution or about a long run.

## 5. Speed series: NO DIFFERENCE measurable at +/-15.1% (2026-10-07)

One series, 16:40Z-17:16Z (35.6 minutes), at `4e692d31`, in a declared window with no
test suite and no other GPU job on the box. 40 of 40 runs completed; no failed run and
no harness fault. Every stamp shows AC power; before the runs 13.3-16.5 GB of RAM was
available and the commit charge was 52.6-56.8%. The Windows Search service was running.
Files: `benchmarks/results/probe-rtx5070/issue1702/series.jsonl` (one line per run, in
run order) and `verdict.json`.

| seed | RAM tier `d` | control `d` |
|---:|---:|---:|
| 3 | -2.71% | +3.03% |
| 17 | +4.75% | -0.39% |
| 29 | -1.41% | +9.69% |
| 43 | +3.58% | -0.34% |
| 59 | +5.61% | -15.13% |
| mean (sample SD) | **+1.96%** (3.77%) | -0.63% (9.09%) |

By §2: the control's mean is -0.63%, inside +/-3%, so the series stands (rule 3). The
noise band `b` is 15.13%, the largest `|d|` in the control (rule 2). The RAM tier's mean
`d` of +1.96% is inside +/-15.13% (rule 4): **no difference measurable at +/-15.13%.
The PR claims memory only.** No repeat is due: the rule allows one only after an
inconclusive series.

| mode, arm (10 runs each) | tokens/s, mean (SD) | median step | store build | peak RSS | page-locked bytes |
|---|---:|---:|---:|---:|---:|
| RAM, in-order | 515.8 (30.7) | 0.972 s | 11.5 s | 10.84 GB | 8,589,934,592 |
| RAM, packed | 525.5 (31.5) | 0.963 s | 9.0 s | 8.69 GB | 6,442,450,944 |
| disk, in-order | 273.1 (24.2) | 1.866 s | 1.7 s | 6.13 GB | 4,294,967,296 |
| disk, packed | 270.6 (22.2) | 1.882 s | 1.5 s | 6.12 GB | 4,294,967,296 |

Peak VRAM was 2.807 GB allocated and 3.047 GB reserved in every run.

**What this resolution means.** The band is wide because the control ran in two
regimes. 15 of its 20 runs have a median step of 1.86-2.01 s and 5 have 1.62-1.64 s:
the first three runs of seed 29 (one in-order, two packed) and the first and last runs
of seed 59 (both in-order). A block split between the two regimes gives +9.69% and
-15.13%; the other three blocks are within about 3%. The plan is the same in both arms
on the disk tier, so this is the box and not the change, which is what a control is
for. The RAM tier shows no such split (its median steps span 0.875-1.102 s, and its
five `d` span -2.71% to +5.61%), but the rule takes the band from the control and the
rule was written first. So: this series cannot exclude a change of less than 15% in
either direction, and it shows none.

## 6. What this record does not show

- A physical 4 GB card: the card has 8 GB and the runs are under a software allocator
  cap, which does not bound the CUDA context, the driver or page-locked memory.
- Any model other than Qwen3-8B. The planner tests cover two Qwen2.5-14B geometries as
  arithmetic only (about 7% less page-locked memory on each).
- That a page-lock refusal becomes less likely. The packed plan asks Windows for 2 GiB
  less, and that is all that is known.
