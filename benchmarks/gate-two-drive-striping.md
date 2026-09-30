<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any arm ran — before the striped cache existed —
which is the point of it: a rule written after the numbers are in is not a
rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (MZAL81T0HFLB-00BL2, 954 GB each):
C: on a CPU root port (PEG0, 00:06.0), D: behind the chipset (RP13, 00:1D.4,
over the DMI 4.0 x8 link). Windows 11 Pro 10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2.
Soup: branch feat/two-drive-striping at f798c46e (the commit this gate starts
from), worktree C:\Users\user\projects\Soup-stripe; soup_cli.__file__ is
stamped under Soup-stripe\src before every arm.
Harness: benchmarks/harness/stream_probe.py (the cold 70B step of gate-971 and
gate-974; --stripe-dir and the direct-I/O / placement stamps added in
f798c46e), one process per arm.
-->

# Gate record — two-drive striping of the disk-tier cache (R4)

**Status: rule committed before the run; results pending.**

*Amended 2026-09-30, still before any arm ran:* §4 gains a third void
condition, a suspended box. The first build attempt timed out because the
laptop slept with its lid closed (2026-09-28/29), and a step that spans a
suspend reads as a slow step. §2 is unchanged.

---

## 0. The question

The cold 70B-shaped NF4 step on this box is bound by the read (17.0-17.2 s at
seq 512 and 256, `probe-rtx5070-what-bounds-streaming.md` §21). Two drives read
7.65-9.15 GB/s together against ~4 for one (`probe-rtx5070-two-drive-read.md`,
no formal verdict — its validity row fired three times). This branch writes
decoder layer i to drive i mod 2 and keeps one read in flight per drive. Does
the training step get faster, and by how much?

What the rule's two thresholds mean, as arithmetic and not as a prediction. The
step moves 70.38 GB (157 decoder-layer loads of 441.4 MB plus the embedding and
the head, 536.9 MB each; gate-974 §8). **12.0 s** is 5.87 GB/s averaged over
the step, above the best rate the unbuffered read primitive ever reached on C:
(5.65 GB/s, gate-974 §4), the drive the single-root cache lives on: a step at
or under 12.0 s is not reachable from C: alone at any rate this box has
recorded. **14.0 s** is 5.03 GB/s, 1.2x the 4.09-4.14 GB/s the single-drive
step averages. The floor with the read removed is 6.43 s at seq 512 (§21, arm
B), so no layout takes the step below that.

## 1. The fixture and the two arms

Weights `D:/synth/llama-70b-shape-v32k-f4` — the synthetic Llama-3.1-70B-shaped
bf16 source with a 32k head, random values (`harness/synth_checkpoint.py`);
untied, so the embedding and the head stream through the large-layer slot. NF4,
disk tier, seq 512, batch 1.

- **Arm SINGLE:** the existing one-root cache under
  `C:\Users\user\.soup\layer-stream\` (default `--shards`), `--read-ahead 2`
  (the shipped default).
- **Arm STRIPED:** `--shards
  C:/Users/user/.soup/layer-stream-striped/D___synth__llama-70b-shape-v32k-f4
  --stripe-dir D:/soup-stripe --read-ahead 3` (the feature's default for two
  drives).
- Both arms pinned (the harness default).

The command per arm, with the arm's flags from above in place of `<arm flags>`:

```
PYTHONPATH="C:/Users/user/projects/Soup-stripe/src" python benchmarks/harness/stream_probe.py --weights "D:/synth/llama-70b-shape-v32k-f4" --quant nf4 --tier disk --seq 512 --batch 1 --steps 3 --warmup 1 --step <arm flags> --label "R4 gate round <r> arm <ARM>" --out benchmarks/results/probe-rtx5070/two-drive/gate_r<r>_<arm>.json
```

Everything else is the harness's default and the same in both arms: LoRA r 8 on
q/k/v/o, adapter seed 3, input seed 17, `PagedAdamW8bit`, 2 layer buffers,
`read_ranges` 4. `--read-ahead 3` is passed explicitly because the harness
builds the model through `shard_checkpoint` -> `build_streamed_model` directly;
the N + 1 default for N roots lives in the setup path, which the harness does
not go through. Without the flag the gate would measure a striped cache at
depth 2.

The striped layout: decoder layer i goes to root i mod 2. Root 0 is the
`--shards` folder on C:, which also keeps `extras.safetensors`, the two
`large_*.safetensors` files (embedding, head) and `index.json`; root 1 is
`D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4`, a per-model folder the
sharder creates inside the empty `D:\soup-stripe` made for this gate (the
owner approved it). The 80 decoder layers split 40 / 40 and C: carries the two
large files besides — from the file sizes, roughly 35.7 GB of each step's
reads land on C: and 34.7 GB on D: (arithmetic, not measured).

**Before the rounds, not timed.** The striped cache does not exist yet. It is
built by one STRIPED run with `--steps 1 --warmup 0 --step --skip-events`, which
reads the bf16 source on D: and writes the even layers to the C: folder above
and the odd ones to `D:\soup-stripe\...`; its sharding time and its live
placement line (`direct_io`, layer roots 0,1,0,1,...) are recorded. The SINGLE
cache was checked on the CPU with the sharder's own cache check
(`inspect_shard_cache`, with the arguments the harness passes: bfloat16, nf4,
double quant, device kind cuda, no stripe roots) at 22:57 on 2026-09-28: **hit,
"cache is reusable"** — index format 5, 80 layers, written 2026-09-12 — so it is
reused as is and gets no build run. Build runs are published and are not rounds.

## 2. The decision rule, written before the run

Copied verbatim from the task brief (§2 of `task-8-gate-brief.md`). Nothing in
this section is edited once a number exists.

| measured, per arm: `step_s_mean` of the `--step` block's uninstrumented steps (3 timed after 1 warm-up) | verdict |
|---|---|
| any SINGLE arm outside **15.0-21.5 s** — gate-974 §8's own cold rows are 16.175 s (15.77-17.06) and 17.311 s (16.28-20.42), widened ~5% each side | **no verdict**: the session does not reproduce the known step; record and stop |
| either arm with `direct_io` false or `pinned` false | **no verdict**: the reader did not run the path under test |
| STRIPED <= **12.0 s** in every round | **SHIP**; quote the median and the speed-up over the same round's SINGLE |
| STRIPED <= **14.0 s** in every round, the row above not met | **SHIP, stating the measured number** and that the 12 s target was missed |
| anything else | **DO NOT SHIP**; record why |

Also recorded, not part of the verdict: per-step losses of the two arms in a round (same seeds, same bytes, so equal losses are expected — a difference is reported and investigated), `implied_h2d_gb_per_s`, `stall_share`, peak VRAM, SM clock.

Rounds: three, interleaved, first arm alternating — round 0: SINGLE, STRIPED; round 1: STRIPED, SINGLE; round 2: SINGLE, STRIPED. One process per arm.

**The inputs, as the JSON names them** (fixed with the rule, before the run):

- `step_s_mean` is the `kind: "step"`, `label: "step_plain"` record of the
  arm's JSON. `--step` then runs the same steps again with the CUDA-event
  brackets on (`step_events`); that point is recorded and not judged.
- `direct_io` and `pinned` are `meta.direct_io` and `meta.pinned`.
- The speed-up is the round's SINGLE `step_s_mean` divided by its STRIPED
  `step_s_mean`; "the median" is the median of the three STRIPED means.
- `stall_share` is read from `step_events`: the plain point does not bracket
  the compute stream's waits, so its `stall_share` is 0 by construction.
  `implied_h2d_gb_per_s`, `peak_alloc_gb` and the two SM clock samples are
  printed from both points.
- Numbers are quoted from the JSON at three decimals; none is rounded in a way
  that moves a row.

**The losses, and what a difference would mean.** Equal losses are the
expectation the rule states, and the record already holds evidence that they
will NOT be bit-identical across processes even for one arm: gate-974 §8's NEW
arm at two positions (same code, same cache, same seeds) reads 11.835622 and
11.836516 on its first timed step. The backward kernels on this card are not
deterministic, and every timed step here comes after at least one optimizer
update. So a SINGLE-vs-STRIPED difference within a round is read against the
same-arm difference across rounds (SINGLE against SINGLE, STRIPED against
STRIPED), and "same bytes" is checked directly: after the six arms, never
between them, the striped cache is compared tensor by tensor with the
single-root cache.

## 3. What this does NOT measure

- **A real 70B checkpoint.** Shape only, random values — no quality claim.
- **Warm (page-cached) reads.** Both arms read through direct I/O (the rule's
  second row gives no verdict otherwise), which never touches the page cache,
  so every read is cold whatever ran before it.
- **More than two drives**, or any other pair: one PM9B1 on CPU lanes and one
  behind the chipset is this box's layout, not a general one.
- **Batch > 1**, and any seq other than 512.
- **The RAM tier.**
- **The setup path.** `soup train` reaches striping through the stream setup,
  which applies the N + 1 read-ahead default and the distinct-volume and NVMe
  checks; the harness calls the sharder and the model builder directly and
  passes the depth itself, so none of that wiring is exercised here.
- **Sustained load.** An arm is eight steps (the plain point and the
  instrumented one); a DRAM-less drive's thermal behaviour over a long run is
  not exercised.

## 4. Box state, per arm

Other Claude sessions run test suites on this box. Before the build run they
are asked, by message, to avoid CUDA runs until the gate reports done. Right
before each arm, and again right after it, one stamp is appended to
`benchmarks/results/probe-rtx5070/two-drive/gate_box_state.log`:

- AC status and battery % (`GetSystemPowerStatus`);
- available physical RAM, and commit used / limit (`GlobalMemoryStatusEx`,
  whose page-file fields are the counters behind `Win32_OperatingSystem`'s
  `TotalVirtualMemorySize` / `FreeVirtualMemory`; checked equal on this box
  before the run);
- the Python process count (`tasklist`, the driver itself included);
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory` and the GPU's
  total memory in use (on WDDM the per-process column prints `[N/A]`, so the
  total is what bounds any one process);
- the active scheme's AC ASPM index (`powercfg /q SCHEME_CURRENT SUB_PCIEXPRESS
  ASPM`, read only — this gate changes no setting);
- `soup_cli.__file__` as the arm's own environment resolves it.

**Before an arm starts:** AC on, GPU memory in use <= 1024 MiB, commit headroom
(limit - used) >= 8 GiB. The driver waits for those, polling every 30 s, for at
most 15 minutes; if they do not hold by then the sequence stops and the arms
not run are listed as not run. The first round starts at least 60 s after the
build run ends: the sharder does not flush its writes, and the lazy writer
should not be draining them onto either drive while an arm reads.

**While an arm runs**, the driver samples AC status and commit every 2 s and,
through PDH (English counter names), `\LogicalDisk(C:)` and `\LogicalDisk(D:)`
read and write bytes/s, `\Memory\Pages/sec`, `\Memory\Committed Bytes`,
`\Processor(_Total)\% Processor Time` and
`\Processor Information(_Total)\% Processor Performance` every 1 s. The harness
only reads the two drives, so a write on either one during an arm is someone
else's I/O. These are recorded, never used to set a verdict.

**A void arm.** An arm during which any AC sample reads battery, during which
the box was suspended (two consecutive 2-s power samples more than 30 s apart:
a closed lid sleeps this laptop, and the sampler freezes with it), or that exits
without writing its `step_plain` point, is void: its JSON and log are kept and
published under a `_void` name, and it is re-run once, in the same position,
when the pre-arm checks hold again. A second void in the same position ends the
gate with no verdict.

**Stopping early.** After each arm the driver applies the two rows a single arm
can decide — a SINGLE arm outside 15.0-21.5 s, and `direct_io` or `pinned` not
true in any arm — and stops if either fires, because no later arm can turn a
no-verdict outcome into a verdict. Nothing else stops the sequence.

The driver is `benchmarks/harness/two_drive_stripe_gate.py`, committed on its
own after this file and before the build run. It runs the §1 command, one arm
per process and never two at once, writes each arm's stdout and stderr to its
own `.log` beside its JSON, and keeps every stamp and sample in
`gate_driver.json`.

## 5. Results

*Pending.*

## 6. Verdict

*Pending.*
