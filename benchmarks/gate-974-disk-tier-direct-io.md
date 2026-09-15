<!--
Working measurement record, published verbatim. A hypothesis the handoff
carried in as the first thing to test (refuted in 30 s), a 2.4x-slow block
that would not come back, the buffered ceiling nobody had measured, and the
numbers that changed the fix: kept in the order they happened.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB by nvidia-smi /
8.518 GB by torch, driver 616.92), Intel i9-14900HX, 31.7 GB DDR5-5600, two
Samsung PM9B1 NVMe (the NF4 shard caches on C:, the bf16 fixtures on D:),
Windows 11 Pro 26100. Stack: Python 3.12.10 · torch 2.14.0+cu130 ·
transformers 5.17.0 · peft 0.20.0 · bitsandbytes 0.50.2.
Soup: branch fix/stream-974 (cut from fix/stream-901 at f3cfa85c — the #901
arena packer this reuses — and rebased onto main after #989 merged), worktree
C:\Users\user\projects\Soup-stream, own venv, soup_cli.__file__ checked under
Soup-stream\src before every run.
Harness: benchmarks/harness/issue974_warm_stages.py (stage timers over the
shipped build), benchmarks/harness/issue974_cache_populate.py,
issue974_cold_ranges.py, issue974_unbuffered.py, issue974_evict.py (the read
primitives), benchmarks/harness/stream_probe.py (the cold 70B step, as in
gate-971). Raw JSON under benchmarks/results/probe-rtx5070/issue974/.
Fixtures: the unsloth/mistral-7b-instruct-v0.3 NF4 shard cache (4.14 GB, the
warm store, fits the page cache) and the synthetic Llama-70B-shaped NF4 cache
from gate-971 (36.39 GB, larger than RAM, never fully cached).
-->

# Gate record — #974: the disk tier's reads, warm and cold

**Status: the warm regression #974 was filed over does not exist
position-matched (0.91-0.97x of the synchronous control in both run orders);
the handoff's page-cache hypothesis for the one 2.4x-slow block is REFUTED and
that block did not come back; and the cold read — 84.7% of a 70B step in
gate-971 — is bounded by BUFFERED I/O at 1.2-2.9 GB/s on this drive whatever
the request shape, where unbuffered I/O reads the same layers at 3.5-5.65
GB/s.** The change this record gates therefore reads each layer's data section
as sector-aligned byte ranges through direct I/O (`FILE_FLAG_NO_BUFFERING` /
`O_DIRECT`), K ranges in parallel, into one staging region per slot packed
into the #901 pinned arenas. Its own numbers are in §6 onward.

Unit convention: decimal GB and GB/s unless a figure is a power of two, which
is written in GiB. Every block states the host baseline it ran under (free
physical RAM, commit charge, Python processes), because two other sessions run
test suites on this box and one earlier pilot had to be re-measured for it.

---

## 0. The claim, and what decides it

#974 (from gate-971 §5): with the whole 4.14 GB Mistral-7B NF4 store in the
page cache, the async source stepped 1.20x slower than the synchronous
`DiskSource` when it ran first and 1.03x when it ran last — a bracket, because
the two arms never occupied the same run-order slot. The issue's acceptance
criterion is a same-session, position-matched pair in BOTH orders landing at
1.0x within the instrument's noise, or a documented decision.

The number that actually matters to the project is a different one and the
issue does not name it: the cold 70B step (gate-971 §10) is 84.7% read, at
1.46-2.32 GB/s from a drive rated 3.5+ GB/s, and the user's target for it is
~30 s -> ~20 s. Anything done to the reader for #974 has to be measured there
too, and §4 is where that changed the plan.

## 1. Warm, position-matched: no regression

Instrument: `issue974_warm_stages.py` — `stream_probe.build` plus host timers
around each stage (`consumer_wait` inside `AsyncDiskSource.get`, `reader_read`
inside the layer read, `reader_drain` inside the pool event's `synchronize`
the reader waits on before refilling a slot, `consumer_read` inside
`DiskSource.get` for the control, `consumer_load_async` inside the pool's
`load_async`). Mistral-7B NF4, batch 1 x seq 512, 8 timed steps after 3
warm-up, one arm per process. Baseline for the quiet block: free physical
19.9-20.4 GB, commit 22.1-22.3 GB, 4 Python processes, the peer sessions idle
(13:51-13:56; measured by the previous session of this work, JSON
`i974_quiet_*.json`).

| order / slot | arm | step, mean (min-max) | tok/s | where the time is |
|---|---|---|---|---|
| A / 1 | async | **4.817 s** (4.667-4.962), FLAT over 8 steps | 106 | reader_read 4.470 s/step; consumer_wait 2.995 s; reader_drain 0.008 s |
| A / 2 | control (sync mmap) | 2.159 s (2.035-2.229) | 237 | consumer_read 0.105 s; consumer_load_async 1.459 s (the pageable driver memcpy) |
| B / 1 | control | 2.038 s (1.986-2.171) | 251 | consumer_read 0.090 s; load_async 1.390 s |
| B / 2 | async | **1.970 s** (1.939-1.990) | 260 | reader_read 1.827 s (4.3 GB/s); consumer_wait 1.229 s; reader_drain 0.001 s |
| C / 1 | RAM tier | 1.098 s (1.082-1.148) | 466 | copy 0.288 s |

**Finding 1 — position-matched, the async source is 0.91x the control in
slot 2 (1.970 vs 2.159 s) and 0.97x across the two warm-slot readings (1.970
vs 2.038 s).** The 1.03-1.20x was the run-order effect gate-971 §3 had already
named for the cold fixture, showing up warm.

**Finding 2 — the drain-serialisation hypothesis (#974's candidate 2) is
refuted**: `reader_drain` is 1-8 ms per step. Warm, the async arm is bound by
its OWN read — `reader_read` is 93% of its step, ~4.3 GB/s of per-tensor
`readinto` from the page cache — and the control by the pageable-copy memcpy
on the compute thread (`get_tensor` is a free mmap view, 0.1 s).

## 2. The 2.4x-slow block: the handoff's hypothesis, tested first

Order A slot 1 ran first after the #901 14B probe had read 9.9 GB and pinned
10 GiB, and stayed at 1.6 GB/s for its whole block — three warm-up steps plus
eight timed, 86 GB of reads, per-step `reader_read` 4.43 4.54 4.33 4.50 4.60
4.48 4.55 4.34 s with no warming at all — while the control's mmap reads in
the same slot (order B) ran warm and the async arm was fast again once the
control had run. The handoff's hypothesis: the async source's buffered
`readinto` does not repopulate the page cache, mmap does.

**Refuted at the file-API level in 30 seconds.** `issue974_cache_populate.py`
runs the two primitives over the 32 Mistral decoder shards (3.601 GB) in a
chosen order, with an eviction where the sequence says (touch 17 GB of
pageable RAM, which drove available physical to 2.3-2.6 GB, then free it).
Baseline: available 18.70 GB, commit 26.00 of 50.84 GB (the pagefile grew
under the pressure — the 47.35 GB limit in the other blocks is the box at
rest). JSON `cache_populate_run1.json`.

| step | primitive | GB/s | reading |
|---|---|---|---|
| 0 | readinto (plain `open`) | 3.85 | warm from the previous pilot |
| evict | | | |
| 2 | readinto | 2.60 | cold, or mostly |
| 3 | readinto | **3.96** | warm again — readinto populated it |
| 4 | mmap (`safe_open().get_tensor` + copy) | 2.56 | |
| 5 | readinto | 3.73 | |
| evict | | | |
| 7-8 | readinto through `O_RANDOM` | 2.41 -> 3.94 | populates |
| evict | | | |
| 10-11 | readinto through `O_SEQUENTIAL` | 2.85 -> 3.67 | populates |
| evict | | | |
| 13-14 | readinto, `buffering=0` | 2.45 -> 3.72 | populates |
| evict | | | |
| 16-17 | mmap, twice | 1.68 -> 2.28 | mmap cold is the slowest primitive here |

**Finding 3 — every buffered `readinto` flavour repopulates the page cache;
the second pass is warm.** The hypothesis is wrong. (Pressure alone leaves
the most recently used ~2.5 GB of the standby list resident — i.e. exactly the
store under test — so the eviction used from here on adds a second stage:
read 8 GB of OTHER files through buffered readinto, which Finding 3 says
displaces the survivors. `issue974_evict.py`.)

**Then through the shipped path, and the slow block did not come back.**
Two-stage eviction, then async / async / control / async, same harness and
flags as §1 (14:12:26; baseline free physical 18.9 GB, commit 21.63 of 47.35
GB, 2 Python processes; after the block 18.88 / 21.7). JSON
`i974_evictB_slot{1..4}_*.json`:

| slot | arm | step, mean (min-max) | tok/s | reader_read / consumer_read |
|---|---|---|---|---|
| 1 (first after eviction) | async | 1.887 s (1.851-1.960) | 271.4 | 1.745 s — warm within the 3 warm-up steps |
| 2 | async | 1.942 s (1.881-1.981) | 263.7 | 1.801 s |
| 3 | control | 1.983 s (1.968-1.995) | 258.2 | 0.084 s (load_async 1.375 s) |
| 4 | async | 1.915 s (1.857-1.980) | 267.4 | 1.773 s |

**Finding 4 — after a deliberate eviction the async arm warms up inside its
warm-up steps and runs 0.95-0.97x the control in every slot.** The order-A
block of §1 is not reproduced by the only mechanism proposed for it; what
made it slow for 55 s is not established, and this record does not guess.
The finding that survives is the one that matters for the design: cold-start
of a store that fits the cache is NOT a 2.4x tax on the async source.

## 3. What the warm micro-benchmark said the fix should be

`.claude/probes/2026-09-15-stream-901-974/bench_read_paths.py` (the previous
session, quiet box, 16 Mistral layers, pinned staging, best of 3 passes, store
in the page cache): per-tensor `readinto` 3.88 GB/s at 1 thread, 4.57 at 2
and 4, 5.14 at 8; mmap+copy 2.1-2.5; ONE `readinto` per layer of the whole
contiguous data section into an arena 3.66 at 1 thread, 6.31 at 2, 8.41 at 4,
**9.32 at 8**. Every shard cache here stores a layer's tensors dtype-descending
with zero gap bytes and every offset aligned to its dtype (checked on the 70B
store: 30 tensors, first start 2976, span 441 430 044 = the sum of the
tensors, gaps {0}).

That is the plan the handoff carried: one read per layer, K ranges in
parallel, ~9 GB/s. It is a page-cache -> pinned memcpy number, and §4 is why
it is not the number that decides anything.

## 4. Cold: the buffered ceiling, then the drive

The user's target is the cold 70B step. Every configuration below reads its
OWN six fresh layers of the 36.39 GB 70B-shaped NF4 store (34 GB on disk
against 31.7 GB of RAM, so it is never fully cached), after the two-stage
eviction, so no configuration benefits from a previous one's pages; two runs
with the configuration order reversed, so a drive or position effect cannot
masquerade as a primitive's. 441.4 MB per layer, 30 tensors.

**Buffered** (`issue974_cold_ranges.py`; 14:17-14:18; baseline free physical
18.78 GB, commit 21.75 of 47.35 GB, 2 Python processes; JSON
`cold_ranges_run{A,B}.json`):

| primitive, threads | run A (layers) | run B, reversed order (layers) |
|---|---|---|
| per-tensor readinto, 1 | 1.81 GB/s (0-5) | 1.47 (42-47) |
| per-tensor readinto, 2 | 2.15 (6-11) | 1.60 (36-41) |
| per-tensor readinto, 4 | 2.23 (12-17) | 1.51 (30-35) |
| per-tensor readinto, 8 | 1.49 (18-23) | 1.52 (24-29) |
| one readinto per range, 1 | 1.25 (24-29) | 1.28 (18-23) |
| one readinto per range, 2 | 1.27 (30-35) | 1.99 (12-17) |
| one readinto per range, 4 | 1.44 (36-41) | 2.32 (6-11) |
| one readinto per range, 8 | 1.57 (42-47) | **2.38** (0-5) |

**Finding 5 — cold, every buffered primitive lands between 1.25 and 2.4 GB/s,
and the second half of each run is slower than the first whichever primitive
is there.** The request shape is not the ceiling; the cache manager is. The
async source's 1.46-2.32 GB/s in gate-971 was already this ceiling, so the
§3 plan — parallel BUFFERED ranges — would have bought the cold step ~1.1-1.3x
at best. It cannot reach ~20 s from ~30 s.

**Unbuffered** (`issue974_unbuffered.py`: `CreateFileW(FILE_FLAG_NO_BUFFERING
| FILE_FLAG_SEQUENTIAL_SCAN)`, one handle per thread, synchronous `ReadFile`
into a 4 KiB-aligned pinned arena over the aligned superset of the data
section; 14:22-14:24; baseline free physical 19.34 GB, commit 21.6 of 47.35
GB, 2 Python processes; JSON `unbuffered_run{A,B}.json`; the last layer of
each run byte-checked against a buffered read: OK):

| primitive, threads | run A (layers) | run B, reversed order (layers) |
|---|---|---|
| unbuffered, one request per range, 1 | 3.89 GB/s (0-5) | 4.17 (66-71) |
| unbuffered, one request per range, 2 | **5.30** (6-11) | 4.02 (60-65) |
| unbuffered, one request per range, 4 | 5.12 (12-17) | 4.18 (54-59) |
| unbuffered, one request per range, 8 | 5.10 (18-23) | 4.09 (48-53) |
| unbuffered, 4 MiB requests, 1 | 3.68 (24-29) | 3.53 (42-47) |
| unbuffered, 4 MiB requests, 2 | 3.89 (30-35) | 5.61 (36-41) |
| unbuffered, 4 MiB requests, 4 | 3.85 (36-41) | **5.65** (30-35) |
| unbuffered, 4 MiB requests, 8 | 3.86 (42-47) | 3.92 (24-29) |
| buffered, one readinto per range, 1 | 1.76 (48-53) | 1.22 (18-23) |
| buffered, one readinto per range, 2 | 2.05 (54-59) | 2.15 (12-17) |
| buffered, one readinto per range, 4 | 2.44 (60-65) | 2.22 (6-11) |
| buffered, one readinto per range, 8 | 2.88 (66-71) | 2.53 (0-5) |

**Finding 6 — unbuffered I/O reads the cold store at 3.5-5.65 GB/s, best at
2-4 parallel ranges, against 1.2-2.9 buffered in the same positions; a single
unbuffered thread already beats the best buffered configuration.** The
drive's published figure ("3.5+ GB/s") is exceeded, so the buffered number
was never the drive. Two-thirds of the 8-thread unbuffered readings are
slower than 2-4 threads — the queue is deep enough at four.

The run-B second-half slowdown of the buffered table (Finding 5) reappears
only weakly here (unbuffered one-request at positions 9-12 reads 4.0-4.2
against 5.1-5.3 at positions 2-4 in run A), so it is a drive/position effect
of the order of 20-30%, not a property of any primitive; the two orders are
published for exactly that reason and neither is corrected.

Two alignment facts the reader had to be built on: an unaligned request
LENGTH is refused with EINVAL, and a request whose aligned end runs past the
end of the file returns short — the last sector of a data section is the end
of its shard, so every layer's last range does this. (A first draft of the
benchmark asked for the remainder as a second, unaligned request and was
refused with error 87; the fixed loop stops at the bytes the file holds and
never issues an unaligned length.) A handle opened this way wraps as a CRT
descriptor (`msvcrt.open_osfhandle`) and reads through plain
`io.FileIO.readinto` at 4.18 GB/s single-threaded — so the reader stays
portable Python: `O_DIRECT` on Linux, `F_NOCACHE` on macOS, buffered `open`
where none of them is available.

## 5. The change

`src/soup_cli/utils/safetensors_reader.py`: `SECTOR_BYTES`, `aligned_span`,
`plan_ranges` (contiguous, sector-aligned, never empty; fewer ranges than
asked when the span is shorter), `read_range_into` (the front of a view, stops
at the bytes the file holds, refuses a short file by name), `open_direct` (the
three platform modes; `DirectIOUnavailableError` where there is none).

`src/soup_cli/utils/async_disk_source.py`: staging is one contiguous REGION
per slot — the aligned superset of the layer's data section — and every
tensor a view at `entry.start - aligned_start`, per LAYER (a sibling with a
longer header or a tensor the spec does not want keeps the same tensors at
different offsets, and the tests build both). Pinned, the regions are packed
by `plan_pinned_arenas(..., align=4096)` — on the 70B store's four slots
(2 x 441.4 MB decoder + 536.9 MB embed + 536.9 MB head) that is ONE 2 GiB
arena for 1.96 GB, where per-tensor pinning paid the 1.7-1.9x #901 measured
and the pre-flight's `staging_bytes_for` budget was under-counting by. The
reader thread dispatches each layer as `read_ranges` (default 4) sector-aligned
ranges to that many daemon worker threads, each through its own handle, and
waits; `_plan_queue` / `_claim_slot` / `get` / `_hold` / `release` are
untouched. Direct I/O is decided once per source (a probe open of layer 0's
shard) and logged; `direct_io`, `pinned_bytes` and `staging_bytes` are
reported, and the ready line prints the page-locked figure on the disk tier
as it does on the RAM tier.

Tests: `tests/test_issue974_direct_range_reads.py` (33; the byte-identity gate
against `DiskSource` through 1/3/4 ranges, without direct I/O, with a longer
header on one layer, with a foreign tensor in one shard, with pinned staging;
the arena and alignment invariants; the parallelism and the worker threads'
daemon-ness; the wedge seam; the disk tier's own failed-page-lock round trip
on real hardware — the RAM tier had one since #989, this tier did not). Four
tests of `test_issue971_async_disk_source.py` moved from the `read_into` seam
to `read_range_into`, and its `_settle` helper now reads the reader's two
fields under the reader's lock: `_run` pops the queue and sets `_in_flight`
as two statements inside one critical section, and an unlocked observer
landing between them measured the depth before the read it had just missed
("sustains 2 of 3", 2 failures in 8 runs; 0 in 12 after).

**Gates on the new code, 14:50-15:01.** The two 974/971 files: 96 passed
with CUDA. The streaming regression set (`test_v07200-04`, `issue623`,
`issue349`, `issue385`, `issue901_*`, `cli_startup_is_light`, `971_*`,
`974`): 819 passed / 75 skipped on CPU. The race battery
(`.claude/probes/2026-09-13-async-nvme/`, the #971 review's probes driving
the REAL `LayerBufferPool` + `StreamPrefetcher` with pinned staging and the
GPU deliberately behind): `race.py` 12/12 clean and `race_pageable.py` 12/12
at `read_ahead` 1/2/4/8 x 3, `fwdbwd.py` (forward then backward, 16 visits)
12/12, `hetero.py` (decoder + embed groups) 12/12, `backward_prefetch.py` 7/7
targets staged ahead x 3, `control.py disk` clean, `allive.py` PASS,
`slow.py` a 40 s read served (the 300 s wedge limit did not fire). **A first
battery run was discarded**: six of the probes hardcode the MAIN tree's `src`
in `sys.path` and silently measured the old code (found because the one probe
that patches the moved seam raised `no attribute 'read_range_into'`); every
line of the kept run carries the `soup_cli.__file__` it imported. Two probes
predate the queue design and error on a state that no longer exists
(`cv.py` reads `_wanted`; `control.py`'s RAM arm calls `RamSource.close`,
which has never existed) — not regressions, and the same properties are unit
tests in the 971 suite.

## 6. After the change: warm, position-matched — with a control that drifted

Same instrument, fixture and flags as §1, 15:01-15:05, right after the race
battery; baseline free physical 18.74 GB, commit 21.67 of 47.35 GB, 2 Python
processes before, 18.64 / 21.61 after. JSON `i974_fix_order{A,B,C}_slot*_*.json`.

| order / slot | arm | step, mean (min-max) | tok/s | where the time is |
|---|---|---|---|---|
| A / 1 | async, NEW | **1.803 s** (1.770-1.860) | 284.0 | reader_read 1.602 s/60 reads = 4.5 GB/s off the drive; consumer_wait 0.661 s; drain 0.004 s |
| A / 2 | control | 2.806 s (2.788-2.821) | 182.5 | consumer_read 0.146 s; load_async 1.788 s (copy 1.666 s) |
| B / 1 | control | 2.829 s (2.793-2.858) | 181.0 | load_async 1.801 s (copy 1.686 s) |
| B / 2 | async, NEW | **1.628 s** (1.594-1.675) | 314.6 | reader_read 1.416 s/60 = 5.1 GB/s; consumer_wait 0.524 s |
| C / 1 | RAM tier | 0.877 s (0.869-0.888) | 583.8 | copy 0.286 s |

**Finding 7 — position-matched, the new async arm is 0.64x (slot 1) and
0.58x (slot 2) of the synchronous control in this session**, where the same
pairs were 0.95-0.97x (§2) and 0.91-0.97x (§1) before the change.

**Finding 8 — and that ratio is NOT claimed as the change's gain, because the
control moved.** Both control blocks are 1.4x slower than every control
reading an hour earlier (2.806 / 2.829 s against 1.983-2.159 s), flat over
their 8 steps in both slots, and the time is in one place: the pageable
host-to-device copy out of the mmap'd pages (1.67-1.69 s against 0.94-1.06 s);
`consumer_read` itself is unchanged (0.146 vs 0.084-0.105 s). The RAM tier
meanwhile is FASTER than its earlier reading (0.877 against 1.098 s), so the
box's state moved in two directions between §1/§2 and here, and what moved
the pageable copy is not established. Two candidates are named, neither
tested here: the standby list is now full of 70B-store pages from §4's cold
benchmarks (the Mistral store was evicted and is re-faulted by the control's
mmap every step), and the direct-I/O arm no longer warms the page cache for
whatever runs after it — which is a real property of the change, and the
reason the order-A control (slot 2) could not have been warmed by slot 1.
The only cross-session comparison that survives is async against async:
1.628-1.803 s now against 1.887-1.970 s before the change (0.83-0.95x), and
§8 re-measures old against new inside one session, which is the comparison
that decides.

**Finding 9 — warm, the new arm is still bound by its own read**:
`reader_read` is 1.42-1.60 s of a 1.63-1.80 s step, 7.2 GB per step at
4.5-5.1 GB/s from the drive — the §4 unbuffered ceiling, above the 3.9-4.3
GB/s the old per-tensor `readinto` drew from the page cache, but not the
memcpy rate a cached store could give. The RAM tier's 0.88 s is the compute
floor; a store that fits RAM belongs on the RAM tier, and `stream_source:
auto` puts it there.

## 7. The review round, and what it changed (15:06-16:20)

Four read-only reviews of the first commit (`9432c95a`), CPU-only, reports in
`.claude/reviews/2026-09-15-issue974/`. They ran while §8's first cold series
was on the GPU, which is recorded rather than hidden: that series was lost to
its own log filter (below), not to them, and the series that stands ran after
they had finished.

- **HIGH, found independently by three of the four**: `read_range_into`
  raised "short read" on ANY `readinto` that returned fewer bytes than asked,
  not only on a zero return. `io.FileIO` is one syscall, and POSIX lets it
  return short with more still to come — Linux documents exactly that for
  `O_DIRECT` — so a legitimate partial transfer of a ~110 MB range would have
  aborted a run with an error naming the shard. Every fixture in the suite is
  small enough that the next call always returned zero, which is why 33 tests
  passed over it; the TDD reviewer's mutation run found the same gap from the
  other side (removing the guard killed nothing). The loop now asks again
  until the count is met; only a zero return ends the data. The request stays
  `buffer[done:]` rather than `buffer[done:expected]`, because the latter is
  unaligned at every layer's last range and a direct handle refuses it.
- **HIGH (security)**: staging regions are sized to the SPAN between the first
  and last wanted tensor, so a shard with a large foreign tensor between them
  was staged, page-locked, with no cap — the reviewer's hand-built shard put
  two 16-byte tensors 8 MiB apart, a 262,272x allocation. Refused at
  construction by name when the span exceeds twice the wanted bytes plus two
  sectors; the small-foreign-tensor case of §5's tests still passes. Soup's
  own sharder writes contiguously, so the reachable route was a corrupted or
  hand-built shard, or the public `shard_paths` parameter.
- **MEDIUM / LOW**: the page-locked suffix lived twice in the ready line (one
  helper now); `_RangeReaders.close` signalled its workers but never joined
  them (it does, with a bounded deadline — and the TDD reviewer's "make the
  workers non-daemon" mutation hung the interpreter at exit, which is the
  case the daemon flag exists for); an empty spec, a wanted set holding no
  bytes and a negative count are refused by name; the descriptor is closed if
  the `io.FileIO` wrapper itself cannot be built; a dead `_LayerPlan.expected`
  field is gone; the reader is typed without importing torch; a zero-element
  tensor rides through the range reader in a test; the evict harness no
  longer bakes this box's path into a default.
- **Kept, with the reason**: `read_into` (the per-tensor read) is no longer
  called by production but stays — three published harnesses and the
  reader's own tests use it. The dtype-alignment refusal's mutation is killed
  through torch's own `RuntimeError` rather than the named `ValueError`; the
  guard's job is the message, which the test asserts.

Second commit `1601e025`; the two 974/971 files plus the #901 arena tests:
112 passed / 20 skipped on CPU, ruff clean, both modules still torch-free at
import.

**The first cold series was discarded.** Its blocks died out of order: the
order-A old-async arm after its plain point, the order-A control after 44
minutes with no point written (its commit charge had reached 52 GB of a 69 GB
limit — the mmap control's #926 property), and all three order-B arms in
~40 s each. The first explanation written here was a shell defect — the
driver piped each block through `grep | head -12`, and `head` closing the
pipe would kill the producer at its next print. **Corrected 16:30-16:45,
from the other party's own notes** (`.claude/probes/2026-09-15-stream-901-974/
soup-f5/`, gitignored): a second Claude session, soup-f5, had been started
~14:45 with the same brief and the same worktree, took the running probes
for orphans of the previous session and killed them — 15:24:07-15:24:27 the
old-async arm (its events point lost ~5.5 min in, on a misread wall clock),
then at ~16:09:50 and ~16:10:40 a `*stream_probe*` pattern sweep that took
the order-B control, the order-B old- and new-async arms and this driver's
own bash processes. The order-B arms did not fail at build; they were killed.
The order-A control was contaminated from its first second: soup-f5's own
cold A/B ran 15:24:31-15:28:22 on the same card and the same store, and its
measurement of that collision is the number worth keeping — the same new-arm
configuration in the same slot stepped **33.904 s with this series' control on
the card and 17.502 s alone** (`contamination-finding.md`): **two cold 70B
probes on this box cost each other ~1.94x**, with the commit charge at
62.67 GB against a pagefile Windows had grown to 77 GB. Why that control
exited at 16:08 with nothing written was not captured (this driver's filter
ate stderr); it is discarded either way. The `head` hazard is real and the
take-2 driver logs every block to its own file, but it did not fire here.
Every measurement in this record is stamped with the box baseline for exactly
this reason; from 16:30 soup-f5 is reading only, and its 17.502 s is quoted
above as a control on the instrument, not as a measurement of the reader.

Both sessions' review agents wrote to the same four filenames in
`.claude/reviews/2026-09-15-issue974/`: `python-review.md` and
`tdd-review.md` are this session's, `code-review.md` and
`security-review.md` on disk are soup-f5's agents' (this session's code
review — whose HIGH on `read_range_into` is the one 1601e025 acted on — was
overwritten; its summary is kept as `code-review-soup-3d-summary.md`; the
security HIGH above was found by soup-f5's agent and re-verified from
scratch). soup-f5's TDD agent also found two mutations this suite did not
catch, both tests now: the absolute file offset in place of
`entry.start - aligned_start` survived because every fixture's data section —
and the real 70B store's, at byte 2976 — sits inside the first 4 KiB sector,
so the aligned start was 0 everywhere (the long-header fixture now carries
5000 bytes of metadata and asserts its aligned start is past the first
sector; the mutation verified killed); and a `close()` that keeps the regions
and arenas referenced was invisible (asserted now).

The one block that completed, new-async at order A slot 1
(`i974_fix_cold_orderA_slot1_newasync.json`, 15:05, the reviewers just
dispatched and not yet running anything): **16.711 s** (16.007-17.240) plain
/ 16.843 s instrumented, 30.6 tok/s, 70.38 GB moved per step, 12.451 s of
copy brackets (74%, copy stream 5.65 GB/s), peak 4.378 GB. The old-async
block got as far as its plain point: **52.966 s** (49.55-55.69), 9.67 tok/s,
at the early position — gate-971's 48.09 s for the same position. Both are
kept as readings; the series that stands is §8.

## 8. After the change: cold, both orders — series 2, RUNNING when this section was committed

Take-2 driver (`cold_series2.sh`, kept under
`.claude/probes/2026-09-15-stream-901-974/session-soup-3d/`): every block's
stdout+stderr to its own file under `.claude/probes/.../cold2/`, nothing
filtered; gate-971 §2's protocol (batch 1 x seq 512, 6 timed after 2
warm-up, `read_ahead 2`, `--step` = the plain point then the instrumented
one); three arms — the NEW reader, the OLD reader (origin/main a08ae74f's
`src`, `git archive`d into the scratchpad so no other checkout can move
under the run; `meta.soup_cli_file` in every JSON is the proof of which tree
ran), the shipped synchronous `DiskSource`; order A new/old/control, order B
control/old/new; no eviction between blocks. The box otherwise quiet: soup-f5
reading only, no reviewers running. Baseline at 16:20:10: free physical
17.51 GB, commit 23.82 of 47.35 GB, 2 Python processes (both VS Code's LSP
servers). JSON `i974_cold2_<order>_slot<n>_<arm>.json`.

| order / slot | arm | plain step, mean (min-max) | tok/s | instrumented step | copy brackets | SM clock start->end (plain / events) |
|---|---|---|---|---|---|---|
| A / 1 | NEW async | **16.175 s** | 31.7 | 16.413 s | 10.101 s = 62%, copy stream 6.97 GB/s | 180->1552 / 1552->1725 MHz |
| A / 2 | OLD async (origin/main) | **45.566 s** | 11.2 | 33.728 s | 28.927 s = 86%, 2.43 GB/s | 180->1357 / 1357->817 MHz |
| A / 3 | control (sync DiskSource) | running from 16:36:44 | | | | |
| B / 1-3 | control, OLD, NEW | queued behind it | | | | |

**Finding 10 (provisional — order B decides the position-matched pairs).**
At the early position the new reader's cold 70B step is **16.2 s** against
the old reader's **45.6 s** two slots later and gate-971's 48.09 s for the
same slot (2.8x); its read share fell from 86% to 62% at a copy-stream rate
of 6.97 against 2.43 GB/s. The old arm's instrumented point (33.7 s) is
FASTER than its plain one (45.6 s) — the page cache warming across the
block's sixteen steps, gate-971 §3's effect — where the new arm's two points
agree within 1.5%, because it does not use the cache. Peak VRAM 4.378 GB in
every block, unchanged from gate-971. The user's target for this step was
~30 s -> ~20 s.

## 9. What was NOT measured

- **A real 70B, or any quality claim.** The cold fixture is a synthetic
  Llama-70B *shape* with random weights; the byte-identity gates against
  `DiskSource` (CPU, 1/3/4 ranges, with and without direct I/O, pinned and
  pageable) are the correctness evidence, and the four bit-exact ids of
  `test_issue385_stream_dtype` were run before the change, not after — the
  disk tier is not on that test's path.
- **Linux `O_DIRECT` and macOS `F_NOCACHE`.** Read hard by four reviewers,
  executed by none: this box is Windows. CI's Linux runners exercise the
  buffered FALLBACK (tmpfs refuses `O_DIRECT`), which is the right thing to
  exercise there and no evidence about the direct path.
- **`read_ranges` other than 4 through the real streaming step.** 1/2/4/8
  were measured on the read primitive (§4); the step was run at the default
  only.
- **Pageable staging (`stream_pin: false`) through direct I/O**, cold or warm.
  The sector slack is tested; its speed is not.
- **The warm case's cause for the control's 1.4x drift** (§6, Finding 8) —
  two candidates named, neither tested. §8's re-check runs old and new in one
  session after an eviction, which is the comparison that matters; it does
  not explain the drift.
- **Whether the first cold series' dead blocks would have completed** had a
  second session not killed them (§7). The take-2 series is the record.
- **Read share of the step at other shapes**, and any sequence sweep; batch 1
  x seq 512 only, as gate-971.
- **The `--skip-events` flag** the other session added to `stream_probe.py`
  while this record was being written (kept, credited in the commit): not
  used by any block here — every block runs both points, as gate-971 did.
- **#975** (the step-head embedding fetch, 0.07-0.12% of a cold step): the
  read plan did not change, so it was not touched and not re-measured.
