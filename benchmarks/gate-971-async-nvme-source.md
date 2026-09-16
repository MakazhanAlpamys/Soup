<!--
Working measurement record, published verbatim.

Like the records beside it, this is the log kept while the work happened, not a
report assembled afterwards: the reading that was wrong at 13:15 and corrected
at 13:35, the confound that turned a depth result into an ordering result, the
outlier that did not reproduce, and a negative result on the warm fixture all
stay in, in the order they occurred.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB by nvidia-smi /
8.518 GB by torch, driver 616.92, PCIe 5.0 x8 negotiated), Intel i9-14900HX,
31.7 GB DDR5-5600, Windows 11 Pro 26100. Two Samsung MZAL81T0HFLB (PM9B1-class)
NVMe, 954 GB each: disk 0 is C:, disk 1 is D:. Every TIMED read below comes
from the NF4 shard cache on C: (disk 0); D: holds only the bf16 source
checkpoint, which is read during sharding and was a cache hit throughout.
Stack: Python 3.12.10 · torch 2.14.0+cu130 · transformers 5.17.0 · peft 0.20.0
· bitsandbytes 0.50.2 · safetensors 0.8.0.
Soup: branch feat/async-nvme-source at 9e5ce63a (v0.75.0 + Tasks 1-5 of #971 +
the two harness commits described in section 1).
Harness: benchmarks/harness/stream_probe.py, plus benchmarks/harness/
layer0_wait.py for section 8. Raw per-point JSON for every block is under
benchmarks/results/probe-rtx5070/.
-->

# Gate record — #971: the async NVMe source, measured cold and warm

**Status: MEASURED, 2026-09-14, with one negative result and one confound that
changed the headline.** The cold disk tier got **2.1x–3.1x faster** depending on
which position-matched pair you take. **Warm, with the whole store in the page
cache, the async source is SLOWER than the one it replaces — 1.20x in the first
run order and 1.03x with that order reversed** — measured against a same-day
control, not against a published number from another session. What looked at
first like a read-ahead *depth* effect is consistent with the page cache warming
across blocks, and is published as the ordering result it is.

**Revised 2026-09-14 15:30–16:01, after a whole-branch review.** Six things
changed and every one of them is a claim this record made that its own JSON did
not support: a GPU-clock cell printed as `—`, a missing spread on the row that
anchors the best ratio, a "roughly half the step" that contradicted two other
sections, an undisclosed run order under the warm regression, a §8 JSON produced
by a scratchpad copy of the harness rather than the committed one, and no table
naming its source file. The corrections are dated and inline; nothing was
deleted. §8 was re-measured and the warm trio re-run in reverse order.

Unit convention: **decimal GB**, matching the other records.

---

## 0. The question, and what it is measured against

Layer streaming's disk tier read each layer synchronously, on the compute
thread, through a memory map. `probe-rtx5070-what-bounds-streaming.md` measured
what that costs: §17 put a cold 70B-shaped NF4 step at **124.5 s** (4.1 tok/s,
0.57 GB/s) on an NVMe that reads 3.5+ GB/s, §17a found it unstable by an order
of magnitude between blocks, and §18a put the **read-free floor at ~14.25 s** at
seq 256. That record's verdict named the fix: "an asynchronous source with
sequential unbuffered reads and pinned staging, off the compute thread".

Tasks 1–5 of #971 built the first half of that — a background reader with its
own header parsing (no mmap), pinned host staging, and a `training.stream_read_ahead`
depth. This record asks whether the cold step actually got faster, and by how
much. The honest ceiling for this project alone is the ~14 s floor, not the RAM
tier's numbers.

**Two comparisons are available and they are not the same thing.**

* The **published** before: §17/§17a/§8, measured 2026-09-12, in a different
  session, on a card whose boost clock varies ~13% between sessions.
* A **same-day control**: the shipped synchronous `DiskSource`, re-measured in
  this session on this fixture. Section 1(d) says how that was made without
  touching `src/`.

Both are reported. Where they disagree, the same-day control wins, and §2 shows
exactly why that ruling mattered.

---

## 1. The harness had to change first

`stream_probe.py` is the instrument, and as it stood it could not have measured
this source honestly. Four changes, all in the harness, none in `src/`.

**(a) `--read-ahead N`,** default 2, passed to `build_streamed_model` beside
`buffers`. The source allocates `min(N, members)` staging slots per distinct
layer shape, so the run's own `store` line is independent evidence of the depth
actually allocated rather than of the flag that was typed. On the 70B fixture
that arithmetic checks out exactly: 2 × 441.43 MB decoder + 2 × 536.87 MB vocab
= **1.9566 GB**, which is the reported store at depth 2.

**(b) `pin=` no longer forces pageable staging on the disk tier.** The shipped
line was `pin=(args.tier == "ram" and not args.no_pin)`, correct while the disk
tier allocated a fresh tensor per call and had nothing to page-lock. The async
source reads ahead into reusable host staging, so that expression would have
measured the **pageable fallback** rather than the shipped path. `--no-pin` now
selects the pageable arm on either tier.

**(c) The `Instruments` replicas now call `_release_source`,** which the shipped
`LayerBufferPool.load_async` and `LargeLayerBufferPool.load_async` do and the
replicas did not. Under pinned staging the source refuses to lend a second layer
while the first is still on loan, so the replica as it stood raised at the second
layer — that refusal working as designed, not a probe bug. The four release
lines were added to both drift-guard needle sets at the same time. Checked both
ways: the needles are present in the shipped bodies, and a deliberately drifted
needle (`self.events[slot + 1]`) makes the guard raise.

**(d) `--control-sync-source`,** which is how the same-day control was made
without changing `src/`. The runtime no longer constructs `DiskSource` for
`tier='disk'`, so the switch replaces the one name `_build_source` imports
lazily with a subclass of the shipped `DiskSource` that accepts and ignores the
two keyword arguments the async source added. The buffer pool, the prefetcher
and the layer wrapper stay the shipped ones — exactly as before Task 5 — and
`_release_source` is duck-typed, so it is a no-op against a source with no
`release`. The control therefore isolates the **read path**, not a different
scheduler. Every run now prints and records the source class it actually built
(`source_class`, `control_sync_source`, `read_ahead` in the JSON) and refuses
when that disagrees with the flag, so a control block cannot be mistaken for a
measurement; `--tier ram` is refused up front.

### Verification before any cold block

Warm 7B, `--steps 2 --warmup 1`. These three are FUNCTIONAL verification, not
throughput evidence: another session's full pytest suite was running throughout
(baseline commit charge 28.46 GB of 39.44 GB, free physical 15.04 GB).

| arm | the run's own `store` line | step |
|---|---|---|
| disk, pinned (default) | `0.762 GB pinned on tier disk (disk 4.138 GB)` | 3.484 s |
| disk, `--no-pin` | `0.762 GB pageable on tier disk (disk 4.138 GB)` | 3.627 s |
| ram (control) | `4.138 GB pinned on tier ram (disk 0.000 GB)` | 0.819 s |

The first row could not have been produced before change (b): the disk tier
reports **pinned** staging.

**These three figures and every `store` line quoted in this record were read off
the run logs; no JSON was retained for them** (`--out` was written and then
overwritten by the next verification arm). They are labelled functional
verification and are never cited as throughput, so this is a note rather than a
defect — but every other table here names a committed file and these cannot.
(Added 2026-09-14 15:30.)

---

## 2. The cold headline

Synthetic Llama-70B **shape** (random weights — timing only, never a quality
claim), NF4, 36.39 GB on disk against 20–25 GB of free physical RAM, so no page
cache can hold it. Batch 1 × seq 512, 6 timed steps after 2 warm-up,
**uninstrumented** (§4 says why that qualifier is load-bearing). Sharding was a
cache hit in every block (`shard 0.0 s`), so every block read the same store.

| block | order | source | step | tok/s | source rate | GPU clock | JSON |
|---|---|---|---|---|---|---|---|
| control, sync `DiskSource` | 2nd | pageable, no reader | **100.35 s** (89.6–105.7) | 5.10 | 0.701 GB/s | 180 → 180 MHz | `control_sync_cold_synth70b_nf4.json` |
| async, `read_ahead 2` | 1st | 1.957 GB pinned staging | **48.09 s** (43.3–52.5) | 10.65 | 1.464 GB/s | 1792 → 2355 MHz | `async_cold_synth70b_nf4.json` |
| control, sync `DiskSource` | 6th | pageable, no reader | **92.25 s** (85.4–101.8) | 5.55 | 0.763 GB/s | 1417 → 210 MHz | `control_sync_cold_synth70b_nf4_last.json` |
| async, `read_ahead 2` (repeat) | 5th | 1.957 GB pinned staging | **30.28 s** (29.1–34.0) | 16.91 | 2.324 GB/s | 180 → 705 MHz | `async_cold_synth70b_nf4_ra2_repeat.json` |

**Correction, 2026-09-14 15:30.** The third row printed `—` for its GPU clock
and carried no spread, and it is the row the 3.05x rests on. Both were in its
JSON the whole time: `1417 → 210 MHz`, and 85.42–101.75 s. They are filled in
above. The spread is wide — wider than the async repeat's 29.1–34.0 — which
*strengthens* the finding rather than weakening it, and a reader could not see
that. Every table in this record now names the JSON it came from, for the same
reason: nine files, two near-identically named, and the mapping was previously
reconstructible only from `started` timestamps.

Bytes moved is **70.38 GB per step** in every row (157 decoder loads + 2 large),
and peak VRAM is **4.378 GB allocated / 4.80 reserved** in every row — streaming
still bounds the weights exactly as before; only the read path changed.

**Finding 2 — the cold step is 2.09x faster at the early position and 3.05x
faster at the late one.** Both pairs are same-session, same fixture, same shape,
adjacent in run order. Against §17's published 124.5 s — the other first-in-
session block — it is 2.59x. The spread between those ratios is not noise, it is
§3.

**Finding 2a — the async source converts a warming page cache into throughput
and the synchronous one largely cannot.** Across the same warming, early to
late in this session, the control moved 100.35 → 92.25 s (**1.09x**) while the
async source moved 48.09 → 30.28 s (**1.59x**). That is consistent with the
shipped source being bound by per-page synchronous faults *on the compute
thread*, where even a cache hit costs a fault and the GPU still waits.

**Finding 2b — the GPU is at idle clocks in one control block and not in any
async block, on two samples per block.** The harness reads `clocks.sm` — the
INSTANTANEOUS clock — exactly twice, once before the loop and once after, and
that is all the evidence there is. All six cold samples:

| block | start → end |
|---|---|
| control (early) | 180 → 180 MHz |
| control (late) | 1417 → 210 MHz |
| async `read_ahead 2` (early) | 1792 → 2355 MHz |
| async `read_ahead 1` | 1417 → 1095 MHz |
| async `read_ahead 4` | 202 → 435 MHz |
| async `read_ahead 2` repeat | 180 → 705 MHz |

**Correction, 2026-09-14 15:30.** This finding first read "the control's SM
clock is 180 → 180 MHz: it never **leaves** idle across a 10-minute block …
the async blocks reach 1417–2355 MHz". Two samples ten minutes apart cannot
support "never", and `sm_clock_mhz`'s own docstring warns that this box's boost
clock moved 442–952 MHz *inside* one measurement run. The second half was
wrong in a second way: 1417–2355 MHz describes two of the four async blocks,
and the headline block itself STARTS at 180 MHz. What the six samples support is
the weaker claim above — one control block sampled at idle on both ends, no
async block did — and that is all this instrument can say. The claim, as
originally written, propagated to `benchmarks/README.md` and to the changelog
fragment; both are corrected with it.

**Against the floor.** §18a put the read-free step at **14.25 s at seq 256**;
the numbers above are at **seq 512**, so that is not a like-for-like ratio and
is quoted as an order of magnitude, not a factor. For what the read path costs
*in this block*, the instrumented copy brackets are the measurement to use, not
that comparison: **84.7%** of the headline step (25.168 s of copy per 29.708 s
step, `async_cold_synth70b_nf4_ra2_repeat.json`), agreeing with §8's independent
per-layer accounting of 82.6–84.9%. **Still read-bound, and by about the same
fraction as before.**

**Correction, 2026-09-14 15:30.** The sentence here previously read "the best
cold step here is 30.28 s, so the read path is still roughly half the step" —
which is a factor, computed from the pair the sentence before it had just
declared not like-for-like, and it contradicted §8 and §10's 90–93%. It is
replaced by the one figure the block's own instrumentation gives. See also §11:
there is no valid read-free floor at seq 512 to compare against.

---

## 3. The depth series is an ORDERING result, not a depth result

This is the part that changed the headline, and it is left in the order it
happened.

| order | depth | step (uninstrumented) | source rate | free phys at its baseline | JSON |
|---|---|---|---|---|---|
| 1st | `read_ahead 2` | 48.09 s | 1.464 GB/s | 19.95 GB | `async_cold_synth70b_nf4.json` |
| 2nd | (sync control) | 100.35 s | 0.701 GB/s | 19.50 GB | `control_sync_cold_synth70b_nf4.json` |
| 3rd | `read_ahead 1` | 39.23 s | 1.794 GB/s | 24.99 GB | `async_cold_synth70b_nf4_ra1.json` |
| 4th | `read_ahead 4` | 33.59 s | 2.095 GB/s | ~25 GB | `async_cold_synth70b_nf4_ra4.json` |
| 5th | `read_ahead 2` **repeat** | **30.28 s** | 2.324 GB/s | ~24 GB | `async_cold_synth70b_nf4_ra2_repeat.json` |

Read the first four rows alone and depth 4 is the winner and depth 2 the worst.
Read the fifth and that collapses: **the same configuration is 48.09 s run first
and 30.28 s run last, 1.59x apart with nothing changed but position** — larger
than the entire spread across depths (33.59–48.09 s).

**Finding 3 — depths 1, 2 and 4 are not distinguishable on this evidence, and
what looked like a depth effect is consistent with the page cache warming across
blocks.** (That verb was "was" until 2026-09-14 15:30. The *ordering* result is
proved by the repeat; the *mechanism* is correlational — free physical memory
grew 19.95 → ~25 GB alongside 48.09 → 30.28 s — and session-scale clock and
thermal effects are not excluded, which the `sm_clock_mhz` docstring cited in
Finding 2b warns about directly.) The
brief anticipated "if depth changes nothing, say so"; the honest version is
stronger, because the confound would have produced a confident and wrong
recommendation. Free physical RAM grew across the sequence (Windows enlarged the
pagefile during the first control, §6), and with a 36.39 GB store against 20–25
GB of free RAM the cached fraction grows with it.

`read_ahead 8` was **not** run. The brief asks for it only if 4 still moves the
number, and after the repeat there is no evidence that any depth moves it.

Staging cost, which *is* a clean function of depth and is worth stating because
it is what a user pays: **1.515 GB at depth 1, 1.957 GB at depth 2, 2.839 GB at
depth 4** (441.43 MB per decoder slot; the two vocabulary groups hold one member
each and so take one slot apiece however deep the decoder runs).

---

## 4. The instrumentation is not always free here, and one outlier did not reproduce

`--step` runs each block twice, once plain and once with the CUDA-event
instrumentation on. On the RAM tier those agree to 0.06% (§3 of the probe
record). Here:

| block | plain | instrumented | ratio | JSON |
|---|---|---|---|---|
| sync control (cold) | 100.35 s | 104.95 s | 1.046x | `control_sync_cold_synth70b_nf4.json` |
| async `read_ahead 1` | 39.23 s | 38.63 s | 0.98x | `async_cold_synth70b_nf4_ra1.json` |
| async `read_ahead 2` | 48.09 s | **545.66 s** | **11.35x** | `async_cold_synth70b_nf4.json` |
| async `read_ahead 4` | 33.59 s | 33.38 s | 0.99x | `async_cold_synth70b_nf4_ra4.json` |
| sync control (cold, repeat) | 92.25 s | 93.48 s | 1.013x | `control_sync_cold_synth70b_nf4_last.json` |
| async `read_ahead 2` **repeat** | 30.28 s | 29.71 s | 0.98x | `async_cold_synth70b_nf4_ra2_repeat.json` |

**The 545.66 s is a block MEAN over one pathological step, not a block.** Its
six instrumented steps are 49.39, 47.71, 46.42, 47.07, 48.32 and **3035.06** s.
Five of the six sit inside the plain block's own 43.3–52.5 s range, so the
instrumentation was free in that block too; the artifact is a single step, and
3024 s of it is inside the copy brackets. (Added 2026-09-14 15:30. The per-step
shape is a sharper statement than the mean, and it would have refuted the 13:15
localisation below immediately, without needing the `read_ahead 4` block.)

**Correction, 2026-09-14 13:15 → 13:35.** On the strength of the first three
rows I wrote that the artifact was "specific to the async source" and localised
it to the read-ahead/release path — `_plan_queue`'s own docstring says the plan
is empty at depth 1, so depth 1 having no lookahead looked like the discriminator.
**`read_ahead 4` refutes that**: it has lookahead and release, and its
instrumentation is free. The localisation was wrong and is left standing with
this correction beside it.

**Finding 4 — the 545.66 s block is an outlier that did not reproduce.** The
same configuration instrumented, run last, is 29.71 s. §17a recorded the same
shape of thing on the shipped source (124 s plain against a 746 s instrumented
mean, with one step at 3129 s) and attributed it to the page cache. A targeted
pytest run belonging to another session appears in that block's AFTER stamp and
not in its BEFORE stamp, so contention is a candidate and is not established.
It is published as measured. Every headline in this record is an
**uninstrumented** block on both sides, which is like-for-like and unaffected
either way.

---

## 5. Warm, the async source is SLOWER — a negative result

Mistral-7B NF4, 4.138 GB store entirely in the page cache, batch 1 × seq 512,
8 timed after 3 warm-up, uninstrumented. Run in the order listed; the store is
4.1 GB against ~24 GB of free RAM, so all three were fully cached.

| order | arm | step | tok/s | source rate | JSON |
|---|---|---|---|---|---|
| 1st | async disk, `read_ahead 2`, pinned | **1.843 s** | 277.8 | 4.016 GB/s | `async_warm_mistral7b_nf4.json` |
| 2nd | control, shipped sync `DiskSource` | **1.535 s** | 333.5 | 4.822 GB/s | `control_sync_warm_mistral7b_nf4.json` |
| 3rd | RAM tier, same session | **0.816 s** | 627.8 | 9.075 GB/s | `ram_warm_mistral7b_nf4_today.json` |

**The async arm ran FIRST, which is the confound §3 exists to catch, and it was
not disclosed here until 2026-09-14 15:30.** The three `started` stamps are
14:11:34, 14:14:21 and 14:15:15, and the hour before them was cold 70B blocks
whose reads would have pressured the 4.1 GB Mistral store out of the page cache.
Three warm-up steps pull it back in, so the timed steps are cached either way —
but §3 had just established that position alone moved an identical configuration
by 1.59x on this box, and the bias here runs *against* the async arm. That is an
argument, not a measurement, and this record's standard is the difference.

**So the trio was re-run in the reverse order, 2026-09-14 15:57–16:01**, same
fixture, same flags, same harness, baseline commit charge `27.25 GB of 51.32 GB,
free phys 17.50 GB` before and `27.39 GB / 17.30 GB` after:

| order | arm | step | tok/s | source rate | JSON |
|---|---|---|---|---|---|
| 1st | RAM tier | **1.101 s** (1.090–1.121) | 465.1 | 6.723 GB/s | `ram_warm_mistral7b_nf4_reverse.json` |
| 2nd | control, shipped sync `DiskSource` | **2.022 s** (2.003–2.057) | 253.2 | 3.660 GB/s | `control_sync_warm_mistral7b_nf4_reverse.json` |
| 3rd | async disk, `read_ahead 2`, pinned | **2.089 s** (2.034–2.141) | 245.1 | 3.544 GB/s | `async_warm_mistral7b_nf4_reverse.json` |

**Finding 5 — with the whole store in the page cache the async source is slower
than the source it replaces in BOTH orders, by 1.20x when it runs first and
1.03x when it runs last.** The direction survives the confound; the magnitude
does not. Neither order gives a position-matched pair — the async arm and the
control never occupy the same slot — so 1.03–1.20x is the honest range and
neither end is the answer.

Two things make that reading defensible rather than a shrug. The arms that are
NOT the async source agree across the two sessions: control/RAM is 1.88x in the
first block and 1.84x in the second. And every absolute is ~1.3x slower in the
second block (RAM 0.816 → 1.101 s, control 1.535 → 2.022 s), so cross-session
absolutes are not comparable at all — only the within-block ratios are, which is
exactly why the same-day control exists.

The earlier claim here was "the async source costs **1.19x**" (1.84/1.54 rounded
from already-rounded inputs; from the stored values it is 1.20x), stated without
the order. Against §8's published warm before (2.46 s, 177–208 tok/s) the same
number would read as a **1.33x improvement**, and that framing would be wrong:
today's box is simply faster than that session. **This is the single clearest
argument for the same-day control**, and the reason §8 is not used as the warm
baseline here.

The reading is unsurprising once stated: with the store cached, a synchronous
`mmap` read is close to a `memcpy`, so there is nothing for a background reader
to hide, and the handoff, the staging copy and the release/drain synchronisation
are pure overhead. The async source exists for a store that does **not** fit
RAM. When it does fit, the RAM tier is the right answer anyway — 1.88x faster
than the synchronous disk arm and 2.26x faster than the async one in the first
block, 1.84x and 1.90x in the reverse one. (Those two figures read "1.9x" and
"2.2x" until 2026-09-14 15:30; both were rounded from already-rounded inputs,
and the second is 1.84315/0.81554 = 2.2601.)

Ablation on the warm async source (2 interleaved rounds, uninstrumented):

JSON: `async_warm_mistral7b_nf4.json` (the same file as the first warm block —
the ablation arms are extra records in it).

| arm | round 0 | round 1 |
|---|---|---|
| A baseline | 1.710 s | 1.752 s |
| B no source read, no copy | **0.740 s** | 0.738 s |
| C no NF4 dequantisation | 1.789 s | 1.808 s |
| D neither | 0.601 s | 0.589 s |

Removing the read path buys 57% (§10 measured 65% on the shipped source in its
own session), and arm B at 0.74 s sits just under the RAM tier's 0.82 s, as it
should — B removes the device copy too.

**Arm C is SLOWER than arm A in both rounds and that is not explained here.**
Removing NF4 dequantisation "costs" 4.6% and 3.2%; the sibling record's §10 saw
the same sign on the same fixture (−8.7% / −2.4%). The two rounds of each arm
differ by 2.5% (A) and 1.1% (C), so the effect is around the size of the
round-to-round spread and this record does not claim a mechanism for it.
(Added 2026-09-14 15:30 — publishing a negative saving with no comment was the
one place this section fell short of the record's own standard.)

---

## 6. Two incidental observations

**The commit limit moved, and only under the control.** Block 1's AFTER stamp
reads `commit charge 24.87 GB of 39.44 GB`; the first control's reads
`21.48 GB of 51.36 GB`. Windows grew the pagefile by ~12 GB **during the
control** and not during any async block. `DiskSource` memory-maps all 80 layer
shards and holds them for the run (36.39 GB), and a private mapping charges
commit for the file — #926 measured 48.99 GB of mappings raising the charge
46.17 GB. The async source never maps: it parses headers and `readinto`s a
pre-allocated buffer. That is the "no mmap" half of the spec showing up as a
side effect rather than as a designed measurement.

**The 30 s no-progress guard never fired.** No log from any block contains
"no progress", including the 3035.06 s step (the block whose MEAN is 545.66 s;
see §4). Reading the code, that is expected rather than lucky: the raise in
`get` is guarded by `self._in_flight != idx and idx not in self._queue`, so a
read that is merely slow keeps looping and only a reader that has stopped making
progress trips it. **The guess was not tested by these runs** — nothing here
establishes whether 30 s is the right number for a reader that genuinely stalls.

**Follow-up, 2026-09-14 15:30.** The whole-branch review then showed the
conjunction could not fire at all on the single-consumer path: the demand push
puts the layer in `_queue` before the wait, and the reader moves it
queue → `_in_flight` → `_slot_of` under the lock, so it is always in exactly one
of the three. It has been replaced by two checks that CAN fire — a reader thread
that exited without recording an error, and one read in flight past a limit
derived from §8 below. The 3035.06 s step would not have tripped the new limit
either, and should not have: its brackets are a consumer waiting, not one read.

---

## 7. `read_ahead` staging, for the operator

| depth | staging on the 70B fixture |
|---|---|
| 1 | 1.515 GB |
| 2 (default) | 1.957 GB |
| 4 | 2.839 GB |

All three measured, read off each run's own `store` line. The 7B fixture was
only ever run at depth 2, where it reports **0.762 GB**; the other depths are
not tabulated for it because they were not measured.

Each level costs one more decoder layer of **pinned** host memory. Since no
depth was distinguishable on throughput here (§3), the default of 2 is not
challenged by this record, and 1 is the cheaper choice if host RAM is tight.

---

## 8. The step-head stall is the embed fetch, not layer 0 — and neither is material

`_plan_queue` walks one index out of a single-member group, so a
vocabulary-sized read is planned ahead of decoder layer 0 at the moment the
compute thread blocks on layer 0. `benchmarks/harness/layer0_wait.py` drives the
probe's own instruments and labels each load bracket with the layer it belongs
to. Cold 70B, depth 2, three steps, no warm-up, instrumented from step 0 — so
its **absolute** step times (49.2, 39.4, 39.3 s) are not throughput evidence;
the ratio is what it measures. JSON:
`layer0_wait_cold_synth70b_nf4.json`.

| step | layer 0, forward / backward | other layers, mean / median / max | vocabulary loads | brackets |
|---|---|---|---|---|
| 0 | **17** / 271 ms | 252 / 283 / 420 ms | 662, 386 ms | 40.6 s of 49.2 s = 82.6% |
| 1 | **17** / 259 ms | 214 / 266 / 533 ms | 33, 22 ms | 33.5 s of 39.4 s = 84.9% |
| 2 | **16** / 268 ms | 213 / 270 / 327 ms | 33, 22 ms | 33.3 s of 39.3 s = 84.8% |

**RE-MEASURED 2026-09-14 15:53, and this table is the re-run.** The JSON
published first was produced by a copy of the driver in a session scratchpad: it
recorded `"driver": "C:\\…\\scratchpad\\layer0_wait.py"` and carried no
`source_class` key at all, while the committed `benchmarks/harness/
layer0_wait.py` hardcodes its own path and always writes `source_class`. The
block was identifiable as the async source by inference (`pinned: true`,
`read_ahead: 2`, `store_gb: 1.9566` — exactly depth-2 async staging, and
impossible for the control, whose `store_gb` is 0.0), but not verifiable, and
#379 — published numbers whose harness left with the machine — is the whole
reason that file is committed. Re-run from the committed path, baseline commit
charge `27.65 GB of 51.32 GB, free phys 16.96 GB` before and `27.25 GB /
17.24 GB` after, with another session's Python resident throughout (§9's
standing condition). The new JSON records `"driver":
"benchmarks/harness/layer0_wait.py"` and `"source_class": "AsyncDiskSource"`.

**Every figure moved and one finding did not hold.** The steps are faster
(49.2/39.4/39.3 against 65.0/55.0/52.9 — a warmer page cache, §3's effect
again), the absolute brackets are smaller, and the first-touch vocabulary read
is 662 ms rather than 874 ms. The previous table's numbers are left above in
this paragraph rather than deleted. **Every first-run figure quoted here
(65.0/55.0/52.9 s, 874 ms, layer 0's 23.21 and 19.49 ms, the 90–93% bracket
share) is SUPERSEDED, and because the re-run wrote the same filename the
working tree no longer holds the JSON they came from — it survives in git
history, at `0fd0f42f`, and it is the file that records a `driver` path under a
session scratchpad and no `source_class` key at all: the copy of the driver
that predates the committed harness.** They are quoted as history, never as
evidence, and the commit is named so that "history" stays checkable. (Added
2026-09-14 16:45, mirroring §1's note.)

**Finding 8 — layer 0's forward load is the FASTEST load of the step**, 16–17 ms
against a 213–252 ms mean, because the reader has the whole build and embedding
phase to stage it before the compute thread asks for it. Its backward visit is
ordinary. The vocabulary reads cost 662 and 386 ms exactly once, on the first
touch of the first step, then 33 and 22 ms. The addendum offered a follow-up
(refuse to plan out of a single-member group); **on this evidence it is not
needed**, and that is scoped to one fixture, one depth, one shape.

**Finding 8a — the largest single compute-stream stall is the VOCABULARY fetch,
not layer 0 — and that IS the step-head item the addendum asked this section to
check.** `_plan_queue` walks one index out of a single-member group, so the
embedding is the step's first read with nothing staged ahead of it. Per step,
from the `stalls` array of the JSON named above:

| step | total stall | `large:model.embed_tokens.weight` | `layer000` forward | mean, other 79 decoder layers |
|---|---|---|---|---|
| 0 | 33.17 ms | **32.66 ms** | 0.17 ms | 0.0021 ms |
| 1 | 47.92 ms | **32.61 ms** | 14.99 ms | 0.0020 ms |
| 2 | 47.58 ms | **32.91 ms** | 14.35 ms | 0.0020 ms |

The embedding out-stalls layer 0 in every one of the three steps, layer 0 is
second in every one of them, and the other 79 decoder layers (158 stall entries)
sit at 0.002 ms — four orders of magnitude below the embedding in all three
steps (15,700x / 16,300x / 16,200x), and three to four below layer 0 in steps 1
and 2 (7,500x / 7,100x; in step 0 layer 0 barely stalls, so that gap is only
84x). In steps 1 and 2 the embedding's stall is nearly its entire load — 32.61
of 33.12 ms, 32.91 of 33.46 ms — i.e. the compute thread waits out the step's
first read almost exactly, which is the predicted single-member-group behaviour
and not an inference about it. Step 0 is the exception in the useful direction:
the reader has the build phase to start that 662 ms first touch, so only
32.66 ms of it reaches the compute thread.

**It is still not material, so Finding 8's ruling stands.** The step's entire
stall — the embedding, `lm_head`, layer 0 and all 79 other decoder layers
together, every one of the 162 entries — is 33/48/48 ms, which is
**0.07–0.12% of the step** (0.067% / 0.122% / 0.121%), against per-load read
brackets that are 82.6–84.9% of the same steps. Refusing to plan out of a
single-member group would buy back at most that.

**Correction, 2026-09-14 16:45.** This finding first read "layer 0 IS the largest
single compute-stream stall … against a 0.206 ms mean over every other layer",
and `benchmarks/README.md` carried the same sentence. Both halves are wrong
against the file this section names, and they fail TOGETHER. The 0.206 ms mean
is only reachable by counting the 32.6 ms embedding entry among "every other
layer" — the mean over all 160 non-`layer000` entries is 0.2062 ms, of which the
embedding alone supplies 0.2041 (32.657/160), so the comparison figure is
carried almost entirely by the one entry that outranks layer 0. Exclude the
embedding and the mean is 0.002 ms; include it and layer 0 is not the largest.
**And it was not true of the FIRST run either**, which is worth stating because
the obvious story — a conclusion refreshed with new numbers and left standing —
is not what happened here. Recovered from git history (`0fd0f42f`, the JSON this
re-run overwrote), that run stalls on `large:model.embed_tokens.weight` for
39.06 / 39.06 / 32.97 ms against layer 0's 0.002 / 23.21 / 19.49 ms: the
vocabulary fetch out-stalled layer 0 there too, in all three steps. What layer 0
did have in that run was a three-to-four-order gap over the other decoder layers
in the two steps where it stalled at all (23.21 ms against a 0.018 ms mean,
19.49 against 0.002). So the defect was never a stale figure. It was comparing
layer 0 against the decoder layers, and then quoting a mean that silently
included the one entry that comparison had left out.

The same table carries the step's own accounting: the per-load brackets are
**82.6–84.9% of the step**, agreeing with the headline block's own instrumented
share of 84.7% (§2). This step is the read path. (That figure read "90–93%"
before the re-run; §2 and §10 are corrected to the re-measured one.)

**The read limit in `get` is derived from this table.** The slowest single load
bracket anywhere in the three steps is 662 ms; `_MAX_READ_SECONDS = 300` is
three orders of magnitude above it, and ~15x the ~20 s a 441 MB decoder layer
would take at this project's own worst measured source rate of 22 MB/s. A wedge
detector that fired on a merely slow read would be deleted by the first operator
to meet it.

---

## 9. Contention, block by block

The box is shared with a contributor-PR review session that runs test suites
and cannot be stopped. Every block stamps the commit charge, the free physical
memory and the other Python processes immediately before and after itself.

| block | baseline commit charge | free phys | other work |
|---|---|---|---|
| §1 verification (3 arms) | 28.46 GB of 39.44 GB | 15.04 GB | **a full pytest suite throughout** |
| async `read_ahead 2` | 21.66 GB of 39.44 GB | 19.95 GB | none at start; **a targeted pytest in the AFTER stamp** |
| control (early) | 21.82 GB of 39.44 GB | 19.50 GB | none at start; 1 pytest by 12:55, 3 by 12:59 |
| async `read_ahead 1` | 21.54 GB of 51.36 GB | 24.99 GB | none |
| async `read_ahead 4` | 21.38 GB of 51.36 GB | 24.88 GB | none |
| async `read_ahead 2` repeat | ~21 GB of 51.36 GB | ~24 GB | none |
| control (late) | ~21 GB of 51.36 GB | ~24 GB | none |
| warm 7B, three arms | ~21 GB of 51.36 GB | ~24 GB | none |
| layer-0 probe (first run) | 25.46 GB of 51.33 GB | 20.40 GB | none |
| layer-0 probe (**re-run**, §8) | 27.65 GB of 51.32 GB | 16.96 GB | another session's Python resident (1.3 GB) |
| warm 7B, **reverse order** (§5) | 27.25 GB of 51.32 GB | 17.50 GB | another session's Python resident (1.3 GB) |

Three of those stamps are approximations (`~21 GB` / `~24 GB`), including both
halves of the late pair the verdict rests on, where the protocol asks for the
measured one-liner immediately before every timed block. The load-bearing half —
whether other work was running — is stated definitely, so this is presentation
rather than a gap in the protocol, but it is a gap in the presentation and is
named rather than tidied away. The two 15:53–16:01 blocks carry real stamps.

The early control ran with pytest present and the early async block did not,
which biases that pair **in favour of the reported speedup**. The late pair
(§2, rows 3–4) is clean on both sides, which is why it is the one quoted as the
position-matched result.

**The warm trio's ORDER is its own confound and §5 now says so**, with a
reverse-order re-run beside it.

---

## 10. Verdict

The cold disk tier is no longer bound by synchronous page faults on the compute
thread. Position-matched and same-session, a cold 70B-shaped NF4 step went from
**92.25 s to 30.28 s (3.05x)** at the late position and from **100.35 s to
48.09 s (2.09x)** at the early one, with the source rate rising from 0.70–0.76
GB/s to 1.46–2.32 GB/s. Against §17's published 124.5 s it is 2.59x. The GPU
clock samples are deliberately NOT part of this verdict: two per block support
only the weaker statement §2b makes, and an earlier draft of this paragraph
claimed "the GPU leaving idle clocks for the first time on this tier" — the
claim §2b retracts (corrected 2026-09-14 16:45). **What bounds it now is
still the read**: the per-load brackets are **84.7%** of the headline step, and
§8's independent per-layer accounting puts them at 82.6–84.9%. There is no
valid read-free floor at this sequence to compare against — see §11 — so no
ratio is quoted for the remaining headroom. The remaining levers are the ones
§19 of the probe record already named and this project did not touch:
unbuffered sequential reads, both NVMe drives, layer-major micro-batching, and
#842.

Two things temper it. **Warm, this is a regression** (§5): with the store in the
page cache the async source costs 1.03–1.20x against the one it replaces
depending on run order, and the right answer there is the RAM tier. And **no
read-ahead depth was distinguishable** (§3) once the ordering confound was
controlled, so the depth knob is, on this evidence, a staging-cost knob rather
than a throughput one.

---

## 11. What was NOT measured

- **A real 70B.** The cold fixture is a synthetic Llama-70B *shape* with random
  weights. Nothing here is a correctness claim; the bit-exactness gates are
  Task 3's and were not re-run.
- **Correctness of any kind.** This record is timing only.
- **The NVMe itself.** No block-level read test; 3.5+ GB/s sequential is the
  drive's published figure, not measured here.
- **A second drive, striping, unbuffered/O_DIRECT reads, or #842.** All four are
  named in §19 of the probe record as the remaining levers and none is in #971.
- **`read_ahead 8`,** and any depth on the warm fixture (§3 says why 8 was
  dropped).
- **A read-free floor at seq 512, so there is none to compare the cold step
  against.** §18a's ~14.25 s is at seq 256 and extrapolated from 128/256; the
  sibling record's only seq-512 read-free arm (`probe-rtx5070-what-bounds-
  streaming.md` §18, 39.8 s) is *higher* than the best step measured here and is
  invalidated there by WDDM spill. §2 and §10 therefore quote the measured
  bracket share and no headroom ratio at all.
- **A position-matched warm pair.** §5 now carries both run orders, but the
  async arm and the control never occupy the same slot in either, so the warm
  regression is bracketed (1.03–1.20x) rather than measured to a figure.
- **Any shape other than batch 1 × seq 512,** and any sequence sweep. The
  read-free floor this is compared against was measured at seq 256/128.
- **Whether 30 s is the right no-progress timeout.** It never fired (§6); a
  reader that genuinely stalls was not constructed. The guard has since been
  replaced (§6's follow-up) and the new limit is derived from §8's slowest
  measured bracket — which is a bound taken from data, still not a stall
  constructed and observed on this hardware.
- **Any architecture beyond llama (the 70B shape) and mistral,** any OS other
  than Windows 11, any card other than this one, and any tier interaction with
  `stream_source: auto`'s own RAM-vs-disk decision.
- **A pageable-staging cold arm.** `--no-pin` was verified to work on the disk
  tier (§1) but no cold block was run with it, so the value of pinning *on this
  tier* is unquantified.
