<!--
Working measurement record, published verbatim.

Like the records beside it, this is the log kept while the work happened, not
a report assembled afterwards: the wrong assumption about the model cache, the
foreign process, and the drifting ceiling stay in, in the order they occurred.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB, driver 616.92,
PCIe 5.0 x8 negotiated), Intel i9-14900HX (8P+16E, no AVX-512/AMX), 31.7 GB
DDR5-5600 dual channel, 2 x Samsung PM9B1 NVMe, Windows 11 Pro 26100.
Stack: Python 3.12.10 · torch 2.14.0+cu130 · transformers 5.17.0 · peft 0.20.0
· bitsandbytes 0.50.2 · safetensors 0.8.0. Soup main at 2a97512a (v0.75.0 + 0).
Harness: benchmarks/harness/stream_probe.py (new in this record, re-runnable).
-->

# What bounds layer streaming on an RTX 5070 Laptop — a new baseline, not a comparison

**Status: THREE SECTIONS MEASURED (2026-09-12). §1 the RAM tier on a 7B NF4;
§2 the shipped disk tier on the same 7B with the page cache warm; §3 the
shipped disk tier COLD on a synthetic 70B-shaped NF4 store larger than RAM —
which first surfaced a shipped sharder defect (#926, §12–§14). Raw per-point JSON
for every block is under `benchmarks/results/probe-rtx5070/`.**

The dev box changed on 2026-09-10 (CLAUDE.md header). Every published
streaming number is from an RTX 3050 4 GB; this card has twice the VRAM, a
different architecture and a PCIe 5.0 x8 link, so **nothing here is compared
against a stored number** — this is the first baseline on this card, and the
question is the same one `probe-v0.73.0-what-bounds-streaming.md` asked of the
old one: is the streamed step bound by the bus, by the GPU, or by something
streaming adds?

Two things differ from that record, stated up front:

- **The model is Mistral-7B-Instruct-v0.3 (unsloth mirror), not Llama-3.1-8B.**
  Llama-3.1 is gated and this box has no Hub token. Mistral-7B is in the
  streaming allowlist since v0.72.3, has the same per-layer shape (hidden
  4096, intermediate 14336, 8 KV heads, 32 layers, **218.1M params per layer,
  112.5 MB per NF4 layer**) and a 32768 vocabulary instead of 128256, so the
  logits term is 4x smaller. Same layer, smaller head.
- **The harness is checked in.** The v0.73.0 probe's scripts were scratchpad
  files that left with the machine (#379). `benchmarks/harness/stream_probe.py`
  builds the model through the SHIPPED path (`shard_checkpoint` ->
  `build_streamed_model`), replicates the two shipped copy paths with CUDA
  events added (a source guard refuses to run if the shipped body drifts), and
  writes JSON after every point.

Unit convention: **decimal GB**, matching the other records.

---

## 0. Two things went wrong before the first number

1. **The "cached" Mistral had no weights.** `~/.cache/huggingface/hub` listed
   `models--unsloth--mistral-7b-instruct-v0.3`, and I read that directory name
   as "downloaded". It held config and tokenizer only; the three safetensors
   shards (14.5 GB) were fetched by the probe's own `resolve_model_weights`
   at ~2 GB/min, unauthenticated, before any measurement. Lesson: a hub cache
   directory is evidence of a *visit*, not of weights.
2. **A foreign CPU-bound process was running on the box** (`Soup_Connectome_Runtime`,
   ~1180 s of CPU by 20:19, writing 100 MB blocks) during the download and
   shard phases. It had exited by the time the timed steps started (system CPU
   load 16% at 20:24; sharding finished after the download). The timed numbers
   below were taken with it gone; the download/shard timings were not.
3. **The System log carries a `Kernel-Power 105: Power source change` at
   20:19:57**, i.e. during the download, before any timed step. At the time of
   writing the box is on AC (`Win32_Battery.BatteryStatus=2`, 59% and
   charging, scheme Balanced, GPU P0, `clocks.max.sm` 3090 MHz). Every timed
   number in this record was taken on AC; whether a charging battery shares
   the power budget with the GPU on this laptop is not known, and it is one
   candidate for the in-session ceiling drift in §2.

---

## 1. Setup and the two costs that are not the step

| | |
|---|---|
| model | unsloth/mistral-7b-instruct-v0.3, NF4 + double quant, 32 layers, untied head |
| store | **4.138 GB pinned** RAM (page-locked on first attempt; no fallback) |
| buffers | 2 x 112.5 MB decoder + one 268.4 MB large slot (embed / lm_head, bf16) |
| adapter | LoRA r=8, alpha 16, q/k/v/o; PagedAdamW8bit, lr 1e-4 |
| shape | batch 1, seq 512, synthetic ids, labels = ids |
| sharding | **19.6 s** (NF4 quantised on the GPU, one tensor at a time) |
| build | **4.6 s** (meta skeleton -> extras -> LoRA -> pinned store) |

Sharding 14.5 GB of bf16 into 4.1 GB of NF4 in 19.6 s is worth recording on
its own: the shard cache is a one-time cost of well under a minute here.

## 2. GEMM ceiling, same session

| probe | TFLOPS | reported SM clock |
|---|---|---|
| shipped square 4096^3, best-of-3 (samples 15.21 / 26.78 / 27.95 — boost ramping) | **27.95** | 1867 MHz |
| shape-matched at M=512: q/o 512x4096x4096 | 41.22 | |
| k/v 512x4096x1024 | 26.60 | |
| gate/up 512x4096x14336 | 30.40 | |
| down 512x14336x4096 | 29.63 | |
| **FLOP-weighted per layer** | **31.71** | 1657 MHz |
| the same two probes at the END of the session | 39.92 / **34.97** | 1530 MHz |

**The ceiling moved 10% inside one session**, and in the direction opposite to
the reported clock (a higher rate at a lower instantaneous reading). The clock
column is a point sample from `nvidia-smi`, and on this card the boost state
evidently changes faster than that sample resolves. Every fraction-of-ceiling
below is therefore quoted against the START value (31.71) with a ±10% band, and
nothing below turns on that fraction.

For scale: the RTX 3050's shape-matched ceiling was 7.55 TFLOPS. **This card is
~4.2x the old one on the GEMMs that matter.**

## 3. The headline step, and where its time goes

CUDA-event instrumentation on the copy stream and the compute stream, same
method as the v0.73.0 record (§3): copy time is how long the transfers take on
their own stream; stall is the interval the compute stream sits blocked in
`wait_event` — the only part of a transfer that is not hidden.

| quantity | per step | share |
|---|---|---|
| step time (8 timed steps, spread 1.041–1.069 s) | **1.053 s** | 100% |
| throughput | **486.0 tok/s** (486.3 instrumented — free) | |
| layer loads (measured, prefetcher skips owned slots) | 61.0 + 2 large | |
| bytes moved | 61 x 112.5 MB + 2 x 268.4 MB = **7.40 GB** | |
| copy time on the prefetch stream | **0.443 s** | **42%** |
| copy-stream rate | **16.7 GB/s** (PCIe 5.0 x8 pinned) | |
| average rate over the step | 7.03 GB/s | |
| **compute stream stalled on a copy** | **3.5 ms** | **0.34%** |
| effective rate (C=6 decoder, C=4 head: 42.41 GFLOP/token) | **20.6 TFLOPS** | 65% of ceiling (58% vs the end value) |
| peak VRAM | 1.342 GB allocated / 1.437 GB reserved | |

**Finding 3 — transfers are still hidden, but the margin is gone from 5.7x to
2.4x.** On the 3050 the copy stream was busy 17.5% of the step; here it is busy
42%. A card ~2.4x faster than this one at this shape, or this card at a
quarter of the tokens per step, is transfer-bound on this link. The stall says
the double buffer is still doing its job today.

**Cross-check against the old card.** The 3050 ran the 8B NF4 step (same
per-layer shape, bigger head) in 4.19 s; this card runs the 7B step in 1.05 s —
**4.0x**, against a 4.2x GEMM ceiling ratio. Compute-bound scaling, as the
stall number says.

## 4. The ablations, interleaved

Four arms switched by flag inside ONE process, A/B/C/D per round, two rounds.
Arms B, C and D compute garbage and are timing-only: B leaves stale bytes in
the pool (skips both the source read and the H2D copy), C multiplies by a
cached zero weight instead of dequantising, D does both.

| arm | round 0 | round 1 | vs A |
|---|---|---|---|
| A baseline | 1.065 s / 480.8 tok/s | 1.063 s / 481.8 | — |
| B **no host-to-device copies** | 0.994 s / 515.3 | 1.003 s / 510.6 | **−6.7% / −5.6%** |
| C **no NF4 dequantisation** | 0.819 s / 625.0 | 0.820 s / 624.7 | **−23.1% / −22.9%** |
| D neither | 0.780 s / 656.1 | 0.803 s / 637.6 | **−26.7% / −24.4%** |

Round-to-round spread: A 0.2%, B 0.9%, C 0.1%, D 2.9%.

**Finding 4 — the two streaming-specific costs are 25% of the step here, and
the split has inverted.** On the 3050 they were 11.3% (1.4% copies, 9.8%
dequant). Here copies are ~6% and the **NF4 dequantisation is ~23%** — the
per-visit `dequantize_4bit` into a dense bf16 tensor (the cost of the #331
repair) writes ~437 MB of VRAM per layer visit, and that write does not get
4.2x faster when the GEMMs do. **#842 (a checkpoint-visible fused
dequantise-and-multiply) is worth up to a quarter of the step on this card, not
a tenth.**

**Finding 4b — with both removed, 74% of the step remains, at ~27.5 TFLOPS =
87% of the start ceiling (79% of the end value).** The eager per-layer loop is
close to the GEMM ceiling once streaming's own costs are gone; the "gap to
ceiling" pool that was 28.7% on the 3050 is 13–21% here.

## 5. Sequence sweep: the fixed cost, and where the copy stops hiding

Batch 1, 8 timed steps per point, instrumentation on.

| tokens | step | tok/s | copy on its stream | stall | peak VRAM |
|---|---|---|---|---|---|
| 16 | 0.497 s | 32.2 | 0.368 s | 49 ms | 1.073 GB |
| 32 | 0.514 s | 62.2 | 0.368 s | 40 ms | 1.079 GB |
| 64 | 0.508 s | 126.0 | 0.369 s | 40 ms | 1.090 GB |
| 128 | 0.498 s | 257.1 | 0.367 s | 38 ms | 1.118 GB |
| 256 | 0.694 s | 368.8 | 0.376 s | 7.7 ms | 1.193 GB |
| 384 | 0.849 s | 452.2 | 0.413 s | 5.8 ms | 1.270 GB |
| 512 | 1.067 s | 479.8 | 0.454 s | 3.7 ms | 1.342 GB |

Least squares over 128–512: **step(S) = 0.311 s + 1.455 ms/token**, asymptote
687 tok/s. At S=512 that is 29% fixed / 71% proportional (the 3050 was 11% /
88%: the proportional part shrank 4.9x, the fixed part only 1.5x).

**Finding 5 — below ~128 tokens the step is a flat ~0.5 s, and it is NOT the
copies.** The copy stream is busy 0.37 s of that 0.5 s, but the compute stream
stalls on it for only 40–50 ms; the rest is the per-layer-visit CPU-side
overhead the v0.73.0 record measured at ~10 ms/visit (#841). On this card it is
~7 ms/visit (0.5 s / 64 visits) and it now binds up to ~200 tokens per step,
because the arithmetic it used to hide behind got 4x faster while the Python
did not. **The crossover moved up, not down: a faster GPU makes #841 worse.**

## 6. Section-1 verdict, one sentence

**On the RAM tier this card is compute-bound at batch 1 x 512 — the compute
stream waits on a copy 0.34% of the step and deleting every host-to-device byte
buys 6% — but the copy stream is now 42% occupied and the NF4 dequantisation is
23% of the step, so the two levers, in order, are #842 (fused dequant, up to
~23%) and #841 (per-visit fixed cost, binding below ~200 tokens), and the
transfer term becomes the bound at roughly 2.4x this card's speed or a quarter
of these tokens per step.**

What this does NOT say: anything about the disk tier, about batch > 1, about
bf16 streaming, or about a resident reference (an 8B does not fit resident on
8 GB at this shape either). Those follow.

## 7. What was NOT measured in Section 1

- **A resident baseline.** Not attempted; the point of this section is the
  decomposition, and the bit-exactness question on this card is #776.
- **bf16 streaming** and **batch > 1** under the ablation method.
- **Whether a fused dequantise-and-multiply recovers the 23%.** Arm C measures
  what removing the dequantisation is worth, not what a real fused kernel
  costs.
- **The clock.** `nvidia-smi` point samples disagreed with the measured rates
  in direction; a sustained-clock trace (dmon) was not run.

---

# SECTION 2 — the shipped disk tier, same model, page cache WARM

**Status: MEASURED (2026-09-12). This is the first RAM-vs-disk number the
project has (#325), and it is a measurement of the read PATH, not of the
NVMe: a 4.1 GB store on a 31.7 GB box sits entirely in the page cache.**

Same shards, same adapter, same shapes; `--tier disk` builds the shipped
`DiskSource` (one `safe_open` handle per layer, `get_tensor` on demand,
pageable, no pinning). The GEMM ceiling re-measured in this session: square
43.12 / shape-matched 36.26 TFLOPS at the start, 34.56 at the end.

## 8. The headline step from disk

| quantity | RAM tier (§3) | disk tier, warm |
|---|---|---|
| step | 1.053 s | **2.46 s** uninstrumented (spread 1.70–3.01) / **2.89 s** instrumented (2.77–2.94) |
| throughput | 486 tok/s | **177–208 tok/s** |
| "copy" bracket on the prefetch stream | 0.443 s | **1.72 s** (4.3 GB/s) |
| compute stream stalled on a copy event | 3.5 ms | **0.2 ms** |
| peak VRAM | 1.342 GB | 1.342 GB |

Two readings that only make sense together. First, the bracket on the copy
stream now includes the CPU-side `get_tensor` that runs BEFORE the copy is
enqueued, so "4.3 GB/s" is the rate of mmap page-cache read + pageable
staging + H2D, not of PCIe. Second, the compute-stream stall is essentially
ZERO even though the step is 2.3–2.8x longer: the compute stream never waits
on the copy event because the **Python thread itself is blocked inside
`get_tensor` and cannot enqueue the next layer's kernels**. The GPU starves,
and the starvation appears as wall time, not as a stall. That is the
mechanism, and it is the thing an async tier has to fix: the read must leave
the compute thread, not merely get faster.

Uninstrumented was faster than instrumented here (2.46 vs 2.89 s) with a
1.70–3.01 s spread; on the RAM tier the two agreed to 0.06%. The disk path's
variance is the page cache's, and the instrumented figure is the one the
ablation rounds reproduced (2.88–3.02 s).

## 9. Sweep from disk: flat from 16 to 512 tokens

| tokens | step | tok/s | bracket | stall |
|---|---|---|---|---|
| 16 | 2.756 s | 5.8 | 1.729 s | 2.6 ms |
| 32 | 2.880 s | 11.1 | 1.800 s | 2.7 ms |
| 64 | 2.779 s | 23.0 | 1.761 s | 2.7 ms |
| 128 | 2.905 s | 44.1 | 1.848 s | 2.8 ms |
| 256 | 2.938 s | 87.1 | 1.844 s | 2.8 ms |
| 384 | 2.920 s | 131.5 | 1.807 s | 1.2 ms |
| 512 | 2.958 s | 173.1 | 1.749 s | 0.3 ms |

**Finding 9 — the disk tier is I/O-path-bound at every token count.** The
step does not move between 16 and 512 tokens; the entire RAM-tier
proportional term (0.75 s at 512) hides under the read. Tokens per step are
free until the read is hidden, which is the amortisation argument in one
table: on this tier, batch is the lever and it costs nothing until ~1000
tokens.

## 10. Ablation from disk

| arm | round 0 | round 1 | vs A |
|---|---|---|---|
| A baseline | 3.020 s / 169.5 tok/s | 2.881 s / 177.7 | — |
| B **no source read, no copy** | **1.043 s / 490.8** | **1.058 s / 483.7** | **−65.5% / −63.3%** |
| C no NF4 dequantisation | 2.756 s / 185.8 | 2.811 s / 182.2 | −8.7% / −2.4% |
| D neither | 0.851 s / 601.9 | 0.832 s / 615.4 | −71.8% / −71.1% |

**Finding 10 — arm B is the RAM tier, to the millisecond.** Removing the read
path returns 1.04–1.06 s, i.e. §3's 1.053 s. So the shipped disk tier costs
**~1.9 s per step on top of the RAM tier, with the whole store in the page
cache**, for 7.4 GB moved: an effective 3.9 GB/s through
mmap -> tensor -> pageable `copy_`. The dequantisation (23% on the RAM tier)
is worth 2–9% here because it hides under the read. A cold store, one that
does not fit the page cache, can only be slower than this; that is Section 3.

## 11. What Section 2 does NOT say

- Nothing about NVMe throughput: no byte reached the SSD during the timed
  steps (4.1 GB store, 31.7 GB RAM, the shards had just been written).
- Nothing about a pinned staging path or a reader thread — neither exists in
  the shipped code (`DiskSource.get` is a synchronous `get_tensor`, prefetch
  depth is exactly one layer). This section measures the shipped path only.

---

# SECTION 3 — a cold store: the sharder crashed first

**Status: BLOCKED ON A SHIPPED DEFECT, found and reproduced 2026-09-12.**

The cold measurement needs a store that cannot fit the page cache. No 70B
checkpoint is on this box and Llama-3.1-70B is gated, so
`benchmarks/harness/synth_checkpoint.py` (new) writes a Llama-70B-SHAPED
checkpoint with random N(0, 0.02) weights: 80 layers of hidden 8192 /
intermediate 28672 / 8 KV heads, **855.6M params per layer, 441.4 MB per NF4
layer**, and a 32k head so the resident large slot is 0.5 GB rather than 2.1 GB
(the harness bypasses the trainer's VRAM pre-flight, so the 8 GB budget was
kept by hand). 68.99B params, 137.98 GB of bf16, one safetensors file per
layer, written to the second NVMe in **154.8 s**. Timing is real; values are
not, and nothing here is a correctness claim.

## 12. `shard_checkpoint` dies with an access violation at layer 61, twice

Both attempts (the probe's own sharding, then a bare re-shard under
`faulthandler`) wrote `layer_000..060` (26.9 GB) and then died with
**`Windows fatal exception: access violation`**, no Python traceback, exit 139,
no Event Log entry, no crash dump. `faulthandler` places it in
`torch/storage.py:480 __getitem__`, reached from `layer_shard._read_tensor`
line 1506 — i.e. inside `safe_open(...).get_tensor(key)`.

## 13. Reproduced without Soup, without CUDA, without bitsandbytes

`benchmarks/harness/mmap_limit_repro.py` (checked in): open the 80 layer
files with `safe_open`, call `get_tensor` on every key, keep nothing.

| pattern | result |
|---|---|
| one `ExitStack` holds ALL 80 handles open (the shipped sharder's pass-2 pattern) | **access violation after file 61 — 62 files, 106.1 GB of mappings, 26 s** |
| open / read / close each file in turn | **OK, all 80 files, 3 s** |

Three facts that narrow the cause:

- `get_tensor` is a **zero-copy view over the mmap** on this stack: it returns
  in 0.000 s, a repeat call returns the same `data_ptr`, and the first
  `.float().sum()` over a 470 MB tensor takes 0.588 s (0.8 GB/s — page faults
  through the mapping, not sequential reads); the second takes 0.045 s.
- The system commit limit is **36.7 GB** (31.7 GB RAM + a 5 GB page file), so
  the crash at ~106 GB is **not** copy-on-write commit exhaustion; read-only
  file mappings do not charge commit.
- The per-file variant "read" 137 GB in 3 s, i.e. it read nothing: with no
  consumer touching the pages the views are free. The all-open variant spent
  0.42 s per file, which is the D: drive's sequential rate — so with 60+ live
  mappings the SAME calls were paging data in. What Windows does differently
  past ~60 concurrently mapped 1.7 GB sections is not established here.

**What is established:** the shipped sharder holds every source shard's
mapping open for the whole of pass 2 (`layer_shard.py:1262-1267`, one
`ExitStack`), and on Windows that pattern dies after ~62 files x 1.7 GB.
Every published sharding ran on checkpoints of 4 (8B) to ~37 (72B on Linux)
files; the Windows box never sharded more than 4. The fix shape is the
per-file pattern above: open the file a layer needs, read it, release it —
an LRU of one or two handles, since layers are contiguous in files. Whether
the limit is a COUNT of live mappings or their total BYTES decides whether a
real 30-file 70B checkpoint is affected; §14 tests that with the same bytes in
20 files.

## 14. Count or bytes: the same checkpoint in 20 files

The same 137.98 GB written again as 20 files of 4 layers (6.85 GB each),
second NVMe, 154 s. The all-open reader over it:

| files live | crashed after | bytes mapped at the crash |
|---|---|---|
| 80 x 1.71 GB | file 61 (62 files done) | **106.1 GB**, +1.7 GB in flight |
| 20 x 6.85 GB | file 13 (14 files done) | **95.8 GB**, +6.85 GB in flight |

**Finding 14 — it is bytes, not files: the box dies once ~96–103 GB of
safetensors mappings are alive in one process, at 14 handles or at 62.** A
real Llama-3.1-70B checkpoint is 30 files and ~141 GB, so the shipped sharder
cannot shard it on this OS; a 32B bf16 (~65 GB) is under the line and a 72B
is not. The threshold's mechanism is not established (commit is 36.7 GB, so
it is not commit; the reads are lazy views, so it is not resident memory);
its value is. Filed as #926 with this table.

## 15. The workaround used to get past it, and what it does NOT change

`stream_probe.py --lazy-shard-handles` replaces `safetensors.safe_open` for
the duration of `shard_checkpoint` ONLY with a wrapper that keeps at most two
real handles alive (least-recently-used release) and returns a copy from
`get_tensor` rather than the zero-copy view, because the sharder keeps
non-quantised tensors (norms, embeddings, the head) until it writes them and
a view over a released mapping is precisely the access violation above. The
sharder's own logic runs unchanged; the runtime (`DiskSource`, the pool, the
prefetcher) and every timed number below are the shipped code. The runtime
itself holds one handle per decoder layer for the whole run — 80 x 441 MB =
35 GB of mappings here, under the line; a 400B-class NF4 store would not be.

**Correction 2026-09-13, found while writing the fix (#926):** the reason
given above for the copy is wrong, and the copy is still needed. Measured
on this stack (safetensors 0.8.0, torch 2.14): a `get_tensor` view SURVIVES
its handle's `__exit__` — the tensor's storage owns the mapping, so a later
`get_tensor` on the closed handle raises `SafetensorError: File is closed`
while the earlier view still reads its bytes correctly. A released handle is
therefore not a use-after-unmap. It is the opposite problem: every retained
view keeps its whole file mapped, so an LRU of two handles alone would not
bound the live mappings; only tensors the sharder OWNS (a copy) release
them. The workaround did the right thing for the wrong reason, and the
sentence above is left as written.

## 16. Sharding and build, with the workaround

| | |
|---|---|
| source | 20 files, 137.98 GB bf16, second NVMe |
| sharding | **216.9 s** — 138 GB in, **36.39 GB** NF4 out, one tensor at a time on the GPU |
| store | 0 GB resident (disk tier), 36.39 GB on disk: **larger than the 31.7 GB of RAM**, so no page cache can hold it |
| buffers | 2 x 441.4 MB decoder + one 536.9 MB large slot |
| build | 5.2 s |
| adapter | LoRA r=8 q/k/v/o on 80 layers |
| ceiling, this session | square 30.34 / shape-matched **31.43 TFLOPS** at M=512 |

## 17. The cold step

Batch 1, seq 512, 6 timed steps after 2 warm-up, uninstrumented:

| quantity | per step |
|---|---|
| step time (spread **93.6–158.1 s**) | **124.5 s** |
| throughput | **4.1 tok/s** |
| layer loads | 157 + 2 large (80 layers x 2 visits, minus the owned-slot skips) |
| bytes moved | **70.4 GB** |
| average source rate | **0.57 GB/s** |
| effective compute | 1.69 TFLOPS = **5% of the ceiling** |
| peak VRAM | **4.38 GB** allocated / 4.83 GB reserved (a 70B-shaped decoder trains inside 8 GB, as the arithmetic said) |
| SM clock at the end of the block | 210 MHz — the GPU is idling between layers |

**Finding 17 — the shipped disk tier, cold, is ~12x I/O-bound, and the
NVMe is not the reason.** The drive reads 3.5+ GB/s sequentially and there
are two of them; the run pulls **0.57 GB/s**, because `DiskSource.get` is a
synchronous `get_tensor` over a memory map, so every byte arrives through
4 KB page faults on the compute thread, one layer at a time, with the GPU
waiting. The arithmetic floor for the same step is ~10 s of compute (the
§18 ablation measures it); the other ~115 s is the read path. A source that
issued sequential, unbuffered reads on a background thread into pinned
staging would have ~6x the bandwidth at one drive and ~12x at two — and would
still need tokens-per-step (§9) to hide under compute. Both halves are needed
for a 70B on this laptop; neither exists in the shipped code.

The per-step spread (93.6 to 158.1 s) is the page cache's: with 17 GB free
around a 36 GB working set, which layers survive between steps is up to
Windows.

## 17a. The same step again, instrumented: a 6x spread, and a 22 MB/s floor

The second block (same shape, CUDA-event instrumentation on) did not
reproduce the first:

| block | steps, s | mean | source rate |
|---|---|---|---|
| uninstrumented | 123, 109, 150, 94, 113, 158 | **124 s** | 0.57 GB/s |
| instrumented | **3129**, 165, 199, 235, 393, 354 | 746 s | 0.094 GB/s |

The instrumentation is free on the RAM tier (§3) and costs nothing here
either — the bracket on the copy stream simply reports where the time went
(653.6 s of the 746 inside the read+copy path; compute-stream stall 1.4 ms).
What differs is the page cache. **One step took 3129 s: 70.4 GB at 22 MB/s**,
which is what a memory-mapped read degrades to when every page misses and
each miss is a synchronous 4 KB fault at queue depth 1 — random-read
territory on a drive that streams sequential reads at 3.5 GB/s. The other
steps sat between 165 and 393 s as some layers survived between visits.

**Finding 17a — on a store larger than RAM the shipped disk tier is not
merely slow, it is unstable by an order of magnitude from step to step**:
0.7 to 4.1 tok/s across two blocks of the same configuration, 22 MB/s to
0.57 GB/s at the source. Neither number is the NVMe's. The run was stopped
after these two blocks; a full A/B/C/D ablation at this pace would take
hours, so §18 measures only the arms that remove the read path.

## 18. The compute floor with the read path removed, and a card at its edge

Arms B (no source read, no copy) and D (B plus no dequantisation) only,
`--steps 3 --warmup 1`, same shape (batch 1, seq 512). Sharding was skipped
(the cache index now exists), build 5 s.

| arm | step | tok/s | effective |
|---|---|---|---|
| B no read, no copy | **39.8 s** | 12.9 | **5.3 TFLOPS** (17% of the 31.4 ceiling) |

That is not the compute floor the arithmetic predicts (~210 TFLOP per step at
the 7B's 20.6 TFLOPS effective would be ~10 s). Sampled during arm B:
`nvidia-smi` reported **7,620 of 8,151 MiB in use, 100% utilisation, 1.8 GHz,
40 W** — torch's own peak was 4.38 GB allocated / 4.83 GB reserved. A card
reporting full at 100% "utilisation" while drawing 40 W and delivering a
quarter of its measured rate is the WDDM spill signature the v0.72.3 and
v0.73.0 records describe: allocations past the card go to shared host memory
without an error and the run "merely" collapses. The likely driver here is the
#331 dequant-forward, whose dense transients for ONE 70B layer are 1.7 GB
(seven linears up to 470 MB each, alive inside the recomputed block), on top
of 0.88 GB of buffers, a 0.54 GB large slot, adapters and activations. The
harness bypasses the trainer's VRAM pre-flight, so nothing refused this shape;
§18a re-measures the floor at a shape that leaves headroom.

Arm D at seq 512, for the record: 67.2 s — slower than B, with the reported
clock falling 1965 -> 1462 MHz across the block. The card was at its memory
edge and, an hour into sustained load, evidently also at its thermal one
(67 °C and 1.7 GHz at idle afterwards); neither number at this shape is used
below.

## 18a. The floor at seq 256 and 128, with headroom

Same arms, `--steps 3 --warmup 1`, peak allocated 3.0–3.9 GB (the card is no
longer full), reported clock 1.07–1.86 GHz throughout.

| tokens | B: no read, no copy | D: B + no dequantisation | dequantisation = B − D |
|---|---|---|---|
| 256 | **14.25 s** / 18.0 tok/s / 7.4 TFLOPS | **6.73 s** / 38.0 tok/s / 15.7 TFLOPS | **7.5 s = 53% of B** |
| 128 | 14.46 s / 8.9 tok/s | 7.44 s / 17.2 tok/s | 7.0 s |

**Finding 18a — with the read path removed, a 70B-shaped step on this card
is ~14 s regardless of tokens, and half of it is the NF4 dequantisation.**
Dequantisation is paid per layer visit — 157 visits x 1.7 GB of dense bf16
written per step, 268 GB — so it does not shrink when the batch does; at
these token counts it is the same size as all the arithmetic put together.
The other half is also nearly flat between 128 and 256 tokens: the GEMMs at
M=128–256 read the 1.7 GB dense layer once per pass and are memory-bound, and
the ~160 per-visit launch/Python overheads (#841) come to seconds at this
depth where they came to a third of a second on 32 layers.

**What that means for "70B on this laptop", in numbers:** the shipped code's
step is ~14 s of dequant+compute plus 100–3100 s of read. A source that
streamed the 70 GB per step at the drive's sequential rate (3.5 GB/s, ~20 s;
two drives, ~10 s) would bring the step to ~25–35 s and ~10 tok/s at 256
tokens — worth having, and still 3x short of the floor, because the read
and the dequantisation are BOTH per-visit costs. Amortising visits over
several micro-batches (the L2L layer-major loop: load and dequantise a layer
once, run k micro-batches through it) divides both by k at the memory cost of
k sets of boundary activations. That, not a faster SSD alone, is what closes
the gap between "runs" and "usable" at this size.

## 19. Verdict, in one paragraph

On the RTX 5070 Laptop the RAM tier is still compute-bound at batch 1 x 512
(486 tok/s on a 7B NF4; stall 0.34%; deleting every H2D byte buys 6%), but
the copy stream is already 42% busy and the NF4 dequantisation is 23% of the
step, so #842 (fused dequant) is the first lever and #841 (per-visit fixed
cost) binds below ~200 tokens. The shipped disk tier is a different animal: a
synchronous memory-mapped read on the compute thread that costs 2.3x with the
whole store in the page cache and degrades to 22 MB/s–0.57 GB/s when it is
not, so a 70B-shaped model trains at 0.7–4 tok/s here while the same card,
with the read path removed, would do 18 tok/s (B) and 38 tok/s (D). The path
to a 70B on a laptop is therefore, in order: (1) an asynchronous source with
sequential unbuffered reads and pinned staging, off the compute thread, over
both NVMe drives; (2) layer-major micro-batching so each read AND each
dequantisation serves k micro-batches; (3) #842. And before any of that, the
sharder has to stop holding every source mapping open (§13–§14), or a real
70B checkpoint cannot be sharded on Windows at all.

## 20. What Section 3 did NOT measure

- **A real 70B.** Shapes and bytes are Llama-3.1-70B's decoder; the head is
  32k; the values are random. Nothing here is a correctness claim, and the
  bit-exactness gates were not run on the synthetic model.
- **The NVMe itself.** No block-level read test was run; 3.5 GB/s sequential
  is the drive's published figure, not measured here. The `--evict-gb`
  heuristic was not used (the store exceeds RAM, so cold is structural).
- **A/B/C/D at seq 512 with headroom**, and any arm A on the cold tier
  beyond the two step blocks — the run was stopped for time.
- **Thermal state.** The last blocks ran an hour into sustained load with
  reported clocks 1.0–1.9 GHz against 1.65–1.9 GHz at the start; the B/D
  ratios within one block are the defensible numbers, absolute rates less so.
- **The Windows mapping limit's mechanism** — only its value (~96–106 GB of
  live safetensors mappings) and its independence from file count.

---
