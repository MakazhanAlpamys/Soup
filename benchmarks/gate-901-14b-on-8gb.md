<!--
Working measurement record, published verbatim. The wrong turn (a memory
hypothesis that the first probe run refuted), the pure-torch reductions, and
the numbers that were measured and then found to be the wrong instrument are
kept in the order they happened.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB by nvidia-smi /
8.518 GB by torch, driver 616.92), Intel i9-14900HX, 31.7 GB DDR5-5600,
Windows 11 Pro 26100. Fixture on D: (NVMe), NF4 shard cache on C: (NVMe).
Stack: Python 3.12.10 · torch 2.14.0+cu130 · transformers 5.17.0 · peft
0.20.0 · trl 0.29.x · bitsandbytes 0.50.2.
Soup: branch fix/stream-901 (v0.75.0 + the four commits this record gates),
worktree C:\Users\user\projects\Soup-stream, own venv, soup_cli.__file__
checked under Soup-stream\src before every run.
Harness: benchmarks/harness/synth_checkpoint.py (the 14B fixture),
benchmarks/harness/issue901_stream_step_probe.py (the step probe), plus three
throwaway pure-torch scripts quoted inline in section 3. Raw JSON for the two
surviving probe runs and the end-to-end log are under
benchmarks/results/probe-rtx5070/issue901_*.
-->

# Gate record — #901: "14B OOMs inside SFTTrainer.__init__ on an 8 GB card"

**Status: REPRODUCED, EXPLAINED, FIXED, 2026-09-15.** The report was never
about VRAM. A failed page-lock of the 9.93 GB store left a stale CUDA error on
the runtime, the shipped fallback to a pageable store did not clear it, and
the run's first kernel launch reported that stale error as `CUDA error: out of
memory` with 7.3 GB free. The page-lock itself failed because torch's caching
host allocator rounds every pinned request up to the next power of two, so
pinning the store one tensor at a time cost 1.73x its size. After the fixes the
same config completes end to end on this 8 GB / 32 GB box with the store
page-locked: measured peak **2.944 GB allocated / 3.244 GB reserved** against
the pre-flight's **3.39 GB** prediction, which was right all along.

**No throughput and no quality claim is made here.** The fixture is a synthetic
Qwen2.5-14B *shape* with random weights: peak memory and pinned-memory cost
depend on tensor sizes, not on values, and that is all this record measures.

Unit convention: decimal GB unless a figure is a power of two, which is
written in GiB.

---

## 0. The report, and the box that reproduced it

| | reporter (#901) | this box |
|---|---|---|
| GPU | RTX 3070 8 GB, Windows 11 | RTX 5070 Laptop 8 GB, Windows 11 |
| RAM | 32 GB | 31.7 GB |
| free VRAM at pre-flight | 7.41 GB | 7.35 GB |
| model | Qwen/Qwen2.5-14B-Instruct | synthetic Qwen2.5-14B shape (`--arch qwen2`) |
| store | 9.93 GB NF4, 48 layers, "could not page-lock … PAGEABLE" | 9.932 GB NF4, 48 layers, same message |
| pre-flight prediction | 3.39 GB at batch 1 x seq 384 (logits 0.82 GB) | 3.39 GB, identical line |
| failure | `AcceleratorError: CUDA error: out of memory` in `SFTTrainer.__init__` (or in the probe's forward) | `AcceleratorError: CUDA error: out of memory` at the first CUDA op after the build |

The reporter's own control — gemma-2-2b streamed to completion on the same
box — is what said "something at 14B scale", and it turns out to be "something
at 9.93 GB of store": the 2B store pins, the 14B one does not, and only a
failed pin poisons the run.

## 1. The fixture

`benchmarks/harness/synth_checkpoint.py` gained `--shape qwen2.5-14b --arch
qwen2` (48 layers, hidden 5120, intermediate 13824, 40/8 heads, a 152064-row
untied head, q/k/v biases): 14.77 B parameters, 29.54 GB bf16 on D:. The
generator took 2178 s because of one 2124 s stall before layer 28 (timestamps
11:51:24 → 12:27 on the layer files); the per-layer write is 0.9 s before and
after it, and the stall is not explained — two other sessions were running
suites on the box. Timing hygiene only; nothing here depends on the generator.

Sharding to NF4 (double quant, quantised on the GPU): 56.2 s, 9.932 GB across
48 layer shards + 2 large shards + extras.

## 2. Reproduction — and the memory hypothesis it refuted

Baseline before the run: free physical RAM 15.48 GB, commit charge 23.54 GB of
47.35 GB, two other Python processes at ~0.1 GB each; GPU 0 MiB used.

`issue901_stream_step_probe.py --seq 384 --batch 1 --lora-r 16 --lora-targets
q_proj,v_proj` (the reporter's shape; `q_proj,v_proj` is what PEFT's
`target_modules: auto` picks for Qwen2, and it reproduces the reporter's
12.58M trainable):

```
layer streaming could not page-lock the base (AcceleratorError); falling back to a PAGEABLE RAM store. ...
shards        ... (56.2 s); build 40.0 s
store         9.932 GB pageable tier ram; buffers 2 x 142 MB + large slot 1557 MB
after build   allocated 1.895 GB  reserved 1.919 GB  free 5.393 GB
...
    ids = torch.randint(0, int(config.vocab_size), (args.batch, args.seq), device=device)
torch.AcceleratorError: CUDA error: out of memory
```

**Finding 2 — the first CUDA op after the build fails, and it is a 3 KB
allocation with 5.39 GB free.** Whatever this is, it is not the arithmetic:
`torch.randint` for 384 int64s cannot need 5 GB. The hypothesis I brought to
the run — that the 1.557 GB embedding, the 1.557 GB `lm_head` and the 1.557 GB
large-layer slot were all resident and the formula counted one of them — died
here, before a single decoder layer was touched. The maintainer's comment on
the issue had two candidates, "the formula under-predicts" and "WDDM
fragmentation of the 1.5 GB slot"; neither survives a 3 KB allocation failing.

The one thing the failing run had done to the CUDA runtime before that line
was the failed page-lock.

## 3. Reduction to pure torch, no Soup

Three throwaway scripts (kept in the session scratchpad, quoted here; the
durable form is the test file).

**(a) Pin N GB, then run CUDA ops.** `torch.empty(int(gb*1e9), uint8,
pin_memory=True)`, then `randint` + a 2048² matmul + a 1.5 GB device alloc:

| pin request | pin | next CUDA ops | after freeing |
|---|---|---|---|
| 9.93 GB (the store) | **FAILED** `AcceleratorError: CUDA error: out of memory`, 0.1 s | **FAILED**, same error, free VRAM 7.322 GB | cuda ok |
| 4 GB | ok, 0.9 s | ok | ok |
| 7 GB | ok, 1.8 s | ok | ok |
| 8 GB | ok, 1.7 s | ok | ok |
| 9 GB | **FAILED**, 0.0 s | **FAILED** | — |
| 9.93 GB in 5 x 1.99 GB chunks | ok, 2.1 s total | ok | ok |

Two facts at once: the failure is per-request (9 GB in one block fails, 9.93 GB
in chunks succeeds), and after a failed pin the *next* CUDA op fails while the
one after it works ("after freeing … cuda ok" — nothing was held).

**(b) Which op trips, which clears.** After a failed 9.93 GB pin:

| sequence | result |
|---|---|
| malloc, sync, memcpy, kernel, kernel | malloc ok · sync ok · memcpy ok · **kernel FAILED** · kernel ok |
| pinsmall, kernel, kernel | pinsmall ok · **kernel FAILED** · kernel ok |
| sync, sync, kernel, kernel | sync ok · sync ok · **kernel FAILED** · kernel ok |
| lasterr, kernel, kernel | no `cudaGetLastError` binding in `torch.cuda.cudart()` (only `cudaError`, `cudaGetErrorString`) · **kernel FAILED** · kernel ok |

**Finding 3 — a failed `cuMemHostAlloc` leaves the runtime's per-thread last
error set; `cudaMalloc`, `cudaDeviceSynchronize` and a memcpy do not consume
it, a kernel launch's error check does, and clears it.** So the fallback path
hands the run a context whose first kernel launch is doomed: in the report that
launch was `param.data.to(torch.bfloat16)` inside `SFTTrainer.__init__`; with
the probe on it was the probe's forward, which the probe filed as "the CUDA
context may no longer be usable"; here it was `randint`. Lowering `max_length`
could never have changed it.

**(c) Why the pin failed at 9.93 GB with 15 GB of RAM free.** `RamSource` pins
one `torch.empty` per tensor (30 per layer). Replicating that against the real
14B shard cache, with `psutil` private bytes beside the request:

| pattern | requested | private commit delta | ratio |
|---|---|---|---|
| per tensor (shipped) | 6.82 GB (48 decoder layers) | **+11.83 GB** | 1.73x |
| one flat buffer per layer | 6.82 GB | +12.64 GB | 1.85x |
| 100 x 35.39 MB (one gate_proj's packed NF4) | 3.54 GB | +6.72 GB | **1.90x** |
| 100 x 64 MiB immediately after freeing the above | 6.71 GB | +0.00 GB | cache reuse |
| 100 x 65 MiB after that | 6.82 GB | `CUDA_ERROR_OUT_OF_MEMORY from cuMemHostAlloc` | — |

And torch's own counters say it without a heuristic:
`torch.cuda.host_memory_stats()['active_bytes.current']` moves by **131 072**
for a 100 000-byte request, by 131 072 for a 131 072-byte one, and by
**67 108 864** for a 35 389 440-byte one; freeing leaves
`allocated_bytes.current` at 67 371 008 (cached) with `active_bytes` at 0, and
`torch._C._host_emptyCache()` returns it to 0.

**Finding 3c — torch's caching host allocator rounds every pinned request up
to the next power of two.** The 9.93 GB store (6.82 GB of decoder tensors +
3.11 GB of vocabulary shards) therefore asked the driver for roughly 17 GB of
page-locked memory against ~15.5 GB free, which is what "could not page-lock
the base" meant. It also explains the table in (a): 9 GB rounds to 16 GiB and is
refused, 8 GB is 2^33 and is granted, five 1.99 GB chunks round to 5 x 2 GiB
and fit. And it explains the reference box's "7.12 / 7.65 GB page-locked
ceiling" in the earlier records: a 5.55 GB bf16 3B store pinned per tensor was
asking for ~10 GB of 16.9 GB.

## 4. The fixes, one at a time, each verified on the reproduction

**Fix A — drain the stale error before building the pageable store**
(`eb923a3f`). `drain_stale_cuda_error` launches one trivial kernel to let its
check consume the error and a second to prove the context is healthy; a
second failure or any other error propagates. `release_cached_pinned_memory`
hands back the blocks the abandoned attempt left in the host cache.
`recover_from_failed_page_lock` runs both between the failed pinned constructor
and the pageable one, on the RAM tier and on the disk tier's staging.

Same probe, same shape, store still pageable:

```
store         9.932 GB pageable tier ram; ...; build 34.6 s
after build   allocated 1.896 GB  reserved 1.898 GB  free 5.414 GB
  [before step             ] alloc 1.896 GB  peak 1.896 GB  reserved 1.898 GB
  [embed fwd out           ] alloc 1.900 GB  peak 1.900 GB
  [L0 fwd out              ] alloc 1.938 GB  peak 2.105 GB
  [L47 fwd out             ] alloc 2.122 GB  peak 2.290 GB
  [lm_head fwd out         ] alloc 2.247 GB  peak 2.290 GB
  [after forward (loss)    ] alloc 2.477 GB  peak 2.714 GB  reserved 2.816 GB
  [lm_head bwd start       ] alloc 2.360 GB  peak 2.944 GB  reserved 3.051 GB
  [L0 bwd start            ] alloc 2.137 GB  peak 2.944 GB  reserved 3.242 GB
  [after backward          ] alloc 2.130 GB  peak 2.944 GB  reserved 3.244 GB
RESULT ok: peak allocated 2.944 GB  peak reserved 3.244 GB  alloc retries 0  ooms 0
```
JSON: `issue901_14b_seq384_pageable_after_fixA.json`.

**Finding 4a — the step fits with 4.4 GB to spare, and the pre-flight's
3.39 GB was an over-prediction of 15%, the direction it is fitted for.** The
peak lands at the start of the backward through `lm_head` (the fp32 logits
gradient beside the bf16 logits), exactly where the formula's logits term says
it should.

**Fix B — pin the store in power-of-two arenas** (`6a3cfd1e`).
`plan_pinned_arenas` packs the tensors into 256 MiB arenas (widened to fit the
largest tensor, each trimmed to the power of two above what it holds), and
`RamSource` carves every tensor as a 256-byte-aligned view. Same probe:

```
store         9.932 GB pinned tier ram; ...; build 10.2 s
RESULT ok: peak allocated 2.944 GB  peak reserved 3.244 GB  alloc retries 0  ooms 0
stats.pinned_bytes = 10 737 418 240
```
JSON: `issue901_14b_seq384_pinned_arenas.json`.

**Finding 4b — the same store now page-locks as exactly 10 GiB (1.081x its
bytes) where per-tensor pinning asked for ~17 GB, so the fallback never
fires; the build drops from 34.6 s (a failed pin, a recovery and a pageable
copy) to 10.2 s.** Peak VRAM is unchanged to the byte, as it must be — the
change is host-side. The plan's arithmetic bound on this exact store is
pinned by `tests/test_issue901_pinned_arenas.py` at <= 1.10x.

**Revised 2026-09-15 13:47, after three reviews changed the packer's rule.**
The run above used a fixed 256 MiB arena with an exception for oversized
tensors. Review found that rule packed a *bf16* 14B store at 1.46x (two
141.6 MB projections cannot share a 256 MiB arena), and a first replacement —
capacity from the tensor that opens each arena — failed the same store the
other way (an arena opened by a 52 MB projection stayed at 256 MiB). The
shipped rule is the power of two above four times the store's largest tensor,
floored at 256 MiB and capped at 2 GiB. On THIS store the largest tensor is
the 1.557 GB vocabulary matrix, so every arena is 2 GiB: 5 arenas, 10 GiB, by
arithmetic — the same figure, reached differently. Re-measured with the final
code (`issue901_14b_seq384_pinned_arenas_final.json`, baseline free physical
19.04 GB, 5 Python processes): store 9.932 GB pinned, **10.737 GB
page-locked**, peak 2.944 GB allocated / 3.244 GB reserved, 0 retries — the
row above is unchanged to the byte, build 15.4 s (10.2 s before; the box was
not quiet). The bf16 case is arithmetic only, pinned at <= 1.10x by the test
file; no bf16 14B store was built here.

**Fix C — the probe calls an out-of-memory `AcceleratorError` an OOM**
(`b602bb15`). Not exercised by the reproduction after Fix A (nothing raises
any more); it is the #649 shape the reporter's probe output showed, and it only
changes which of two refusals the operator reads.

## 5. End to end, through the real `soup train`

The reporter's `soup.yaml` verbatim except for the base path, 80 chat rows
(72/8 after `val_split: 0.1`), `stream_vram_probe: true` as in their second
run, the Qwen2.5 tokenizer copied beside the synthetic weights, run 13:03-13:06:

```
│   base store   9.93 GB across 48 layers (pinned)
│   VRAM buffers 2 x 142 MB + 1 x 1557 MB large-layer slot = 1841 MB
│   peak VRAM    ~3.39 GB at batch 1 x seq 384 (logits 0.82 GB)
│   free VRAM    7.32 GB
measured peak 2.94 GB (3.24 GB reserved) in 4.61 s at batch 1 x seq 384; predicted 3.39 GB
Layer streaming ready: 48 layers, 9.93 GB pinned RAM store (10.74 GB page-locked),
  2 x 142 MB decoder buffers + 1 x 1557 MB large-layer slot
LoRA applied: 12,582,912 trainable / 14,770,033,664 total (0.09%)
Training started!
100%|██████████| 18/18 [01:56<00:00,  6.49s/it]
{'train_runtime': '116.8', 'train_samples_per_second': '0.616', 'train_steps_per_second': '0.154', ...}
│ GPU peak: 2.7/7.9 GB
```
exit 0, adapter written to `output_qwen14b_streamed`. Log:
`issue901_e2e_soup_train.log`.

**Finding 5 — the configuration the issue is about trains to completion on
an 8 GB card with 32 GB of RAM, through `SFTTrainer.__init__` and 18 optimizer
steps, with the store page-locked.** The loss (~13, random weights) means
nothing and is not quoted as anything. The first step took 28.5 s and the
steady state ~4.5 s per optimizer step of four rows; that is a functional
run on a synthetic shape, not a throughput figure, and none is claimed.

A fixture defect surfaced on the way: the generator wrote
`"transformers_version": "synthetic"` into `config.json`, which `AutoConfig`
tolerates and `AutoTokenizer` refuses (`InvalidVersion`). The generator now
writes the installed version; the `_soup_synthetic` marker stays.

## 6. Tests and gates

- `tests/test_issue901_failed_page_lock_recovery.py` (15 tests, 7 of them on
  real hardware, including a characterisation of the upstream behaviour that
  will fail the day torch clears the error itself) and
  `tests/test_issue901_pinned_arenas.py` (14, 16 with the parametrised
  cases): green here, CUDA tests included.
- The streaming regression set (`test_v07200/02/03/04`, `test_issue623`,
  `test_issue971_*`, `test_issue349`, `test_issue385`): **755 passed, 10
  skipped**, and the four `TestFloat16StreamingIsBitExact` ids — `[none]` and
  `[nf4]`, fp16 and bf16 — pass explicitly with the arena store. The pinned
  store is bit-identical to the shards by a direct `torch.equal` test as well.

## 7. What was NOT measured

- **A real Qwen2.5-14B-Instruct.** Shapes only; no quality, no loss curve, no
  throughput. The 5070's step times above are not comparable to anything.
- **The reporter's card.** An RTX 3070 is Ampere; the stale-error mechanism is
  a CUDA-runtime property and the rounding is a torch host-allocator property,
  neither card-specific, but the reporter has offered to test a fix and the
  record says so rather than assuming.
- **Linux.** Every number here is Windows/WDDM. The characterisation test in
  the test file will say whether the stale error exists there.
- **The disk tier's staging**, which still pins per tensor and pays the same
  1.7-1.9x; the planner's `staging_bytes_for` budgets requested bytes, so its
  refusal threshold under-counts by that factor. Left for the disk-tier work.
- **The pin ceiling as a function of RAM.** Chunked pinning reached 9.93 GB
  and per-tensor pinning reached 14.1 GB of private commit before this box
  said no; the actual limit was not sought.
- **Baseline commit-charge stamps for runs 2, 3 and the end-to-end run** were
  not taken; the run-1 stamp is above, and the pin experiments started at
  21.0-21.7 GB free physical. Named rather than tidied away.
