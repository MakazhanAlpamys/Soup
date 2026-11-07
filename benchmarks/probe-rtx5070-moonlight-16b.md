# Probe: LoRA on Moonlight-16B-A3B (DeepSeek-V3 architecture) with streaming, RTX 5070 Laptop 8 GB

Status: **both arms COMPLETE** (runs 2026-10-14 18:53Z-18:59Z). The rule in section 2 was committed in d1396a508 before any run. The product-path check COMPLETES too (section 4.1, 2026-10-15).

`moonshotai/Moonlight-16B-A3B`: `model_type: deepseek_v3` (checked in its
`config.json` before anything was built), 27 layers, 64 routed experts + 2 shared
per MoE layer, top-6, 15.96 B parameters, 31.92 GB of bf16 safetensors. It is
the same decoder Kimi K2 uses, at 1/64 of the size. It does not fit the 8 GB
card resident, at bf16 or at NF4 (~8.5 GB of weights alone).

## 1. Question

Does a LoRA training step of Moonlight-16B run on this laptop under layer
streaming (#1914, routed experts fused and at bf16) and under expert streaming
(the per-expert NF4 layout)? What are its seconds per step, tokens per second,
peak VRAM and peak host memory in each mode?

## 2. Decision rule (written before any run)

Harness: `benchmarks/harness/moe_stream_step.py`, real weights, random token ids,
`--seq 512 --batch 1 --steps 5 --lora-r 8`, LoRA on the attention projections
(Moonlight has `q_lora_rank: null`, so `q_proj`, `kv_a_proj_with_mqa`,
`kv_b_proj`, `o_proj`). Two stores, sharded once beforehand (the sharding time is
reported, not judged):

- arm **L**: `--expert-layout fused`, `--tier disk`. Attention, shared and dense MLP are NF4; the routed experts stay bf16 in the layer files.
- arm **E**: `--expert-layout per_expert`, `--tier ram`. Every expert matrix is NF4, read one expert at a time.

Order: L, E, then E, L (two rounds, interleaved). Step 0 of every run is warm-up
and is reported but not used.

- An arm **COMPLETES** iff in both rounds:
  - all 5 steps finish with a finite loss and finite adapter gradients;
  - peak allocated VRAM is at most 7,500 MiB (the card has 8,151 MiB; above that, Windows spills to shared memory silently);
  - the process's peak working set stays below 20 GB (the owner's 10 GB-free floor on a 31.7 GB box).
- Reported, with no threshold:
  - the median seconds of steps 1-4 per run, and tokens/s = 512 / that;
  - expert bytes read per step;
  - layer loads;
  - store bytes on disk;
  - shard seconds.

  Speed is not the goal: no number is promised before the run.
- A product-path check follows (outside this rule's numbers): `soup train` with a
  `soup.yaml` on this base, one epoch over a small alpaca file. It **COMPLETES**
  iff it finishes with a finite loss and writes an adapter. Wall time and the
  first/last logged loss are reported.

## 3. Box state

- AC power on. Free RAM before each run was 12.6, 13.9, 13.6 and 13.8 GB; GPU ~114 MiB in use (the desktop).
- A measurement hold was on the board for 18:50Z-19:00Z, and the other sessions confirmed no pytest. D: had 711 GB free.
- Software: torch 2.14.0+cu130, transformers 5.17.0, bitsandbytes 0.50.2.
- Harness `benchmarks/harness/moe_stream_step.py` at 9cbe1cd20 (merged into this branch as bd97a5d4b).
- **Two earlier runs (18:49Z, 18:51Z) were discarded.** The harness then computed `grads_finite` after `optimizer.zero_grad(set_to_none=True)`, so it was always true (a review found it). The fix moved the check to right after `backward`; all four runs below are after the fix.
- Stores, sharded once beforehand from the bf16 checkpoint on D: (both NF4 with double quantisation; seconds not judged):
  - fused: 30,598,520,590 bytes, 61.7 s;
  - per_expert: 9,235,807,046 bytes, 51.3 s.
- The fused store had just been written and is 29 GB on a 31.7 GB box, so part of it was very likely in the page cache: arm L's reads are NOT cold-disk reads. Arm E's expert reads are direct I/O, bypassing the cache.

## 4. Results

Real weights, random token ids, seq 512, batch 1, LoRA r 8 on `q_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` / `o_proj` (3,552,768 trainable parameters). Every step of every run had a finite loss and 216 of 216 adapter gradients present and finite. Raw JSON: `benchmarks/results/probe-rtx5070/moonlight/`.

| run | arm | steps 1-4 seconds | median | tokens/s | peak alloc VRAM | peak reserved | peak working set | expert reads per step | expert GB per step |
|---|---|---|---|---|---|---|---|---|---|
| r1 | L | 9.81, 9.80, 9.84, 9.83 | 9.82 | 52.1 | 3,993 MiB | 4,264 MiB | 6.76 GB | 0 | 0 |
| r1 | E | 20.30, 18.08, 16.95, 17.99 | 18.04 | 28.4 | 1,882 MiB | 2,138 MiB | 4.60 GB | 4,089-4,287 | 18.25-19.13 |
| r2 | E | 16.89, 18.41, 18.03, 15.92 | 17.46 | 29.3 | 1,882 MiB | 2,138 MiB | 4.60 GB | 4,086-4,296 | 18.23-19.17 |
| r2 | L | 8.84, 8.91, 8.95, 17.36 | 8.93 | 57.3 | 3,993 MiB | 4,264 MiB | 6.76 GB | 0 | 0 |

- Step 0 (warm-up, not used): L 12.95 s and 11.01 s; E 20.22 s and 18.28 s.
- Arm L's last step in r2 (17.36 s) is an outlier at twice its neighbours. It is reported, not dropped; the median absorbs it.
- An expert read is one expert fetched once.
- 26 MoE layers x 64 experts x 3 passes (forward, checkpoint recompute, backward) caps a step at 4,992 reads. The 4,086-4,296 measured means random ids routed to ~82-86% of the experts per pass.

### 4.1 Product path: `soup train`

2026-10-15 08:36:52Z to 08:45:39Z, AC power, 14 GB RAM free before, GPU idle, the other
sessions told beforehand. Code at 2846155470 (branch `feat/moe-moonlight`). Command:
`soup train --config soup.yaml --trust-remote-code --yes`, run from `D:/soup-bench` with
`SOUP_LAYER_STREAM_CACHE_DIR=D:/soup-bench/soup-cache`.

The config:
- base: the local Moonlight-16B-A3B checkpoint;
- `task: sft` on `examples/data/alpaca_tiny.jsonl` (9 train rows, 1 validation row);
- `max_length 512`, `epochs 3`, `batch_size 1`, `quantization: 4bit`;
- `stream_layers: true`, `stream_source: auto`;
- LoRA r 8 on `q_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`.

`--trust-remote-code` reaches only the tokenizer, which Moonlight ships as remote code; the
model is transformers' own class (this branch's fix).

What Soup reported:
- **Tier and VRAM plan.** `stream_source: auto` fell back to the NVMe disk tier, because the 30.60 GB store does not fit in RAM. The VRAM plan was 2 x 1,159 MB layer buffers plus a 671 MB vocabulary slot, with a predicted peak of ~4.32 GB; the reported peak was 3.1 of 7.9 GB.
- **Steps.** 27 steps in 422.7 s, ~15.7 s per step, with short alpaca rows.
- **Loss.** Training loss 4.922 at step 10, then 3.124 at step 20; mean train loss 3.866; validation loss after epoch 3 was 3.472.
- **Adapter.** Written: `adapter_model.safetensors`, 7,134,912 bytes, plus `checkpoint-27`.

**COMPLETES:** finite loss, adapter written. A 9-row dataset says nothing about quality; this
shows that the CLI path, from config to adapter, runs on this base on an 8 GB card.

### 4.2 Product path with `training.stream_expert_layout: per_expert` (#2029)

2026-10-19 12:32:19Z to 12:52:59Z, AC power, 6 GB RAM free at start. The other sessions were told
beforehand; free RAM dipped near zero during the run because a foreign WSL VM held ~6.5 GB.

Setup: the same config and command as section 4.1, plus `stream_expert_layout: per_expert`, with
`SOUP_LAYER_STREAM_CACHE_DIR=D:/soup-bench/soup-cache-experts`. Code: the #2029 branch at
c40776c1b plus its uncommitted changes, which are what 76bab8e7 committed.

What Soup reported:
- **Store.** The per-expert NF4 cache came to 8.7 GB, so `auto` chose the RAM tier: 1.81 GB pinned (the dense parts); the experts are read from the cache volume.
- **Pre-flight.** It printed `experts 0.044 GB for the routed expert in flight`.
- **Steps.** 27 steps in 1,128 s, ~41.8 s per step, against ~15.7 s in section 4.1.
- **Loss.** Training loss 4.775 at step 10, then 3.165 at step 20; mean 3.827; validation loss 3.611 after epoch 3.
- **GPU peak.** 1.0 of 7.9 GB, against 3.1 GB in section 4.1.
- **Adapter.** Written: `adapter_model.safetensors`, 7,134,912 bytes.

**COMPLETES:** finite loss, adapter written. Under 4bit the routed experts are NF4 here and bf16
in section 4.1, so the two runs' losses come from numerically different bases.

## 5. Verdict

**Both arms COMPLETE.** LoRA training steps of Moonlight-16B, a 16 B-parameter DeepSeek-V3-architecture MoE that does not fit this 8 GB card resident, run on the laptop:

- **layer streaming:** 3.9 GB peak VRAM, ~52-57 tokens/s, store partly in the page cache;
- **expert streaming:** 1.9 GB peak VRAM, ~28-29 tokens/s, direct-I/O expert reads.

**Expert streaming halves the VRAM and is about half as fast here.** It reads each routed expert three times per step, one at a time and synchronously (~19 GB per step). Layer streaming reads whole layers through the async reader from a partly cached store.

**The page-cache caveat means arm L's speed is not a cold-disk number.** These are tokens/s of random-id steps, with no claim about convergence.
