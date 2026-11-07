# Probe: one LoRA training step on a Kimi K2-shaped store, RTX 5070 Laptop 8 GB

Status: **COMPLETES**, both runs. Run 1 (the model's own routing) is real but not a K2 step
time (section 4.1). Run 2 (forced uniform routing, addendum rule in section 2.1 committed in
1fe403a18 before it ran) read every expert: **one full LoRA step of a 1.026 T-parameter
K2-shaped model took 868 s cold and 1,109 s next on the 8 GB laptop.** Random weights, shape only.

**Every number in this record is "random weights, shape only".** The store is
synthetic: the parameter count and tensor shapes of `moonshotai/Kimi-K2-Instruct`
(~1.03 T parameters, 61 layers, 384 routed experts of 3 x 7168 x 2048 per MoE
layer), random NF4 bytes. It says nothing about the quality, convergence or
throughput of training the real model. Prior art for streaming layers through a
small device memory is L2L (Pudipeddi et al. 2020, "Training Large Neural
Networks with Constant Memory using a New Execution Algorithm"); this record
does not use L2L's layer-major micro-batching.

## 1. Question

With layer streaming plus expert-granularity streaming (`expert_layout="per_expert"`,
`benchmarks/gate-expert-streaming.md`), does ONE full LoRA training step (forward,
backward, AdamW on attention adapters) of a K2-shaped model complete on this
laptop, and what does it cost in time, VRAM, host RAM and bytes read?

## 2. Decision rule (written before any run)

Store: `benchmarks/harness/synth_moe_store.py --shape k2 --out D:/soup-bench/moe-k2-synth`.
Step: `benchmarks/harness/moe_stream_step.py --seq 512 --batch 1 --steps 2 --lora-r 8`
(step 0 is cold, step 1 is reported separately; neither is averaged into the other).

- **COMPLETES** iff step 0 and step 1 both finish with a finite loss and finite
  adapter gradients, the peak allocated VRAM is at most 7,500 MiB (the card has
  8,151 MiB; the margin is for the CUDA context, and Windows would otherwise spill
  to shared memory silently), and the process's peak working set stays below
  20 GB (the box has 31.7 GB; the rule keeps the owner's 10 GB-free floor).
- **DOES NOT COMPLETE** otherwise; the record then states which limit, at which
  layer, with the error text.
- Reported, no threshold: seconds per step, tokens per second, expert bytes read
  per step against the 3 x routed-expert bytes the design predicts (forward,
  checkpoint recompute, backward), layer loads.

Estimate before the run (an estimate, not a measurement): ~570 GB of experts read
3 times per step, ~1.7 TB. The async disk tier reads this drive at 4.09-4.14 GB/s
(`probe-rtx5070-what-bounds-streaming.md` §21); the expert reader is synchronous
(one expert, then its compute), so expect it slower: 7 minutes at 4.1 GB/s, about
24 at 1.2 GB/s.

## 2.1 Addendum rule for run 2 (written after run 1, before run 2)

Run 1 routed 512 random tokens to only ~10 of the 384 experts per layer and pass
(section 4.1). A trained model routes a 512-token step to 96.7-100% of its experts
(`probe-moe-expert-coverage.md`), so run 1 does not measure what a K2 step reads. Run 2
repeats the same command with `--route uniform`: every router sends token t to experts
8t .. 8t+7 (mod 384), so each pass routes to all 384 experts of every MoE layer, the
coverage a trained model nearly has. The router weights still stream; only its choice
is replaced. Same thresholds as section 2: **COMPLETES** iff both steps finish with a
finite loss and finite adapter gradients, peak allocated VRAM <= 7,500 MiB and peak
working set < 20 GB. Reported, no threshold: seconds per step, expert reads (the design
predicts 60 x 384 x 3 = 69,120 per step) and expert bytes per step. Label of every
number: "random weights, shape only; routing forced uniform".

## 3. Box state

Run 2: 2026-10-14 19:21:23Z to 19:54:38Z; AC power on; 12.4 GB RAM free and GPU 0 MiB in
use before; the same hold, extended on the board and by message; harness commit 1fe403a18,
same command plus `--route uniform`.

Run 1: 2026-10-14 19:17:00Z to 19:18:58Z; AC power on; 11.3 GB RAM free before;
GPU 0 MiB in use before; D: 214 GB free after the 533 GB store was written; a
measurement hold on the board, every other session confirmed idle. Harness commit
0fa52150e. Run with `--tier disk --no-pin`: the RAM tier would pin the ~5 GB of
dense layer files plus the two 2.35 GB vocabulary matrices on a box with ~11 GB
free, and pinned disk-tier staging rounds to an 8 GiB arena (seen in a build-only
check), so staging stays pageable. The rule did not fix the tier; this is the
configuration that keeps the owner's free-RAM floor.

Store: `synth_moe_store.py --shape k2`, 1,026,408,232,448 parameters,
533,242,299,230 bytes, written in 588.6 s (`synth.json`), NF4 with double
quantisation, `expert_layout: per_expert`.

## 4. Results

### 4.0 Run 1: the model's own routing (random weights, shape only)

Build 4.22 s; 30,513,664 trainable parameters (LoRA r 8 on `q_a_proj`, `q_b_proj`,
`kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` of all 61 layers). Raw JSON:
`benchmarks/results/probe-rtx5070/giant-moe/k2-step.json`.

| step | seconds | tokens/s | loss | grads finite | peak alloc VRAM | peak reserved | peak working set | expert reads | expert bytes | layer loads |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (cold) | 53.61 | 9.55 | 13.478 | 610 / 610 | 4,342 MiB | 5,052 MiB | 7.67 GB | 1,890 | 42.94 GB | 120 |
| 1 | 45.21 | 11.33 | 13.473 | 610 / 610 | 4,723 MiB | 5,068 MiB | 7.68 GB | 1,752 | 39.80 GB | 118 |

Against section 2: both steps finish, loss and all 610 adapter gradients finite, peak
VRAM 4,723 MiB <= 7,500, peak working set 7.68 GB < 20 GB: **run 1 COMPLETES.**

### 4.1 Why run 1's 45 s is not a K2 step time

1,752 expert reads in a step is 1,752 / (60 MoE layers x 3 passes) = ~9.7 distinct
experts per layer and pass out of 384: the 512 random tokens collapsed onto a few
experts (with random attention weights every token attends to an average of the same
prefix, so the hidden states, and with them the routing, nearly coincide). The design
predicted ~1.5-1.7 TB of expert reads per step for a model routing like a trained one;
run 1 read 40 GB. Its time is real for what it did, and says nothing about a real K2
step. Run 2 (section 2.1) measures that case.

### 4.2 Run 2: forced uniform routing (random weights, shape only; routing forced uniform)

Raw JSON: `benchmarks/results/probe-rtx5070/giant-moe/k2-step-uniform.json`.

| step | seconds | tokens/s | loss | grads finite | peak alloc VRAM | peak reserved | peak working set | expert reads | expert bytes | layer loads |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (cold) | 867.95 | 0.590 | 13.477 | 610 / 610 | 4,342 MiB | 5,052 MiB | 7.75 GB | 69,120 | 1,570.34 GB | 120 |
| 1 | 1,108.77 | 0.462 | 13.470 | 610 / 610 | 4,723 MiB | 5,170 MiB | 7.89 GB | 69,120 | 1,570.34 GB | 118 |

- Expert reads are exactly the predicted 60 x 384 x 3 = 69,120 per step.
- Expert bytes per step, 1.57 TB, are 2.95 x the 533 GB store: each expert is read in forward, in the checkpoint's recompute, and in its own backward.
- Seconds per expert byte: 1.81 GB/s of expert reads in step 0 and 1.42 GB/s in step 1, compute included.
- Step 1 was slower than step 0. Nothing in the run says why; it is reported as measured.

Against section 2.1, both steps finish:
- loss and all 610 adapter gradients finite;
- peak VRAM 4,723 MiB <= 7,500;
- peak working set 7.89 GB < 20 GB.

**Run 2 COMPLETES.**

## 5. Verdict

**COMPLETES.** On an RTX 5070 Laptop (8 GB) with 31.7 GB of RAM, layer streaming plus expert-granularity streaming runs a full LoRA training step (forward, backward, AdamW) of a Kimi K2-shaped model: 1,026,408,232,448 parameters, random weights, 533 GB NF4 on NVMe. Peak VRAM is 4,723 MiB.

**A step takes 14.5-18.5 minutes (867.95 s and 1,108.77 s) when every expert is routed (run 2); a trained model routes nearly every expert (#992).** That is a capacity demonstration, not a training regime: at 512 tokens per step it is ~0.5 tokens/s. The time is the read volume, 1.57 TB per step, three passes over the experts. L2L (Pudipeddi et al. 2020) is the published way to amortise those reads across micro-batches, and this run does not use it.

**Not shown:**
- convergence or quality: the weights are random;
- a real K2 checkpoint: it is FP8, which the sharder refuses;
- more than two steps;
- whether step 1's slowdown persists.
