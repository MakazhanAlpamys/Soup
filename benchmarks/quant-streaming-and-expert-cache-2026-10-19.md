# Measurement record - early quantization, streaming and cache evidence

Measured by @kok-o. The archived GPU screens ran on 2026-10-19. The expert
cache measurements and follow-up preparation are dated 2026-10-25 and 2026-10-26.

## 1. Questions and setup

How does nested NF4 affect pretrained loss, and does the scoped streaming
kernel preserve updates while reducing memory? Can an immutable expert
hotset avoid the cost of replacing cached experts during two-read streaming?

- The resident quality screen ran on an RTX 3090 (24,576 MiB), Linux,
  driver 565.57.01, PyTorch 2.11.0+cu126, Transformers 5.17.0,
  bitsandbytes 0.50.2, Accelerate 1.13.0 and NumPy 2.4.3. It compared
  original BF16 with nested NF4 and BF16 compute on
  `mistralai/Mistral-7B-v0.3` at `caa1feb0e54d415e2df31207e5f4e273e33509b1`.
- The synthetic Llama and pretrained Mistral streaming screens used
  separately rented RTX 3090 Ubuntu VMs at a measured 1200 MHz. Both used
  nested NF4 with BF16 compute, two pageable RAM buffers and query/value
  LoRA. The compact audit receipts carry the verdicts. CPU, total RAM,
  exact OS release and the VM driver/CUDA and package pins were unavailable
  in those receipts.
- The cache allocation control ran CPU-only on Windows with PyTorch 2.14.0,
  one Torch thread and synthetic payloads shaped like the packed NF4
  Moonlight experts. The GPU follow-up used an RTX 4060 Ti (8188 MiB),
  a 29.756 GiB workload RAM limit, swap disabled, Linux kernel
  `6.8.0-147-generic`, driver 595.71.05 and PyTorch 2.8.0+cu128. It used
  Python 3.12.3, Transformers 5.16.1, bitsandbytes 0.50.2, PEFT 0.20.0,
  safetensors 0.8.0 and TRL 0.29.1. The real Moonlight-16B-A3B recipe used a
  frozen NF4 base, BF16 compute, FP32 attention LoRA, batch 1, sequence 512
  and accumulation 2. The compact results omit the CPU model and distro.

## 2. Method and commands

The quality screen scored the same 16,352 next-token targets in 32 fixed
WikiText-2 test windows of 512 tokens, without LoRA. The streaming screens
compared native and scoped WMMA dispatch after three exact optimizer
updates. The synthetic fixture used five fresh-process ABBA blocks with
20 warm-up and 200 timed updates per arm and a clock guard. The real
Mistral screen used one ABBA block with 5 warm-up and 20 timed updates per
arm, after its numerical check passed.

The evidence commit retains the original archive, audit summaries and
replay scripts. The archived `run.sh` scripts in the streaming run packets
contain the exact GPU launches. These are the published quality launcher and
the separate CPU replay commands, run from their respective evidence folders.

```bash
bash run.sh
python replay.py --repo /path/to/soup
```

For the cache comparison, the CPU replay counted cache operations over a
fixed captured route using the same staging and device filter. The separate
GPU test checked full numerical captures before four balanced rounds of
fresh processes, with two warm-up and six measured updates per process.
The cache harnesses live at the evidence revisions listed below. The
profiling commands are the prepared diagnostic, separate from the unprofiled
timings.
Executed cache-policy source and harness
Profiling harness

```bash
python benchmarks/harness/expert_cache_policy_bench.py \
  --spec "$SPEC" --out "$OUT" --rounds 4 --large-device-cache-gib 1.5
python benchmarks/harness/expert_cache_profile.py \
  --spec /results/prepared/spec.json --arm two_read --out /results/profiles/two_read
python benchmarks/harness/expert_cache_profile.py \
  --spec /results/prepared/spec.json --arm two_read_cache --out /results/profiles/two_read_cache
```

## 3. Results

| Archived GPU screen | Reference | Candidate | Decision |
|---|---:|---:|---|
| Resident Mistral BF16 versus nested NF4, mean NLL | 1.8874408620346017 | 1.9187500773875479 | Descriptive increase of 0.031309215352946174, no equivalence threshold |
| Synthetic streamed Llama, median step ms | 51.567947749958876 | 83.51197350000916 | Five accepted blocks on the VM, candidate slower |
| Synthetic streamed Llama, peak allocated bytes | 371988992 | 207362048 | Lower candidate peak on this fixture |
| Pretrained streamed Mistral, median step ms | 1726.373499000033 | 1730.408540250039 | Exact short-run parity, descriptive timing only |
| Pretrained streamed Mistral, peak allocated bytes | 843303424 | 776718848 | Lower candidate peak in one descriptive block |

Resident quality audit
Synthetic streaming audit
Pretrained streaming audit

| CPU cache policy | Modeled file reads | Cloned payload | Cache-operation wall time |
|---|---:|---:|---:|
| LRU | 22,089 | 91.74 GiB | 7.586 s |
| Immutable first-fit | 22,090 | 2.00 GiB | 0.544 s |

Source for the CPU rows. Cache allocation control

| GPU cache policy on top of two-read | Median update, s | Median whole entry, s | Paired step-time reduction versus two-read |
|---|---:|---:|---:|
| No expert cache | 7.8848 | 75.290 | — |
| LRU RAM and 0.25 GiB GPU | 8.2921 | 90.555 | -5.111% |
| Immutable RAM and 0.25 GiB GPU | 7.0679 | 79.227 | 10.311% |
| Immutable RAM and 1.5 GiB GPU | 6.7791 | 76.847 | 14.026% |

The immutable policies passed the pre-set paired lower-bound gate and the
captured numerical trajectories matched bitwise. Setup kept every cached
eight-update entry slower than the uncached entry. GPU cache follow-up

The Kanana-2-30B-A3B preparation verified checkpoint headers against a
native meta skeleton and exercised tiny CPU NF4 and BF16 trajectories. Its
planned real recipe used nested NF4, BF16 compute and FP32 attention LoRA.
The preparation and the separate profiling harness established no real
30B GPU result or profiled throughput verdict. Kanana preparation

**Verdict — mixed. The scoped streaming kernel reduced peak allocation but was slower on the accepted synthetic VM cohort. Immutable expert admission improved steady update time on the fixed Moonlight workload while short entries remained slower.**

## 4. Limitations and superseded runs

- The accepted Ubuntu VM cohort replaced the rejected container runs for
  the synthetic fixture only. The earlier clock-spread failures retain
  their no-verdict status. A separate A100 follow-up stopped at the input
  gradient gate, so it produced no timing. The original archive preserves
  the superseded reports and raw evidence.
  Archive map
- The resident quality run followed a setup-only pip failure. The
  streamed Mistral run followed a symlink-shard preparation failure
  resolved by materializing the original cached bytes. The quality screen
  says nothing about learning or convergence. The short streaming window
  leaves long-run timing and downstream quality unverified.
- The earlier Moonlight GPU comparison found the combined LRU arm had a
  -6.110% paired reduction against two-read alone. The later balanced
  cache-policy comparison measures the immutable candidate and keeps the
  original LRU result as a separate cohort. The CPU times cover cache
  operations only. Immutable admission depends on visit order, and no
  physical SSD throughput or long-run quality was established.
  Earlier GPU cache comparison

Evidence: https://github.com/SoupTeam/soup-rnd/tree/33a68aac8c6fec6274866aee836850a4ebb473d4/followups
