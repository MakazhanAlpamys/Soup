# Gate record - layer-major micro-batching on an 8 GB RTX 3070

**Verdict (2026-10-19) - Positive.** Each of the four layer-major gates passed. The longer ordinary-batch control established a batch bound of 4 and showed lower peak allocated VRAM for layer-major micro-batching. A speed advantage over ordinary batch 4 remains unestablished.

Measured by @Artix6034.

## 1. Setup

- RTX 3070 8192 MiB, PCIe Gen 3 x16, driver 580.159.04, CUDA 13.0, Core i7-10700K, and a cgroup RAM limit of 32,107,397,120 bytes. The Linux container used a 240 GiB overlay volume. Two NVMe devices and one HDD were visible, with a single accessible storage mount for the experiment.
- Python 3.12.13, torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0.
- Correctness used real `unsloth/mistral-7b-instruct-v0.3` at `60b99dcbc7f853b7cd18472a607e8d045b96f03e` and synthetic Llama-shaped 68.99B weights. Timing and memory used the synthetic model, with 80 layers, hidden size 8192, and vocabulary 32768. Both used double-quantized NF4 with BF16 compute, sequence 512, LoRA rank 8 on q/k/v/o with zero dropout, two buffers and read-ahead of two. The real model used RAM weights and the synthetic NF4 cache occupied 36,388,401,328 bytes on disk.
- `training.l2l` is pending. These runs exercised a prototype harness on runtime base `3da204fadbd278c01ad5a4b23906b443d15e9dd1` with a separate measured source manifest.

## 2. Method and results

The stand required repeatable gradient accumulation (GA), matching per-microbatch losses and nonzero LoRA gradients, a 16-microbatch rate of at least 80% of the read-and-copy-free reference and at least 1.3x the single-microbatch rate, and no more than 0.5 GB of additional allocated VRAM. It used two interleaved rounds with one warm-up and three measured steps. Correctness used the predeclared deterministic fallback. The separate longer ordinary-batch control used fresh processes in ascending and descending batch order and two warm-up and 24 measured steps for batches 1–4. The spread limit stayed at 5%. The batch-5 OOM probes used the shorter profile.

| Test | Result |
|---|---|
| Losses and nonzero LoRA gradients | GA repeat and layer-major comparison equal on both fixtures |
| Layer-major window of 16 versus read-and-copy-free reference | 74.839 versus 75.007 tok/s |
| Layer-major window of 16 versus window of 1 | 74.839 versus 21.646 tok/s, 3.457x |
| Peak allocated VRAM at windows of 16 and 1 | 4.095 versus 3.968 GB, +0.127 GB |
| Separate pinned/spill comparison at window of 16 | Both rounds valid, ratio 0.99943 |

| Ordinary batch | Round 1 tok/s | Round 2 tok/s | Mean tok/s | Spread | Peak allocated GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.181 | 20.112 | 20.146 | 0.34% | 4.459 |
| 2 | 40.356 | 40.760 | 40.557 | 1.00% | 5.426 |
| 3 | 61.406 | 60.301 | 60.849 | 1.82% | 6.393 |
| 4 | 75.069 | 75.031 | 75.050 | 0.05% | 7.360 |
| 5 | CUDA allocator OOM | CUDA allocator OOM | — | — | — |

## 3. Commands

The harness, saved runner and complete per-arm commands live at the evidence commit. The reproduction guide provides the setup for creating the local long-control runner from the archived template and binding it to the new parent session.

```bash
bash benchmarks/a0-bootstrap.sh
.venv/bin/python benchmarks/harness/l2l_prepare.py \
  --out "$PWD/.soup/preparation-01" --power-source server
.venv/bin/python benchmarks/harness/l2l_session.py \
  --f1 "$PWD/.soup/models/mistral-7b" --f2 "$PWD/.soup/models/synth-70b-v32k" \
  --f1-shards "$PWD/.soup/f1-nf4" --f2-shards "$PWD/.soup/f2-nf4" \
  --spill-dir "$PWD/.soup/spill" --out "$PWD/.soup/results/a0-attempt-01" \
  --power-source server --batch-max 4 --max-hours 6 --execute
.venv/bin/python .soup/a0-full04-replay-01-runner.py
```

## 4. Limits and earlier runs

The long control supplies the batch bound and leaves the original short ordinary-batch block incomplete. The original block had invalid pairs at batches 1, 2 and 3. An idle-first repeat still invalidated 1 and 3, and a mixed-duration repeat invalidated 2 and 4. Each block retains its own verdict. The older October 7 laptop run passed equality and the read-and-copy comparison, with timing-growth and VRAM-growth gates unscored after a noisy control.

Power was operator-declared and telemetry visibility was limited to exposed container interfaces. Storage-path variation remains unexplained. Throughput and capacity describe the synthetic dense shape. Model quality and routed experts were outside the measurement.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
