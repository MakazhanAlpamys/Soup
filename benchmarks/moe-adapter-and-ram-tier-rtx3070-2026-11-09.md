# Moonlight adapter updates on an RTX 3070 and a Qwen RAM-tier check

**Verdict (2026-11-09): mixed. Moonlight passed the four-update disk-streaming gate. The Qwen header and planner check ruled out the requested 32 GiB RAM tier for the measured layout.**

Measured by @M1kado0.

## 1. Questions and setup

Can real Moonlight weights stream from a per-expert disk cache through four LoRA
updates and an adapter reload on one 8 GB GPU? Can the separate Qwen checkpoint
use the RAM tier with 32 GiB of physical memory?

- Moonlight used `moonshotai/Moonlight-16B-A3B` at
  `476b36a473d4467f94469414bef6cee75c9c8172`, BF16 compute and a per-expert NF4
  disk cache with double quantization disabled. The host had one RTX 3070
  (8,277,196,800 bytes visible to PyTorch), driver 615.71.09, AMD EPYC 7402 and
  270,105,587,712 bytes of RAM. The container used Ubuntu 24.04, host Linux
  6.8.0-138-generic, Python 3.12.3, PyTorch 2.14.1+cu130 / CUDA 13.0, Transformers
  5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0 and tiktoken 0.14.0.
- The Qwen check used `Qwen/Qwen3.5-35B-A3B` at
  `59d61f3ce65a6d9863b86d2e96597125219dc754`, a BF16/F32 checkpoint. It ran on a Mac
  with CPU only, PyTorch 2.14.1 and Transformers 5.20.2. The planner was given a
  34,359,738,368-byte input. The CPU model, physical RAM and OS version were unspecified.

## 2. Methods

Moonlight used `training.stream_expert_layout: per_expert`, `training.stream_source: disk`
and 4-bit quantization. The harness trained on four authored strings at 512 tokens each,
batch size 1, without padding or accumulation. Attention LoRA targeted `q_proj`,
`kv_a_proj_with_mqa`, `kv_b_proj` and `o_proj` with rank 8, alpha 16 and dropout 0. AdamW
used 1e-4, gradient norm clipping at 1.0 and seed 20261109. The judge required finite
gradients for every trainable tensor and a global pre-clipping norm above 1e-06 at each
step. It required a consistent digest chain, a relative adapter move above 1e-06 after
the first and final updates, and matching trained and reloaded parameter digests.

The Qwen check fetched the model config and safetensors headers using HTTP range requests,
then inspected the CPU skeleton and ran the placement planner. The inspected cache layout
kept packed routed experts in BF16 and quantized eligible dense weights to NF4. The planner
also evaluated a hypothetical layout with the routed experts in NF4. The 55% physical RAM
threshold for `training.stream_source: ram` was 18.90 GB in this calculation.

## 3. Results

| Moonlight step | Loss | Pre-clipping gradient norm | Finite gradients | Distance from initial adapter | Input tokens/s |
|---|---:|---:|---:|---:|---:|
| 0 | 4.6883 | 0.6791 | 216 of 216 | 0.1453 | 19.3 |
| 1 | 4.2917 | 0.7489 | 216 of 216 | 0.2463 | 41.7 |
| 2 | 3.4115 | 0.6092 | 216 of 216 | 0.3397 | 41.7 |
| 3 | 3.6246 | 0.6982 | 216 of 216 | 0.4235 | 42.7 |

| Moonlight measurement | Result |
|---|---:|
| Median throughput for steps 1–3 (input tokens/s) | 41.7 |
| Peak CUDA allocated / reserved (bytes) | 1,938,224,640 / 2,246,049,792 |
| Sampled `nvidia-smi` maximum (MiB) | 2379 |
| Process `VmHWM` at completion (bytes) | 4,463,886,336 |
| Source / cache (`du -sb`, bytes) | 31,924,237,216 / 9,947,958,701 |
| Download / cache preparation / four steps / reload (seconds) | 239 / 16.5 / 63.2 / 1 |
| Expert fetches by step | 4581 / 4575 / 4488 / 4485 |
| Adapter judge / run acceptance | PASS / PASS |

| Qwen header and planner result | Inspected layout | Hypothetical NF4 routed experts |
|---|---:|---:|
| Streamed store (GB) | 65.26 | 18.96 |
| Resident extras (GB) | 2.03 | 2.03 |
| Selected tier at 32 GiB with NVMe available | disk | disk |
| Estimated physical RAM required for RAM tier (GB) | 122.4 | 38.2 |

The Qwen header accounting totals 67,295,631,616 bytes for the inspected layout and
20,990,515,456 bytes for the hypothetical layout. Both estimates include resident extras.

The measurements and estimates above are retained in these files.

```text
benchmarks/evidence/b5-moonlight-20261109/attempt1/{run/journal.jsonl,acceptance-verdict.json}
benchmarks/evidence/b5-moonlight-20261109/attempt1/{du.txt,nvidia-smi-samples.csv,run.log}
benchmarks/evidence/b5-qwen35-20261109/{qwen35_plan.json,qwen35_tier.out}
```

## 4. Commands

The scripts, configs and data below live at the evidence commit. For Moonlight the
recorded `remote-run.sh` sets the environment, downloads the pinned snapshot and runs
the following training invocation. The second command replays the journal gate.

```bash
export PYTHONPATH=/root/b5/repo:/root/b5/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/b5/venv/bin/python benchmarks/harness/b5_real_moe.py \
  --config benchmarks/protocols/b5-moonlight-20261109/soup.yaml \
  --source /root/b5/source --cache /root/b5/cache --output /root/b5/results/attempt1/run \
  --data benchmarks/protocols/b5-moonlight-20261109/data.json \
  --model moonshotai/Moonlight-16B-A3B --revision 476b36a473d4467f94469414bef6cee75c9c8172 \
  --kind real --steps 4 --shard-threads 6
python benchmarks/harness/b5_acceptance.py \
  --journal benchmarks/evidence/b5-moonlight-20261109/attempt1/run/journal.jsonl
```

For the Qwen calculation, set `CONFIG_DIR` to the local config directory for the pinned
revision. The tier estimate accepts either the new plan or the retained plan JSON.

```bash
python -I benchmarks/evidence/b5-qwen35-20261109/qwen35_plan.py "$CONFIG_DIR" qwen35_plan.json
python -I benchmarks/evidence/b5-qwen35-20261109/qwen35_tier.py qwen35_plan.json
```

## 5. Limitations and superseded runs

- Moonlight uses one run on one card. The later-step median describes adjacent steps.
  The losses are from different strings, and convergence and general quality remain
  unmeasured. `--shard-threads 6` was accepted but unused by the historical harness.
- The reload check covers the adapter parameter digest. Reload logits and reload VRAM
  remain unmeasured. Training peaks are allocator counters. OS process disk-read bytes
  were unavailable, and the logical expert counters describe the runtime's requests.
  No hard 32 GiB cap was confirmed for this host. The retained journal, receipt and
  config cover the measured harness invocation. The cache index was discarded.
- The RTX 3070 measurement fills the run slot in the earlier protocol, which named an
  RTX 3060 Ti. Card-name validation was added after the run. No completed Moonlight run
  was superseded. Qwen stayed at the CPU planning stage, with fit, timing, updates and
  reload unmeasured. The planning report later corrected its proposed path for explicit
  RAM selection because actual-store-size validation runs before tier selection. No
  GPU run was superseded by that correction.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
