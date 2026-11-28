# Synthetic 671B MoE cold and warm runs on one RTX 3060 Ti

**Verdict (2026-11-09): positive for cold and warm execution, adapter updates and parameter reload on the same synthetic store. The first-step timing difference has no causal verdict.**

Measured by @M1kado0.

## 1. Question and setup

Can a disk-backed synthetic checkpoint with DeepSeek-V3 geometry complete three
LoRA updates on one 8 GB GPU after the Linux file page cache is evicted, and can
the same store repeat the updates in a fresh process without eviction?

- The generated store has 671,026,419,200 parameter positions, 61 layers, 58 MoE
  layers, 256 routed experts per MoE layer, hidden size 7168 and expert width 2048.
  It uses random weights, a per-expert NF4 layout, double quantization disabled, BF16
  compute and disk streaming. The generator seed was 20261109.
- One RTX 3060 Ti (8192 MiB reported by `nvidia-smi`, 8,222,670,848 bytes visible to
  PyTorch), driver 595.104.02, Xeon E5-2470 v2 with 20 effective CPUs, 202,543,054,848
  bytes of host RAM and a cgroup `memory.max` of 194,440,593,408 bytes. Swap was disabled.
  The container ran Ubuntu 24.04 with host Linux 7.0.0-38-generic. Disk media were unverified.
- Python 3.12.3, PyTorch 2.14.1+cu130 / CUDA 13.0, Transformers 5.20.2, PEFT 0.21.2,
  bitsandbytes 0.50.2, Accelerate 1.15.0 and safetensors 0.8.0.

## 2. Method

The disk writer checks free space before writing, then records the store size. Each run
used three unpadded batches of one sequence of 128 seeded random tokens, no accumulation,
rank 8 attention-LoRA with alpha 16 and dropout 0, and AdamW at 1e-4 with gradient norm
clipped at 1.0. The journal judge required a finite global pre-clipping gradient norm
above 1e-06 at every step, a finite gradient for every trainable tensor, relative adapter
movement above 1e-06 after the first and final steps, a continuous digest chain and a
matching reloaded parameter digest.

For the cold run, the harness used `posix_fadvise(DONTNEED)` on the store files and checked
with `mincore` that at most 1% remained resident. The cold gate also required first-step
`/proc/self/io` `read_bytes` of at least 90% of the logical expert bytes for that step.
The expert path reads with direct I/O, so the byte threshold corroborates disk activity
in either cache state. The warm repeat used the same store and a new process with no
eviction request. The first step is reported separately from the median of steps 1 and 2.

## 3. Results

| Store and cold-start measurement | Result |
|---|---:|
| Planned / actual store bytes | 380,276,485,553 / 380,276,433,625 |
| Preflight required / measured available before writing (bytes) | 405,632,188,590 / 509,206,491,136 |
| Completed store files | 122 |
| Simulated preflight at 200,000,000,000 free bytes | Refused before writing |
| Store page cache before / after eviction (bytes) | 78,781,139,337 / 0 |
| Eviction time (seconds) | 6.3 |
| Cold peak CUDA allocated / reserved (bytes) | 3,848,212,992 / 4,630,511,616 |
| Cold sampled `nvidia-smi` maximum (MiB) | 5850 |
| Cold process `VmHWM` at completion (bytes) | 6,641,844,224 |

| Step | Cold input tokens/s | Warm input tokens/s | Cold `read_bytes` (GB) | Warm `read_bytes` (GB) | Cold logical expert bytes (GB) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.61 | 1.87 | 99.9 | 99.5 | 79.2 |
| 1 | 1.84 | 1.94 | 96.0 | 96.0 | 79.9 |
| 2 | 1.89 | 1.94 | 95.8 | 95.8 | 79.7 |
| Median of steps 1 and 2 | 1.87 | 1.94 | | | |

| Cold step | Loss | Pre-clipping gradient norm | Finite gradients | Distance from initial adapter |
|---|---:|---:|---:|---:|
| 0 | 13.0059 | 0.1613 | 610 of 610 | 0.3640 |
| 1 | 13.3544 | 0.0297 | 610 of 610 | 0.5801 |
| 2 | 13.2704 | 0.0346 | 610 of 610 | 0.7784 |

Both journals passed. Step 0 took 211.0 seconds cold and 68.5 seconds warm. Losses
and adapter digests matched across runs. Both reloads reproduced the final adapter
parameter digest. The cold and warm journals, gate outputs and store record are retained.

## 4. Commands

The scripts and config live at the evidence commit. The preflight-only wrapper runs the
size plan and refusal test. The writer command was followed by the same command with
`--resume`, as recorded in `remote-resume.sh`. The training commands expand the two
invocations of `remote-run.sh` with separate outputs.

```bash
PYTHONPATH="$PWD/src" sh benchmarks/evidence/b5-synthetic671b-20261109/plan-and-refusal.sh \
  "$PYTHON" "$UNUSED_DESTINATION" "$EVIDENCE_OUTPUT"
export PYTHONPATH=/root/b5/repo:/root/b5/repo/src
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHDYNAMO_DISABLE=1 \
  /root/b5/venv/bin/python benchmarks/harness/b5_synth_store.py write --shape deepseek-v3 \
  --dtype bfloat16 --seed 20261109 --out /root/b5/synthetic-v3 \
  --record /root/b5/results/store/store-record.json
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
/root/b5/venv/bin/python benchmarks/harness/b5_real_moe.py \
  --config benchmarks/protocols/b5-synthetic671b-20261109/soup.yaml \
  --store /root/b5/synthetic-v3 --output /root/b5/results/cold1/run --steps 3 --device cuda --cold
/root/b5/venv/bin/python benchmarks/harness/b5_real_moe.py \
  --config benchmarks/protocols/b5-synthetic671b-20261109/soup.yaml \
  --store /root/b5/synthetic-v3 --output /root/b5/results/warm1/run --steps 3 --device cuda
python benchmarks/harness/b5_cold_acceptance.py --journal /root/b5/results/cold1/run/journal.jsonl \
  --store-record /root/b5/results/store/store-record.json
python benchmarks/harness/b5_cold_acceptance.py --journal /root/b5/results/warm1/run/journal.jsonl \
  --store-record /root/b5/results/store/store-record.json --warm
```

## 5. Limitations and superseded runs

- Two starts ended before training. The completed host required resuming an interrupted
  store write, preserving 94 files checked by header and length. The writer's reported
  starting free space after resume was reconstructed. The table uses the original
  `df -B1` receipt. Both cold and warm measurements stand.
- File-page eviction leaves device and controller caches untouched. The runs had similar
  first-step `read_bytes` and the cause of the wall-time gap remains unknown. A single
  cold/warm pair yields no run-to-run variance.
- The narrow gate covers random weights and tokens, one card and three updates. Real
  checkpoint quality and performance, reload logits, an end-to-end `soup train` command
  and 32 GiB RAM fit remain unmeasured. CUDA peaks are allocator counters, and model
  process RSS does not include the full filesystem page cache.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/evidence/b5-synthetic671b-20261109
