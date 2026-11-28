# DeepSeek-V3 671B acceptance on one Quadro P4000

**Verdict (2026-10-29): positive for two attention-LoRA updates, paired loss measurement, and exact reload within the recorded 32 GiB scope.**

Measured by @kok-o.

## 1. Question and setup

Can the real DeepSeek-V3 checkpoint be downloaded, converted, trained for two updates,
evaluated before and after training, and reloaded on one 8 GB GPU within a 32 GiB Linux
RAM limit? The October 28 RTX 5060 pilot tested the updates and reload on a larger host.

- Both runs used `deepseek-ai/DeepSeek-V3` at `e815299b0bcbac849fa540c768ef21845365c9eb`,
  with 671,026,419,200 causal-decoder parameters, 61 layers, 58 MoE layers and 256 routed
  experts per MoE layer. The 163 source shards total 688,586,727,753 bytes. FP8 weights
  were converted to a disk-backed per-expert NF4 cache with block size 64 and double
  quantization disabled. The routed experts stayed frozen.
- October 29 used one Quadro P4000 (8192 MiB reported by `nvidia-smi`, 8,507,949,056 bytes
  visible to CUDA), driver 535.183.01, Xeon W-2133 with six effective CPU threads and
  26,267,844,608 bytes of guest RAM. Linux 5.15.0-1067-kvm ran with a cgroup v2 hard cap of
  34,359,738,368 bytes and `MemorySwapMax=0`. The virtual ext4 device had 1,186,344,341,504
  bytes free before download. The stack was Python 3.12.16, PyTorch 2.6.0+cu118 / CUDA 11.8,
  Transformers 5.16.1, PEFT 0.20.0, bitsandbytes 0.50.2 and Accelerate 1.14.0. Compute used FP32.
- October 28 used one RTX 5060 (8151 MiB reported by `nvidia-smi`, 8,122,859,520 bytes
  visible to PyTorch), driver 610.57.04, Ryzen 9 5950X, 134,948,118,528 bytes of host RAM,
  a 129,549,467,648-byte container limit and CT4000P3PSSD8 NVMe. The stack was Ubuntu Linux
  7.0.0-38-generic, Python 3.12.3, PyTorch 2.8.0+cu128 / CUDA 12.8, Transformers 5.16.1,
  PEFT 0.20.0, bitsandbytes 0.50.2 and Accelerate 1.14.0. Compute used BF16.

## 2. Method

Both runs used rank 2, alpha 4, dropout 0 and AdamW at 1e-4 with gradient norm clipped at
1.0. LoRA covered `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` and `o_proj`.
Two authored training strings ran unpadded at batch size 1, maximum length 64 and without
gradient accumulation. The timer spans the two synchronized training steps only, including
gradient checks, digest copies and optimizer updates.

The October 29 protocol selected eight WikiText-2 test fragments at revision
`b08601e04326c79dfdd32d625aee71d232d685c3` before the run. It required finite losses and
identical token digests and predicted-token counts in both arms, with the same dtype and
cache. It reports the token-weighted NLL and perplexity for any finite result. The initial
LoRA B matrices were zero. Exact reload required the saved adapter digest and the held-out
logits to match on a fresh runtime. RAM samples cover download, preparation, both
evaluation arms, training and reload.

## 3. Results

| Measurement | October 29, Quadro P4000 FP32 | October 28, RTX 5060 BF16 |
|---|---:|---:|
| Cache preparation (seconds) | 4381.027778 | 2568.94 |
| Cache size (bytes) | 384,198,038,058 | 380,276,447,824 |
| Training throughput (input tokens/s) | 0.1445395182325024 | 0.14562352 |
| Peak CUDA allocated (bytes) | 7,365,200,384 | 3,324,763,648 |
| Peak CUDA reserved (bytes) | 7,759,462,400 | 3,714,056,192 |
| Maximum absolute reload logit error | 0 | 0 |

| Run | Update | Tokens | Loss | Pre-clipping gradient norm | Finite gradient tensors | Seconds |
|---|---|---:|---:|---:|---:|---:|
| Quadro P4000 | First | 64 | 4.194044113 | 0.547222912 | 610 | 440.079724 |
| Quadro P4000 | Second | 62 | 3.821666241 | 0.484790951 | 610 | 431.654190 |
| RTX 5060 | First | 64 | 4.22038984299 | 0.56822538376 | 610 | 486.35249 |
| RTX 5060 | Second | 62 | 3.83308029175 | 0.48280650377 | 610 | 378.89232 |

| WikiText-2 source row, Quadro P4000 | Predicted tokens | Base NLL | Adapter NLL | Adapter minus base |
|---|---:|---:|---:|---:|
| 260 | 63 | 3.057403564 | 3.071856022 | +0.014452457 |
| 711 | 63 | 1.961974025 | 1.964350820 | +0.002376795 |
| 1275 | 63 | 1.244180679 | 1.244797111 | +0.000616431 |
| 1775 | 63 | 2.878654242 | 2.876713514 | -0.001940727 |
| 2380 | 63 | 2.596593857 | 2.588669062 | -0.007924795 |
| 2761 | 63 | 2.573756933 | 2.575625181 | +0.001868248 |
| 3373 | 63 | 2.065718889 | 2.033879519 | -0.031839371 |
| 4034 | 63 | 1.701162457 | 1.717752099 | +0.016589642 |
| Token-weighted NLL | 504 | 2.259930580854 | 2.259205415845 | -0.000725165009 |
| Perplexity | 504 | 9.582423940 | 9.575477620 | |

The Quadro P4000 run recorded 5672 RAM samples, a sampled cgroup peak of
25,047,044,096 bytes, a maximum sampled sum of process RSS of 19,820,154,880 bytes,
and a model-process `VmHWM` of 20,387,213,312 bytes. The recorded scope disabled swap,
guest `SwapTotal` was 0, and observed swap use and the cgroup OOM-kill delta were both 0.
Both trials changed the adapter digest on each update and
reloaded the final digest with bitwise-equal held-out logits. The earlier RTX 5060
authored probe loss went from 4.5601024628 to 4.5665259361.

The evidence includes the two event journals and the constrained evaluation and RAM
summaries. The original preparation is recorded in the October 28 failed-reload journal.

```text
benchmarks/evidence/b6-20261028/real-giant/{events.jsonl,failed-reload-trial-events.jsonl}
benchmarks/evidence/b6-20261029/real-giant/{events.jsonl,evaluation-summary.json,ram-summary.json}
```

## 4. Commands

The historical scripts and protocol live at the evidence commit. The October 29 scope
wrapped the retained `runner.sh`, which invoked the RAM monitor around `experiment.sh`.
The latter runs the header planner, downloads the pinned snapshot and trains with the
original protocol and evaluation files. It sets the execution environment in the script.

```bash
systemd-run --scope --unit=b6-final -p MemoryMax=32G -p MemorySwapMax=0 \
  bash /root/b6/final-runner.sh
/root/b6/venv/bin/python /root/b6/repo-final/benchmarks/harness/b6_ram.py \
  --output /root/b6/results/attempt3 --phase-file /root/b6/phase3.txt --interval 2 \
  -- bash /root/b6/experiment-final.sh
/root/b6/venv/bin/python benchmarks/harness/b6_source_plan.py \
  --revision e815299b0bcbac849fa540c768ef21845365c9eb \
  --output /root/b6/results/attempt3/source-plan
/root/b6/venv/bin/python benchmarks/harness/b6_real_moe.py \
  --source /root/b6/source --cache /root/b6/cache --output /root/b6/results/attempt3/real \
  --data examples/b6-pilot-data.json --revision e815299b0bcbac849fa540c768ef21845365c9eb \
  --steps 2 --max-length 64 --shard-threads 6 \
  --evaluation-protocol benchmarks/protocols/b6-acceptance-20261029/protocol.json \
  --evaluation-data benchmarks/protocols/b6-acceptance-20261029/evaluation-data.json \
  --ram-phase-file /root/b6/phase3.txt --source-plan /root/b6/results/attempt3/source-plan/plan.json
```

On October 28 the timed repeat loaded the cache made by the earlier trial. The retained
`retry3.sh` used `--restart-incomplete-cache` with output `/root/b6/results/real-retry3`.
The retained `retry4.sh` ran the following command.

```bash
export PYTHONPATH=/root/b6/repo:/root/b6/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/b6/venv/bin/python benchmarks/harness/b6_real_moe.py \
  --source /root/b6/source --cache /root/b6/cache --output /root/b6/results/real-retry4 \
  --data examples/b6-pilot-data.json --revision e815299b0bcbac849fa540c768ef21845365c9eb \
  --steps 2 --max-length 64 --shard-threads 8 --reuse-cache
```

The acceptance tooling for `training.stream_expert_layout: per_expert` is pending.
Its monitor and replay commands are below. For a new run, point `PROTOCOL` and
`EVALUATION_DATA` at the files under `benchmarks/protocols/giant-moe-acceptance-2026-10-29/`
used by that run. For historical replay, use the original files from the evidence
commit under `benchmarks/protocols/b6-acceptance-20261029/` because the recorded journal
binds their exact bytes. Decompress the retained `ram-samples.jsonl.gz` for `RAM_SAMPLES`.

```bash
# Monitor a new workload that emits the protocol's phase markers.
python -m benchmarks.harness.giant_moe_ram \
  --output "$RAM_OUTPUT" --phase-file "$PHASE_FILE" -- "$WORKLOAD"
# For a new run using the current fixtures:
PROTOCOL=benchmarks/protocols/giant-moe-acceptance-2026-10-29/protocol.json
EVALUATION_DATA=benchmarks/protocols/giant-moe-acceptance-2026-10-29/evaluation-data.json
# Replay the prior pilot journal, or the constrained run with its RAM and evaluation receipts.
python -m benchmarks.harness.giant_moe_verdict \
  --events "$EVENTS" --index "$INDEX" --gpu "$GPU_CSV" --output pilot-verdict.json
python -m benchmarks.harness.giant_moe_acceptance_verdict \
  --events "$EVENTS" --index "$INDEX" --ram-samples "$RAM_SAMPLES" \
  --protocol "$PROTOCOL" --evaluation-data "$EVALUATION_DATA" \
  --physical-vram-mib "$PHYSICAL_VRAM_MIB" --returncode "$WORKLOAD_EXIT_CODE" \
  --output acceptance-verdict.json
```

## 5. Limitations and superseded runs

- The constrained run followed two failed starts, a disk preflight refusal and a
  temporary CPU allocation refusal. The completed run replaces those starts for the
  acceptance measurement. The October 28 run remains its own cohort. On that host the
  validated-cache repeat superseded the earlier reload failure. Its reported conversion
  is from the preceding preparation with warm source reads.
- The cgroup kernel peak and global OOM counter were unavailable on the Quadro host.
  The table uses PyTorch allocator peaks and the RAM figures report sampled maxima where
  indicated. The RTX 5060 pilot left the 32 GiB RAM requirement untested.
- The eight paired fragments qualify the loss measurement only. This was an eager harness
  run at short sequence length and batch size 1, with general quality, convergence,
  target throughput, optimizer-step resume, and end-to-end `soup train` left unmeasured.
  The evidence retains the journals, cache index, RAM samples, evaluation output and
  adapter and source manifests. The temporary weights and cache are required for a new
  hardware run.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
