# Two-stage GPipe LoRA placement on two RTX 3090s

**Verdict (2026-10-29): positive.** Two fresh native GPipe runs updated their stage-owned adapters and passed independent reload checks.

Measured by @kok-o.

## Question and setup

Can a two-stage native GPipe split train the expected LoRA parameters on two GPUs, preserve the ordinary control within tolerance, and reload the assembled adapter in a fresh process?

- Linux host, AMD EPYC 7452, 270,101,573,632 bytes of host RAM, two NVIDIA GeForce RTX 3090s with 24 GB each and PCIe NODE topology without NVLink. Driver 595.58.03 (CUDA compatibility 13.2), torch 2.13.0+cu130 (CUDA runtime 13.0), NCCL 2.29.7, Python 3.12, Transformers 5.17.0, PEFT 0.20.0, Accelerate 1.14.0.
- `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`, IEEE FP32 with eager attention and TF32 off. LoRA r=8/alpha=16 on q/v, dropout 0. Stage 0 owns the embedding and blocks 0–11. Stage 1 owns blocks 12–23, final norm and head. Each stage holds a copy of the frozen tied embedding/head matrix.
- Each fresh run trained four AdamW steps with a global batch of 8 synthetic 128-token sequences. Two accumulated schedule calls per optimizer step each contain two microbatches of two sequences.

## Method and results

The reference was the ordinary full model on each GPU with the same initialization, token order and microbatches. Before clipping, the harness compares stage gradients and updates at atol=1e-5 and rtol=0.02. It verifies disjoint ownership, frozen weights and strict adapter assembly. Fresh reloads compare all logits at atol=rtol=1e-5.

| Run | Steps/stage | Forwards/backwards per stage | Max gradient difference | Max updated-adapter difference | Reload logit error |
|---|---:|---:|---:|---:|---:|
| Repeat 1 | 4 | 16 / 16 | 6.966292858e-7 | 1.057727786e-6 | 0.0 |
| Repeat 2 | 4 | 16 / 16 | 6.966292858e-7 | 1.057727786e-6 | 0.0 |

| Stage | Local parameter elements | LoRA tensors | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|
| 0 | 315,353,600 | 48 | 1813.75 | 1886 |
| 1 | 315,354,496 | 48 | 2832.39 | 3068 |

All 96 adapter tensors changed by the last step. The two final adapters were bitwise identical.

## Commands

The harness and the command receipts live at the evidence commit. Run `launch` and `reload` again with a fresh output directory for the second repeat.

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python benchmarks/harness/h3_pipeline_smoke.py prepare --prepared /tmp/h3-pp/prepared
python benchmarks/harness/h3_pipeline_smoke.py launch \
  --prepared /tmp/h3-pp/prepared --output /tmp/h3-pp/repeat-1 --device cuda --steps 4
python benchmarks/harness/h3_pipeline_smoke.py reload \
  --prepared /tmp/h3-pp/prepared --output /tmp/h3-pp/repeat-1 --device cuda
```

## Limits and superseded runs

The first host failed access setup before training. The first launcher on the replacement used an interpreter outside the venv and failed before model loading. The fixed launcher ran with the active interpreter, leaving the numerical gates unchanged. The saved rank records, tensors and validation index cover both accepted repeats.

The standalone recipe measures stage placement and final adapter reload. It leaves timing, model quality, periodic checkpoints, optimizer resume and general CLI pipeline support unmeasured. Allocator peaks include correctness probes. Host RSS and memory usage are point samples.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/evidence/h3-pipeline-20261029
