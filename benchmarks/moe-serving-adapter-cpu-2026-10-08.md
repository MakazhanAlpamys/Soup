# Measurement record - unmerged MoE adapters in CPU serving

Measured by @Batyrkhantzy111 on 2026-10-08.

## 1. Question and setup

Can llama.cpp apply unmerged LoRA adapters to multi-head latent attention
(MLA) and the shared expert, and does its standard attention path work on a
pretrained MoE? The probe used random, scaled-down DeepSeek-V3 and Qwen3.5-MoE
fixtures in FP32, real DeepSeek-V3 and Kimi K2 configurations for a
configuration-only export check, and a real `Qwen/Qwen3.5-35B-A3B` checkpoint at
`59d61f3ce65a6d9863b86d2e96597125219dc754`. The adapters were seeded synthetic
rank-8 PEFT LoRA with alpha 16 and nonzero B factors.

- Tiny fixtures ran CPU-only on Windows 11 Pro 26200, AMD Ryzen 5 8645HS
  (6 cores, 12 threads), 15.3 GB RAM. The reference used PyTorch 2.14.1+cpu,
  Transformers 5.19.0 and PEFT 0.21.2. The stock and patched local builds used
  MinGW-w64 GCC 15.2.0.
- The pretrained BF16 attempt used an AMD EPYC 7B13 CPU with 32 vCPUs and
  251 GiB RAM. The FP32 rerun used the same CPU model with 48 vCPUs and
  377 GiB RAM. Both were CPU-only Ubuntu hosts with kernel 6.8.0-1069-gcp.
  The cloud stack used Python 3.12.15, PyTorch 2.14.1+cpu, Transformers
  5.19.0 and PEFT 0.21.2 for the reference. The converter used PyTorch
  2.11.0+cpu, Transformers 4.57.6 and GGUF 0.19.0. The engine was llama.cpp
  `b11476`, built with GNU 11.4.0. Driver and CUDA were inapplicable.

## 2. Method

Both engines consumed identical token IDs. The harness checked base-logit
agreement and repeatability, then compared the change in logits with the
adapter enabled. The relative residual `r` divides the norm of the
engine-minus-reference adapter effect by the norm of the reference effect.
The tiny Qwen3.5 fixture served as a positive control. The real-model rule
voided a run if the normalized base gap exceeded 0.05 or a determinism or
top-1 control failed. The export-only check inspected low-rank factor pairs
and the outer dimensions of the split key and value factors.

## 3. Commands

The harnesses live under `benchmarks/harness/` at the evidence commit. The
executed revisions were `4552ce82` for the tiny fixture, `c4c09b43` for
configuration export, `f4375fa3` for the BF16 attempt and `c46ab90b` for the
FP32 rerun. The probe report also gives the local build and patched rerun commands.

```bash
python benchmarks/harness/lora_hook_parity.py \
  --llama-bin <llama-bin> --llama-src <llama-src> \
  --convert-python <convert-venv>/bin/python \
  --work-dir <scratch> --models qwen35moe-tiny,dsv3-tiny --seed 17 --threads 4 \
  --out benchmarks/results/probe-lora-hook-engines/part-a-cpu-run3.json \
  --log benchmarks/results/probe-lora-hook-engines/part-a-cpu-run3.log
python benchmarks/harness/lora_export_real_configs.py \
  --convert-python <convert-venv>/bin/python \
  --llama-src stock=<llama-src> --llama-src hook=<hook-src> --work-dir <scratch> \
  --out benchmarks/results/probe-lora-hook-engines/part-b0-export-real-configs-run3.json \
  --log benchmarks/results/probe-lora-hook-engines/part-b0-export-real-configs-run3.log
HF_XET_CHUNK_CACHE_SIZE_BYTES=0 <ref-venv>/bin/python \
  benchmarks/harness/lora_hook_real_model.py \
  --llama-bin <build>/bin --llama-src <llama-src> \
  --convert-python <convert-venv>/bin/python \
  --model-dir <checkpoint-dir> --work-dir /dev/shm/<scratch> --threads 48 \
  --box "<box>" --precision f32 --out part-b-real-model-run2.json --log part-b-real-model-run2.log
```

## 4. Results

| Check | Stock or first attempt | Patched or repeated |
|---|---|---|
| Tiny DeepSeek-shaped MLA | `q_a_proj`, `q_b_proj` and `kv_a_proj_with_mqa` each dropped with `r` = 1.000. `kv_b_proj` failed conversion. | All four applied, with `r` = 1.4e-5, 1.2e-5, 1.4e-6 and 7.4e-7, respectively. |
| Tiny DeepSeek-shaped output projection and shared expert | Applied with `r` = 1.5e-6 and 2.6e-6. | Applied with the same residuals. |
| Real configuration export | DeepSeek-V3 and Kimi K2 failed on the key-factor transpose. Qwen3.5 exported 200 pairs. | DeepSeek-V3 exported 540 pairs, Kimi K2 546, Qwen3.5 200. |
| Real Qwen3.5 base check | BF16 gap 0.066, above 0.05. Both variants void. | FP32 gap 1.3e-6, top-1 agreement 22 of 22. |
| Real Qwen3.5 FP32, default attention targets | — | Applied, `r` = 1.4e-5 at tolerance 0.020. |
| Real Qwen3.5 FP32, attention plus shared expert | — | Applied, `r` = 8.3e-6 at tolerance 0.020. |

**Verdict — mixed. Stock MLA needed a hook on the tiny fixture. The CPU FP32 rerun confirmed both standard Qwen3.5 adapter variants.**

## 5. Limitations and superseded runs

Tiny run 3 superseded runs 1 and 2 after conversion, base-parity and
instrument problems. The third configuration export run corrected the
provenance stamp and reproduced run 2's results after run 1 stopped with
truncated output. The real-model FP32 rerun superseded
the void BF16 attempt for the adapter decision. It kept the recipe and seeds,
while PEFT stored the adapter A factors at different precision. The optional
flash-attention check used an FP16 KV cache and reported two query-projection
residuals outside the FP32 rule. The full MLA checkpoint was never executed.
The real Qwen3.5 result covers one prompt and two synthetic adapters. SSD
serving, expert prefetch, GPU inference and quantized serving remained unmeasured.

Evidence: benchmarks/evidence/moe-serving-adapter-cpu-2026-10-08/ (source commit d647cefb695d)
