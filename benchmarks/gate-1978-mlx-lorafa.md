# Native MLX LoRA-FA: bounded operating point

Apple M4 Max, 128 GiB unified memory, macOS arm64, Python 3.12.2. MLX 0.32.4 / mlx-lm 0.32.2 / Transformers 5.19.0. The standalone native environments contain no torch, PEFT or TRL. The floor (MLX 0.31.2 / mlx-lm 0.31.3) is separately exercised.

Local NVIDIA Nemotron Nano 4B BF16: 263 BF16 base tensors, config SHA256 `fde9241f66cd414458df444a80eb535f53aef2b4b240a91f092e651fa6f27219`; weights previously verified as `55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`. No HF revision is known. Neither base weights nor config are modified.

Six identical plaintext rows `Question: What is 2 plus 2? Answer: 4.` (JSONL SHA256 `efd84db2d8941cfb258c3a8d9e7681ae047ad63e50bcadc81133fb73aa46d748`), 96 trained tokens per arm. Rank 4, alpha 8, dropout 0, batch/accumulation 1, constant LR 1e-4, no warmup/decay, global clipping 1, gradient checkpointing on, Rewind off. Targets: `mixer.q_proj`, `mixer.v_proj`. Six optimizer updates and a complete A/B checkpoint each iteration.

Each arm runs sequentially in a fresh process (LoRA then FA), without concurrent tests or benchmarks launched by this session. The first report interval includes tracing/compilation and work up to the first report. Subsequent report intervals include the preceding checkpoint save and next update; they are not pure kernel timings. CLI training time includes loading/setup/training/final save. Gram setup is measured inside native optimizer construction. Metal uses `mx.get_peak_memory()`; RSS uses macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss` in bytes, captured at the end of training. Complete-cycle peaks after additional reload/inference are reported separately.

| Measurement | LoRA | LoRA-FA |
|---|---:|---:|
| CLI training seconds | 1.737503 | 1.561980 |
| Gram setup seconds | 0.000000 | 0.001484 |
| First report interval seconds | 0.199559 | 0.128634 |
| Median subsequent interval seconds | 0.094950 | 0.092814 |
| Training Metal peak bytes | 8098651308 | 8097335666 |
| Training RSS peak bytes | 8478392320 | 8482553856 |
| Complete-cycle Metal peak bytes | 24063200718 | 24062775246 |
| Trainable tensors / elements | 16 / 198656 | 8 / 98304 |
| Moment tensors / elements | 32 / 397312 | 16 / 196608 |

This single tiny operating point shows fewer trainable/moment elements, but essentially unchanged total memory: FA training RSS is slightly higher. The short timings are descriptive, with fixed arm order and warm filesystem/model caches; they do not establish a general speedup, memory improvement or downstream quality. Ordinary MLX AdamW and FA intentionally use different documented optimizer math. No answer-quality comparison is made from the capped thinking completions.

Both FA stacks preserve all 263 base tensors and every A exactly, update all eight B tensors, save all sixteen A/B tensors and reproduce probe logits exactly in a fresh process. Native infer matches direct greedy output/count (16-token cap). Tiny local dense/4-bit fixtures additionally qualify masking, accumulation/checkpointing, periodic/final complete saves, warm-start and float32 fusion.

## Reproduction

Use an owned checkout and an MLX-only environment with the optional datasets reader, the pinned local BF16 model, and separate fresh output directories. Run each arm in a fresh process:

```bash
PYTHONPATH=src python benchmarks/mlx_lorafa_nemotron.py --model /path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 --output output/lora --method lora
PYTHONPATH=src python benchmarks/mlx_lorafa_nemotron.py --model /path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 --output output/lorafa --method lorafa
```

Numerical reference: PEFT 0.21.2 `peft/optimizers/lorafa.py`, SHA256 `b9f1616b3120796bd765bc2bd59390bc998d5f81b0b66bf26d58129e2e0bc82b`. CPU FP32, BF16 autocast disabled at the device-capability boundary. NumPy FP32 Gram SVD relative tolerance is rank × float32 epsilon. Betas (.9,.999), epsilon 1e-6, bias correction, post-update decay; raw global clipping precedes projection. Actual PEFT and independent SVD-oracle comparisons cover several updates/moments, not only finite loss.

Run the reference in a separate PEFT environment, preserving the standalone runtime:

```bash
SOUP_PEFT_REFERENCE_PYTHON=/path/to/peft-0.21.2/bin/python PYTHONPATH=src python -m pytest -o addopts= -q tests/test_mlx_lorafa_native.py
```
