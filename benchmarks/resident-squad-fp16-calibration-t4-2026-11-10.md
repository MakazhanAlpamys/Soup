# Gate record - resident SQuAD 2.0 FP16 calibration

**Verdict (2026-11-10) — Mixed.** Fast-LoRA and the adapter-matched NoRA control passed numerical calibration. Liger and Unsloth failed the gate, stopping the full campaign.

Measured by @kok-o.

## 1. Setup

Can six resident training configurations pass the calibration for a full SQuAD 2.0 comparison on a Tesla T4?

- One NVIDIA Tesla T4 (sm75, 15,360 MiB available), driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130, Transformers 5.17.0, PEFT 0.20.0, safetensors 0.8.0, Liger Kernel 0.8.4 and Unsloth 2026.10.4. The launcher specifies TRL 1.9.2 and bitsandbytes 0.50.2. OS release, Python version and host RAM are unspecified in the compact evidence.
- TinyLlama 1.1B Chat, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. FP16 base and autocast, FP32 LoRA parameters, GradScaler starting at 128, rank and alpha 16, seed 792, microbatch 2 and accumulation 8. The measured mixed-precision behavior used a pending update to `training.fast_lora`.
- The fixed tokenized fixture contains 13,032 train, 1,024 validation and 2,048 test examples. Calibration draws matched batches from train. The validation and test splits are reserved for the full campaign.

## 2. Method and results

Each arm completed 20 optimizer updates on the same physical device. The gate compared losses rounded to three decimals and checked 616 saved adapter and gradient tensors per LoRA arm against ordinary PEFT LoRA, using a 1% relative L2 limit. The NoRA arm uses different adapter math. Its control uses ordinary LoRA from the same PEFT build, and a separate NoRA audit checked zero-update behavior, gradients, save/load and merge.

| Arm | Median update | Allocated / reserved peak | Losses matching PEFT | Tensors beyond 1% relative L2 |
|---|---:|---:|---:|---:|
| PEFT LoRA | 2.528 s | 5.524 / 5.936 GiB | Reference | Reference |
| Soup Fast-LoRA | 2.644 s | 4.477 / 4.861 GiB | 20/20 | 0/616 |
| LoRA, same PEFT build as NoRA | 2.621 s | 5.524 / 5.936 GiB | 20/20 | 0/616 |
| NoRA | 2.834 s | 5.526 / 5.938 GiB | Different adapter math | Different adapter math |
| LoRA + Liger | 2.519 s | 4.616 / 4.938 GiB | 13/20 | 56/616 |
| Unsloth | 2.533 s | 6.478 / 6.629 GiB | 8/20 | 148/616 |

Fast-LoRA and the LoRA control each matched all 616 tensors bitwise. Fast-LoRA used 18.95% less peak allocated VRAM and took 4.58% longer per median update in this calibration. NoRA took 8.12% longer per median update than its corresponding LoRA control.

## 3. Commands

The executed launcher and replay script are `source/launch.py` and `source/replay_fp16.py` inside `evidence.zip` at the evidence commit. The launcher prepares the tokenized fixture and model, runs the NoRA audit and issues the following training command once per arm. Replace the shell variables with the arm and directories prepared by the launcher.

```bash
python -u benchmarks/harness/d2_quality_run.py train --backend "$arm" \
  --data "$data" --model "$model" --output "$result_dir" \
  --device cuda --mode calibrate --updates 20 --batch-size 2 --accumulation 8 --seed 792
python source/replay_fp16.py /absolute/path/to/original-evidence-with-tensor-snapshots
```

The recorded arm values are `peft`, `soup`, `nora-control`, `nora`, `liger` and `unsloth`. The replay command requires the original tensor snapshots. Its saved output is `local-replay.json`.

## 4. Superseded runs and limits

No measured arm was superseded. The two hardware requests both yielded T4s, so the planned BF16 calibration remains open. The medians are single-calibration observations without paired timing repeats or a supervisor. The numerical gate stopped the full campaign before convergence training or validation/test scoring. Exact match and F1 remain unmeasured. The evidence retains snapshot digests and per-update records. Full tensor files are outside the public bundle.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/d2-squad-fp16-20261110
