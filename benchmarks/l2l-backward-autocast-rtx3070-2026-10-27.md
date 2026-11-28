# Gate record - disabling autocast during layer-major backward

**Verdict (2026-10-27) — Positive.** Disabling autocast during backward restored bitwise gradient parity with ordinary streamed accumulation on the tested CUDA recipe.

Measured by @M1kado0.

## 1. Setup

Does the trainer's autocast scope explain why layer-major microbatching diverges from ordinary streamed gradient accumulation (GA)? `training.l2l` is pending.

- NVIDIA GeForce RTX 3070, 8 GB, 540,641,517,568 bytes of host RAM visible and a 259,506,831,360-byte cgroup limit. Ubuntu 24.04 development image, CUDA 13.0 image runtime, driver 595.71.05 reporting CUDA 13.2. torch 2.14.1+cu130, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2, TRL 0.29.1 and Triton 3.8.0.
- `mistralai/Mistral-7B-v0.1` in NF4, BF16 autocast and activation storage. The probe uses 64 Alpaca rows, batch size 1, four microbatches per update, 16 updates, deterministic MATH SDPA and LoRA rank 8 / alpha 16 with zero dropout. The CPU and Apple MPS controls use a small synthetic Llama in BF16. The source gives no separate CPU/MPS machine inventory or version freeze.

## 2. Method and results

The trainer originally ran layer-major backward under the enclosing autocast context. The intervention disabled autocast around the loss/head and per-layer backward calls while retaining it for forward and recomputation. The MPS isolation held the weights and inputs fixed and moved GA's backward call across its autocast boundary. CPU hooks independently checked the context seen by autograd.

| CUDA comparison with the first GA run | LoRA gradients | Saved post-update parameters | Relative loss error |
|---|---|---|---|
| Second GA run | Bitwise equal, all 16 updates | Equal, updates 0–3 | 0 |
| Layer-major before intervention | Step 0 maximum difference 1.1e-3 at maximum gradient 5.5e-2 | Difference 1.9e-4 after update 1 | Up to 0.38% at step 10 |
| Layer-major after intervention | Bitwise equal, all 16 updates | Equal, updates 0–3 | At most 1.6e-7 |

| MPS control, maximum gradient difference | GA backward outside autocast | GA backward inside autocast |
|---|---:|---:|
| Layer-major before intervention | 6.41e-04 | 0.00e+00 |
| Layer-major after intervention | 0.00e+00 | 6.41e-04 |

With autocast disabled in both arms the MPS difference was zero. Two of five CPU context tests failed before the intervention and all five passed afterward. In the five-block CUDA repeat, all 10 layer-major runs were within the 1.5× SDPA noise-floor limit at every step. Maximum layer-major loss error was 1.6e-7, minimum limit 2.3e-4 and the floor ranged from 1.5e-4 to 3.4e-3. All nine GA repeats had zero loss error.

## 3. Commands

The harness, configs and test live at the evidence commit. The CUDA probe runs the saved GA and layer-major YAMLs against the pre-intervention and corrected checkouts. Its default `--grad-steps 4` saves full gradient and parameter tensors for the first four updates. For the repeat, use the work directory containing `data/` from the recorded fixture preparation and choose a new output directory.

```bash
PYTHONPATH=src python benchmarks/harness/l2l_ticket08_step_isolation.py cpu bfloat16
PYTHONPATH=src python benchmarks/harness/l2l_ticket08_step_isolation.py mps bfloat16
python -m pytest tests/test_l2l_backward_autocast.py
python benchmarks/harness/l2l_ticket08_gpu_probe.py run --config <arm.yaml> --out <run-dir>
python benchmarks/harness/l2l_ticket08_gpu_probe.py compare <ga-dir> <candidate-dir>
python benchmarks/harness/l2l_ticket04_campaign.py --work <work-dir> --out <output-dir> \
  --card 8gb --stage abba --stage floor
```

## 4. Superseded runs and limits

The corrected imports superseded the first MPS run, which loaded a different checkout. Byte-based hashing superseded the first CUDA probe, whose four arms all stopped on BF16 serialization. The earlier forward/backward probe placed the GA reference inside autocast, so its result did not exercise the trainer's backward boundary.

The CUDA repeat recorded 165.2 tok/s for GA and 153.0 tok/s for layer-major execution, a difference of 12.2 tok/s (95% interval 10.7 to 13.7). All 20 runs had unknown power and stall validity checks, so these rates are descriptive. Later parameter tensors, quality and the corrected RTX 3090 trajectory remain unverified. The public snapshot retains comparison receipts and hashes in place of the full CUDA tensor files.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/l2l-ticket08-loss-divergence
