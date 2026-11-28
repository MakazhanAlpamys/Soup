# CUDA dropout replay for layer-major LoRA accumulation

**Verdict (2026-10-22): positive.** Replaying the native CUDA dropout sequence preserved losses, all adapter gradients and post-step RNG state on the measured fixture.

Measured by @M1kado0.

## Question and setup

Does a layer-major window replay the random-number sequence of ordinary gradient accumulation when LoRA dropout is enabled?

An RTX 3060 with 12 GB VRAM used driver 580.65.06, CUDA 13.0, Ubuntu 24.04, Python 3.12.3, torch 2.14.1+cu130, Transformers 5.19.0, PEFT 0.21.2 and safetensors 0.8.0. Host RAM was not captured in the final replay report. The harness used real `HuggingFaceTB/SmolLM2-135M` weights in BF16 without quantization, with synthetic 128-token rows and labels equal to input IDs. The CLI's layer-major accumulation switch `training.l2l` was pending at the publication base.

## Method and results

Reset the seed before ordinary gradient accumulation and before each layer-major step. Compare loss, all 240 LoRA gradients and both CPU and CUDA generator state with `torch.equal`. Check that sharing the CUDA stream without replay and changing the seed cause differences.

| Case | Window 1 digest | Window 2 digest | Window 4 digest | Max absolute difference |
|---|---|---|---|---:|
| Dropout 0 | 4b268139 | b7ee0773 | 4f031f7c | 0.0 |
| Dropout 0.05 | 30b4844c | 98982288 | 1545f2da | 0.0 |
| Dropout 0.05, fresh process | 30b4844c | 98982288 | 1545f2da | 0.0 |
| Dropout 0.05, deterministic mode | 30b4844c | 98982288 | 1545f2da | 0.0 |

At window sizes 1, 2 and 4, ordinary accumulation recorded 58 / 115 / 229 layer loads and replay recorded 58 / 58 / 58. Baseline GPU allocation peaks were 184–188 MB and deterministic peaks were 234–238 MB. The focused CUDA suite recorded 12 passes and 1 skip after correcting test initialization.

## Commands

The bootstrap, measurement harness and captured `runall3.sh` launcher live at the evidence commit. The launcher uses the staged offline weights and performs the fit check before the full matrix.

```text
benchmarks/a0-bootstrap.sh
benchmarks/harness/l2l_smollm2_check.py
benchmarks/harness/l2l_fit_check.py
runall3.sh
```

## Limits and superseded runs

The native replay supersedes the CUDA portion of an earlier hook-based experiment. The first two replay attempts stopped while staging weights and contribute no measurements. The final test run supersedes an earlier test run that took its RNG snapshot before CUDA initialization. The single-window probe is used for fit estimation only.

An October 21 dropout-free pilot on a separate RTX 3060 passed the windows 1, 2 and 4 with both dense and NF4 base weights. It used driver 595.91.07, the same CUDA 13.0 build and bitsandbytes 0.50.2. Native replay with dropout was measured on dense weights only. Logits and hidden states were not directly compared. The full CUDA trainer with a real collator and data loader, two GPUs, varying-length dropout windows and larger models remain unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results
