# Issue #837 CUDA micro-benchmark

Measured 2026-11-08 in Google Colab on a Tesla T4 (compute capability 7.5, 15 GiB, driver 580.82.07). The fixture is one synthetic Llama-3.1-8B-shaped SwiGLU MLP block (`N=2048`, `d=4096`, `f=14336`, `r=64`). It does not load a language model and is not an end-to-end training result. BF16 on this T4 (sm_75) is emulated, not native.

The recorded run used one process for both dtype cells, in BF16-then-NF4 order. Within each cell it used five warm-ups per arm, then 30 forward-plus-backward iterations per arm, timed with CUDA events. The arms followed a balanced rotating schedule: `fast, plain, liger`; `fast, liger, plain`; `plain, fast, liger`; then that three-iteration pattern repeated. Saved-tensor bytes were captured once per arm before timing; peak memory is the largest absolute `torch.cuda.max_memory_allocated()` reading over timed iterations. The original run did not define a noise threshold before timing, so the medians and IQRs below are descriptive; small differences should not be treated as statistically established. The current harness launches each dtype cell in a fresh process for future reruns. Full machine-readable figures and versions are in [issue-837-cuda-t4.json](issue-837-cuda-t4.json).

| Dtype | Path | Median fwd+bwd | Saved tensors | Peak allocated |
|---|---|---:|---:|---:|
| Emulated BF16 (sm_75) | Fast-LoRA | 811.20 ms | 494,141,440 B | 1,306.2 MiB |
| Emulated BF16 (sm_75) | Plain PEFT | 793.05 ms | 611,844,096 B | 745.8 MiB |
| Emulated BF16 (sm_75) | PEFT + Liger | 791.44 ms | 553,123,840 B | 689.8 MiB |
| NF4 | Fast-LoRA | 820.08 ms | 232,699,084 B | 1,478.4 MiB |
| NF4 | Plain PEFT | 802.62 ms | 259,522,560 B | 1,069.9 MiB |
| NF4 | PEFT + Liger | 804.60 ms | 200,802,304 B | 1,013.9 MiB |

Fast-LoRA reduced saved-tensor bytes versus plain PEFT by 117,702,656 B in emulated BF16 and 26,823,476 B in NF4. Its median was about 2.3% higher in emulated BF16 and 2.2% higher in NF4; these are descriptive differences from one process, not a significance claim. Its measured peak allocated memory was also higher than both comparison paths. The run did not capture an allocation breakdown that explains this higher peak, so its cause remains unknown. This benchmark confirms the saved-tensor reduction but does not demonstrate a speed or peak-memory win on this card.

The outputs matched plain PEFT exactly in both dtypes. The largest relative L2 difference across the input and LoRA gradients was 0.00432 for Fast-LoRA (emulated BF16) and 0.00431 (NF4). Every arm's autograd path proof passed all 30 repetitions. SM clocks varied during measurement; ranges and medians are recorded in the JSON. These numbers describe this single-card, single-block fixture only.
