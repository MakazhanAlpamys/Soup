# Issue #837 CUDA micro-benchmark

Measured 2026-11-08 in Google Colab on a Tesla T4 (compute capability 7.5, 15 GiB, driver 580.82.07). The fixture is one synthetic Llama-3.1-8B-shaped SwiGLU MLP block (`N=2048`, `d=4096`, `f=14336`, `r=64`). It does not load a language model and is not an end-to-end training result.

The harness interleaves Fast-LoRA, plain PEFT and PEFT + Liger. Each arm has five warm-ups and 30 CUDA-event forward-plus-backward measurements. It records saved autograd tensors and `torch.cuda.max_memory_allocated`. Full machine-readable figures and versions are in [issue-837-cuda-t4.json](issue-837-cuda-t4.json); rerun using [fast_lora_mlp_bench.py](../harness/fast_lora_mlp_bench.py).

| Dtype | Path | Median fwd+bwd | Saved tensors | Peak allocated |
|---|---|---:|---:|---:|
| BF16 | Fast-LoRA | 811.20 ms | 494,141,440 B | 1,306.2 MiB |
| BF16 | Plain PEFT | 793.05 ms | 611,844,096 B | 745.8 MiB |
| BF16 | PEFT + Liger | 791.44 ms | 553,123,840 B | 689.8 MiB |
| NF4 | Fast-LoRA | 820.08 ms | 232,699,084 B | 1,478.4 MiB |
| NF4 | Plain PEFT | 802.62 ms | 259,522,560 B | 1,069.9 MiB |
| NF4 | PEFT + Liger | 804.60 ms | 200,802,304 B | 1,013.9 MiB |

Fast-LoRA reduced saved-tensor bytes versus plain PEFT by 117,702,656 B in BF16 and 26,823,476 B in NF4. On this T4 run it was about 2.3% slower than plain PEFT in BF16 and 2.2% slower in NF4, while its measured peak allocated memory was higher than both comparison paths. The benchmark therefore confirms the saved-tensor reduction but does not show a speed or peak-memory win on this card.

The outputs matched plain PEFT exactly in both dtypes. The largest relative L2 difference across the input and LoRA gradients was 0.00432 for Fast-LoRA (BF16) and 0.00431 (NF4). Every arm's autograd path proof passed all 30 repetitions. SM clocks varied during measurement; ranges and medians are recorded in the JSON. These numbers describe this single-card, single-block fixture only.
