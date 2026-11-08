# Issue #837 CUDA micro-benchmark

Measured 2026-11-08 in Google Colab on a Tesla T4 (compute capability 7.5, 15 GiB, driver 580.82.07). The fixture is one synthetic Llama-3.1-8B-shaped SwiGLU MLP block (`N=2048`, `d=4096`, `f=14336`, `r=64`). It does not load a language model and is not an end-to-end training result. BF16 on this T4 (sm_75) is emulated, not native.

The rerun used one fresh Python process per dtype cell (emulated BF16 first, then NF4), preventing allocator state and cached kernels from the previous dtype from carrying over. Inside each process, Fast-LoRA, plain PEFT, and PEFT + Liger remained paired and interleaved in this balanced rotating schedule: `fast, plain, liger`; `fast, liger, plain`; `plain, fast, liger`; repeating. Each arm had five warm-ups and 30 forward-plus-backward CUDA-event measurements. Saved-tensor bytes were captured once per arm; peak memory is the largest absolute `torch.cuda.max_memory_allocated()` reading over timed iterations. No noise threshold was specified before this rerun. Report medians and within-process IQRs descriptively; do not treat small differences as statistically established. This run does not estimate process-to-process variability. Full figures and runtime metadata are in [issue-837-cuda-t4.json](issue-837-cuda-t4.json).

| Dtype | Path | Median fwd+bwd | IQR (p25–p75) | Saved tensors | Peak allocated |
|---|---|---:|---:|---:|---:|
| Emulated BF16 (sm_75) | Fast-LoRA | 595.26 ms | 582.61–610.80 ms | 494,141,440 B | 1,306.2 MiB |
| Emulated BF16 (sm_75) | Plain PEFT | 582.25 ms | 569.57–595.87 ms | 611,844,096 B | 745.8 MiB |
| Emulated BF16 (sm_75) | PEFT + Liger | 582.86 ms | 566.64–595.36 ms | 553,123,840 B | 689.8 MiB |
| NF4 | Fast-LoRA | 673.56 ms | 654.93–688.39 ms | 232,699,084 B | 1,128.9 MiB |
| NF4 | Plain PEFT | 652.18 ms | 640.02–667.04 ms | 259,522,560 B | 720.4 MiB |
| NF4 | PEFT + Liger | 656.89 ms | 637.21–668.26 ms | 200,802,304 B | 664.4 MiB |

Fast-LoRA reduced saved-tensor bytes versus plain PEFT by 117,702,656 B in emulated BF16 and 26,823,476 B in NF4. Its median was 2.23% higher in emulated BF16 and 3.28% higher in NF4. Those are descriptive differences from one process per dtype, with no preregistered noise threshold; no significance claim is made. Fast-LoRA also had higher measured peak allocated memory than both comparison paths. This run did not capture an allocation breakdown that explains the extra peak, so its cause remains unknown. The result confirms the saved-tensor reduction but does not demonstrate a speed or peak-memory win on this T4 at these shapes.

Outputs matched plain PEFT exactly in both dtypes. The largest relative L2 difference across input and LoRA gradients was 0.00432 for Fast-LoRA (emulated BF16) and 0.00431 (NF4). Every arm's autograd path proof passed all 30 repetitions. SM clocks varied during measurement; ranges and medians are recorded in the JSON. These numbers describe this single-card, single-block fixture only.
