### T1. Unprofiled timing: median step over steps 6-105 of each T cell (one cell per round; a round = one seed)

| arm | round (seed) | median step s | tok/s (4096 / median) | tok/s wall (harness) | p10 / p90 step s | 2nd half / 1st half | SM clock median / min MHz | power median / max W | temp max C | peak alloc MiB | peak reserved MiB |
|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| A | r1 (3407) | 2.2582 | 1,814 | 1,807 | 2.257 / 2.259 | 1.000 | 2,715 / 2,520 | 413 / 421 | 78 | 17,425 | 21,592 |
| B | r1 (3407) | 1.6965 | 2,414 | 2,403 | 1.693 / 1.698 | 1.001 | 2,700 / 2,520 | 445 / 452 | 79 | 13,520 | 17,942 |
| C | r1 (3407) | 1.6965 | 2,414 | 2,404 | 1.692 / 1.698 | 1.001 | 2,700 / 2,505 | 445 / 452 | 78 | 13,520 | 17,996 |
| D | r1 (3407) | 2.2163 | 1,848 | 1,840 | 2.215 / 2.218 | 1.001 | 2,730 / 2,715 | 410 / 429 | 74 | 10,852 | 11,944 |
| A | r2 (3408) | 2.2562 | 1,815 | 1,809 | 2.255 / 2.258 | 1.000 | 2,730 / 2,520 | 410 / 417 | 77 | 17,425 | 21,592 |
| B | r2 (3408) | 1.6946 | 2,417 | 2,406 | 1.692 / 1.696 | 1.001 | 2,700 / 2,535 | 445 / 452 | 79 | 13,520 | 17,942 |
| C | r2 (3408) | 1.6944 | 2,417 | 2,406 | 1.691 / 1.696 | 1.001 | 2,700 / 2,520 | 444 / 453 | 79 | 13,520 | 17,940 |
| D | r2 (3408) | 2.2161 | 1,848 | 1,840 | 2.215 / 2.217 | 1.000 | 2,730 / 2,715 | 413 / 430 | 74 | 10,852 | 11,944 |
| A | r3 (3409) | 2.2562 | 1,815 | 1,809 | 2.255 / 2.257 | 1.000 | 2,730 / 2,520 | 411 / 418 | 76 | 17,425 | 21,592 |
| B | r3 (3409) | 1.6949 | 2,417 | 2,405 | 1.692 / 1.696 | 1.001 | 2,700 / 2,520 | 445 / 452 | 78 | 13,520 | 17,942 |
| C | r3 (3409) | 1.6950 | 2,416 | 2,406 | 1.693 / 1.696 | 1.001 | 2,700 / 2,520 | 445 / 451 | 78 | 13,520 | 17,940 |
| D | r3 (3409) | 2.2158 | 1,849 | 1,840 | 2.215 / 2.217 | 1.000 | 2,730 / 2,715 | 414 / 430 | 74 | 10,852 | 11,944 |

### T2. Per arm over rounds (median of the per-cell medians; range = (max-min)/median across rounds)

| arm | rounds | median step s | tok/s | range across rounds | vs A step time | SM clock median MHz (by round) | peak allocated MiB | peak reserved MiB (by round) |
|---|---|---:|---:|---:|---:|---|---:|---|
| A | 1,2,3 | 2.2562 | 1,815 | 0.1% | +0.00% | 2,715, 2,730, 2,730 | 17,425 | 21,592, 21,592, 21,592 |
| B | 1,2,3 | 1.6949 | 2,417 | 0.1% | -24.88% | 2,700, 2,700, 2,700 | 13,520 | 17,942, 17,942, 17,942 |
| C | 1,2,3 | 1.6950 | 2,416 | 0.1% | -24.87% | 2,700, 2,700, 2,700 | 13,520 | 17,996, 17,940, 17,940 |
| D | 1,2,3 | 2.2161 | 1,848 | 0.0% | -1.78% | 2,730, 2,730, 2,730 | 10,852 | 11,944, 11,944, 11,944 |

### L1. Loss against arm A of the same seed (tolerance: step-1 <= 0.5%, last-16 mean <= 1.0% per seed and in the mean)

| arm | round (seed) | step-1 loss | A step-1 | delta | last-16 mean | A last-16 | delta | same batches as A |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | r1 (3407) | 1.5053 | 1.5053 | +0.000% | 1.0931 | 1.0931 | +0.000% | yes |
| A | r2 (3408) | 1.3934 | 1.3934 | +0.000% | 1.0187 | 1.0187 | +0.000% | yes |
| A | r3 (3409) | 1.4804 | 1.4804 | +0.000% | 1.0690 | 1.0690 | +0.000% | yes |
| B | r1 (3407) | 1.5050 | 1.5053 | -0.017% | 1.0939 | 1.0931 | +0.075% | yes |
| B | r2 (3408) | 1.3908 | 1.3934 | -0.185% | 1.0191 | 1.0187 | +0.033% | yes |
| B | r3 (3409) | 1.4790 | 1.4804 | -0.093% | 1.0688 | 1.0690 | -0.017% | yes |
| C | r1 (3407) | 1.5050 | 1.5053 | -0.017% | 1.0936 | 1.0931 | +0.046% | yes |
| C | r2 (3408) | 1.3908 | 1.3934 | -0.185% | 1.0193 | 1.0187 | +0.052% | yes |
| C | r3 (3409) | 1.4790 | 1.4804 | -0.093% | 1.0689 | 1.0690 | -0.013% | yes |
| D | r1 (3407) | 1.5051 | 1.5053 | -0.011% | 1.0932 | 1.0931 | +0.017% | yes |
| D | r2 (3408) | 1.3922 | 1.3934 | -0.086% | 1.0190 | 1.0187 | +0.023% | yes |
| D | r3 (3409) | 1.4785 | 1.4804 | -0.123% | 1.0687 | 1.0690 | -0.028% | yes |

| arm | max abs step-1 delta | per-seed last-16 deltas | mean last-16 delta | inside tolerance |
|---|---:|---|---:|---|
| B | 0.185% | +0.075%, +0.033%, -0.017% | +0.030% | YES |
| C | 0.185% | +0.046%, +0.052%, -0.013% | +0.029% | YES |
| D | 0.123% | +0.017%, +0.023%, -0.028% | +0.004% | YES |

Non-determinism floor (same arm, same seed, P cell against T cell, steps 1-9): A-r1: step-1 0.000%, max over 9 steps 0.075%; A-r2: step-1 0.000%, max over 9 steps 0.150%; B-r1: step-1 0.000%, max over 9 steps 0.192%; B-r2: step-1 0.000%, max over 9 steps 0.198%; C-r1: step-1 0.000%, max over 9 steps 0.113%; C-r2: step-1 0.000%, max over 9 steps 0.137%; D-r1: step-1 0.000%, max over 9 steps 0.129%; D-r2: step-1 0.000%, max over 9 steps 0.135%

### P1. GPU time per step by category (ms; mean of the 3 profiled steps; mean over the P cells of the arm)

| category | A (n=2) | B (n=2) | C (n=2) | D (n=2) |
|---|---:|---:|---:|---:|
| base-weight GEMM | 929.6 | 935.9 | 935.8 | 928.3 |
| LoRA GEMM | 83.7 | 89.2 | 89.2 | 83.6 |
| lm_head GEMM | 50.7 | 51.0 | 50.9 | 84.9 |
| NF4 dequant | 52.5 | 52.8 | 52.8 | 52.5 |
| attention | 22.4 | 22.4 | 22.4 | 22.4 |
| cross-entropy + logits-side | 80.1 | 76.6 | 76.6 | 225.4 |
| RMSNorm/RoPE/SwiGLU/elementwise/casts | 915.8 | 395.2 | 395.3 | 704.8 |
| optimizer + clip | 2.0 | 2.0 | 2.0 | 2.0 |
| memcpy / memset (D2D) | 99.6 | 47.7 | 47.7 | 92.0 |
| GPU busy | 2,236.2 | 1,672.6 | 1,672.6 | 2,195.8 |
| step wall (profiled) | 2,269.5 | 1,705.0 | 1,704.9 | 2,227.5 |
|   of which elementwise ops with an fp32 input | 837.0 | 107.2 | 107.3 | 633.5 |
|   of which elementwise ops with no fp32 input | 78.8 | 288.0 | 288.0 | 71.4 |
| Soup rewind recorder (row_losses) | 9.9 | 9.9 | 9.9 | 0.0 |

### P2. What each arm removed (profile cells, ms per step), against the issue's 818 ms elementwise gap

| arm | elementwise bucket ms | removed vs A ms | share of the issue's 818 ms gap | fp32-input elementwise ms | removed vs A | CE + logits ms | removed vs A | GPU busy ms | removed vs A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 915.8 | 0.0 | 0% | 837.0 | 0.0 | 80.1 | 0.0 | 2,236.2 | 0.0 |
| B | 395.2 | 520.6 | 64% | 107.2 | 729.8 | 76.6 | 3.5 | 1,672.6 | 563.6 |
| C | 395.3 | 520.5 | 64% | 107.3 | 729.7 | 76.6 | 3.5 | 1,672.6 | 563.7 |
| D | 704.8 | 210.9 | 26% | 633.5 | 203.5 | 225.4 | -145.3 | 2,195.8 | 40.4 |

### P3. Memory: per-phase allocator peaks of the P cells (MiB, max over steps 3-9) and parameter dtypes at trainer construction

| arm (cell) | after load | pre_step | pre_fwd | fwd_body | head_ce | backward | optimizer | step peak (max phase) | snapshot peak estimate | blocks >= 1,000 MiB at the snapshot peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-r1-P | 8,423 | 7,633 | 7,633 | 10,350 | 17,421 | 17,425 | 7,787 | 17,425 | 17,425 | 2,376, 2,376, 2,376, 1,040 |
| A-r2-P | 8,423 | 7,633 | 7,633 | 10,350 | 17,421 | 17,425 | 7,787 | 17,425 | 17,425 | 2,376, 2,376, 2,376, 1,040 |
| B-r1-P | 8,423 | 5,552 | 5,552 | 6,984 | 13,516 | 13,520 | 5,706 | 13,520 | 13,520 | 2,376, 2,376, 2,376 |
| B-r2-P | 8,423 | 5,552 | 5,552 | 6,984 | 13,516 | 13,520 | 5,706 | 13,520 | 13,520 | 2,376, 2,376, 2,376 |
| C-r1-P | 5,459 | 5,552 | 5,552 | 6,984 | 13,516 | 13,520 | 5,706 | 13,520 | 13,520 | 2,376, 2,376, 2,376 |
| C-r2-P | 5,459 | 5,552 | 5,552 | 6,984 | 13,516 | 13,520 | 5,706 | 13,520 | 13,520 | 2,376, 2,376, 2,376 |
| D-r1-P | 8,423 | 7,633 | 7,633 | 10,350 | 10,447 | 10,852 | 7,787 | 10,852 | 10,851 | none |
| D-r2-P | 8,423 | 7,633 | 7,633 | 10,350 | 10,447 | 10,852 | 7,787 | 10,852 | 10,851 | none |

### P4. Profiler overhead and agreement with the T cells (seconds per step)

| P cell | unprofiled steps 2-5 mean | profiled steps 6-8 mean | overhead | T cell median (same arm and round) | steps 2-5 mean vs T median |
|---|---:|---:|---:|---:|---:|
| A-r1-P | 2.2607 | 2.2695 | +0.39% | 2.2582 | +0.11% |
| A-r2-P | 2.2609 | 2.2692 | +0.37% | 2.2562 | +0.21% |
| B-r1-P | 1.6972 | 1.7064 | +0.54% | 1.6965 | +0.04% |
| B-r2-P | 1.6949 | 1.7033 | +0.50% | 1.6946 | +0.02% |
| C-r1-P | 1.6957 | 1.7051 | +0.55% | 1.6965 | -0.04% |
| C-r2-P | 1.6961 | 1.7042 | +0.48% | 1.6944 | +0.10% |
| D-r1-P | 2.2215 | 2.2285 | +0.32% | 2.2163 | +0.23% |
| D-r2-P | 2.2186 | 2.2260 | +0.33% | 2.2161 | +0.11% |

Parameter bytes by group and dtype (MiB) at trainer construction, first P cell of each arm:

* A (A-r1-P): embed_tokens [float32] 2,079.0; lm_head [float32] 2,079.0; lora_AB [bfloat16] 77.0; norm [float32] 0.8; bias [float32] 0.5
* B (B-r1-P): embed_tokens [bfloat16] 1,039.5; lm_head [bfloat16] 1,039.5; lora_AB [bfloat16] 77.0; bias [float32] 0.5; norm [bfloat16] 0.4
* C (C-r1-P): embed_tokens [bfloat16] 1,039.5; lm_head [bfloat16] 1,039.5; lora_AB [bfloat16] 77.0; norm [bfloat16] 0.4; bias [bfloat16] 0.2
* D (D-r1-P): embed_tokens [float32] 2,079.0; lm_head [float32] 2,079.0; lora_AB [bfloat16] 77.0; norm [float32] 0.8; bias [float32] 0.5
