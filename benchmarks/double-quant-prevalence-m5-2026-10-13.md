# Measurement record - NF4 double-quantization prevalence

Measured by @Lameda12 on 2026-10-13 UTC.

## 1. Question and setup

How often does NF4 with double quantization exceed a block's error bound
when the source is DeepSeek-V3 or Kimi K2 FP8? The survey also checks whether
scale sign flips cluster at high within-group spread, computes the storage
trade-off, and selects a smaller checkpoint for a subsequent LoRA quality study.

All five runs were CPU-only on one Apple M5 MacBook Air (Mac17,3, 10 logical
CPUs, 24 GB RAM), with four PyTorch threads. The box stamps record
`macOS-26.5-arm64-arm-64bit`, Python 3.12.12, PyTorch 2.14.1, bitsandbytes
0.50.2 and tree `afdb84ee`. Chip, machine model and RAM were read with
`sysctl` after the runs. GPU driver and CUDA were inapplicable.

| Checkpoint | Revision | Source dtype | Selected / measured / void |
|---|---|---|---:|
| `deepseek-ai/DeepSeek-V3` | `e815299b0bcbac849fa540c768ef21845365c9eb` | FP8 E4M3, 128×128 blocks | 850 / 850 / 0 |
| `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | FP8 E4M3, 128×128 blocks | 848 / 848 / 0 |
| `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | BF16 | 224 / 224 / 0 |
| `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` | BF16 | 196 / 196 / 0 |
| `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | BF16 | 3136 / 3136 / 0 |

## 2. Method and commands

The rule fixed the sample before downloading tensors. It took all FP8
attention, dense MLP and shared-expert weights and all three projections of
two routed experts per MoE layer. It sampled experts with one
`random.Random(20261008)` per model, visiting layers in ascending order and
including DeepSeek-V3's multi-token prediction layer. The BF16 candidates
contributed all decoder-layer 2-D linear weights except norms and router
gates. Embeddings and output heads were excluded.

The harness fetched exact safetensors byte ranges and checked the byte counts.
It decoded FP8 independently with a bit-field table, FP32 scale multiplication
and one BF16 rounding, then checked against the blockwise decoder. The resulting
BF16 tensors and the stored BF16 candidate weights were the references.
Bitsandbytes quantized and dequantized NF4 in 64-weight blocks with
`compress_statistics` false and true.

`frac_over` is the share of a tensor's blocks with max absolute error greater
than `0.17 ×` the reference block's maximum absolute weight. `frac_flipped`
is the share with decoded absmax at or below zero despite a positive reference
absmax. Group spread is the ratio of largest to smallest true absmax within a
256-block scale group. The survey reports unweighted per-tensor statistics and
aggregates block counts by group spread.

The harness lives at the evidence commit. The commands show the recorded model
order with an output directory substituted. The later harness journals HTTP
attempts.

```bash
python benchmarks/harness/e4_double_quant_prevalence.py select --out <out>
python benchmarks/harness/e4_double_quant_prevalence.py run --model deepseek-v3 --out <out>
python benchmarks/harness/e4_double_quant_prevalence.py run --model kimi-k2 --out <out>
python benchmarks/harness/e4_double_quant_prevalence.py run --model mistral-7b --out <out>
python benchmarks/harness/e4_double_quant_prevalence.py run --model qwen2.5-7b --out <out>
python benchmarks/harness/e4_double_quant_prevalence.py run --model olmoe-1b-7b --out <out>
python benchmarks/harness/e4_double_quant_prevalence.py summarize --out <out>
```

## 3. Results

The prevalence cells give median / p90 / max fractions from the source record's
tables derived from `summary.json`. With double quantization off,
`frac_flipped` is zero by construction for every row.

| Model | Tensor kind | Tensors | `frac_over`, off | `frac_over`, on | `frac_flipped`, on |
|---|---|---:|---|---|---|
| DeepSeek-V3 | attention | 310 | 0 / 0 / 0 | 2.25e-05 / 0.000923 / 0.316 | 0 / 0 / 3.39e-06 |
| DeepSeek-V3 | dense MLP | 9 | 0 / 0 / 0 | 0.0248 / 0.181 / 0.181 | 0 / 0.000185 / 0.000185 |
| DeepSeek-V3 | shared expert | 177 | 0 / 0 / 0 | 0.000183 / 0.00154 / 0.0144 | 0 / 0 / 0 |
| DeepSeek-V3 | routed expert | 354 | 0 / 0 / 0 | 0 / 4.8e-05 / 0.157 | 0 / 0 / 0.00187 |
| DeepSeek-V3 | all | 850 | 0 / 0 / 0 | 8.6e-06 / 0.000959 / 0.316 | 0 / 0 / 0.00187 |
| Kimi K2 | attention | 305 | 0 / 0 / 0 | 3.1e-05 / 0.000687 / 0.0835 | 0 / 0 / 1.53e-05 |
| Kimi K2 | dense MLP | 3 | 0 / 0 / 0 | 0.00028 / 0.000286 / 0.000286 | 0 / 0 / 0 |
| Kimi K2 | shared expert | 180 | 0 / 0 / 0 | 3.49e-05 / 0.000401 / 0.00111 | 0 / 0 / 0 |
| Kimi K2 | routed expert | 360 | 0 / 0 / 0 | 0 / 0 / 8.72e-06 | 0 / 0 / 0 |
| Kimi K2 | all | 848 | 0 / 0 / 0 | 0 / 0.000231 / 0.0835 | 0 / 0 / 1.53e-05 |
| Mistral-7B-v0.1 | attention | 128 | 0 / 0 / 0 | 8.01e-05 / 0.000343 / 0.00354 | 0 / 0 / 0 |
| Mistral-7B-v0.1 | dense MLP | 96 | 0 / 0 / 0 | 3.54e-05 / 0.000128 / 0.000189 | 0 / 0 / 0 |
| Mistral-7B-v0.1 | all | 224 | 0 / 0 / 0 | 6.1e-05 / 0.000214 / 0.00354 | 0 / 0 / 0 |
| Qwen2.5-7B | attention | 112 | 0 / 0 / 0 | 0.000209 / 0.00101 / 0.00223 | 0 / 0 / 0 |
| Qwen2.5-7B | dense MLP | 84 | 0 / 0 / 0 | 9.94e-05 / 0.000469 / 0.0524 | 0 / 0 / 5.09e-05 |
| Qwen2.5-7B | all | 196 | 0 / 0 / 0 | 0.000144 / 0.00101 / 0.0524 | 0 / 0 / 5.09e-05 |
| OLMoE-1B-7B | attention | 64 | 0 / 0 / 0 | 0.0048 / 0.139 / 0.347 | 0 / 0.0104 / 0.017 |
| OLMoE-1B-7B | routed expert | 3072 | 0 / 0 / 0 | 0 / 6.1e-05 / 0.0239 | 0 / 0 / 0.00397 |
| OLMoE-1B-7B | all | 3136 | 0 / 0 / 0 | 0 / 9.16e-05 / 0.347 | 0 / 0 / 0.017 |

With double quantization on, every sign flip was in a group with spread at
least 100. The mechanism table gives flipped blocks / total blocks from the
same summary.

| Model | [1, 10) | [10, 100) | [100, 1000) | ≥1000 |
|---|---:|---:|---:|---:|
| DeepSeek-V3 | 0 / 289,736,960 | 0 / 25,482,752 | 1,283 / 6,416,640 | 0 / 0 |
| Kimi K2 | 0 / 221,853,184 | 0 / 4,547,072 | 2 / 37,888 | 0 / 0 |
| Mistral-7B-v0.1 | 0 / 107,905,024 | 0 / 1,146,880 | 0 / 0 | 0 / 0 |
| Qwen2.5-7B | 0 / 94,875,136 | 0 / 6,135,296 | 55 / 947,200 | 0 / 0 |
| OLMoE-1B-7B | 0 / 103,676,160 | 0 / 680,192 | 3,757 / 366,336 | 6,760 / 134,912 |

**Computed scale storage.** FP32 absmax per 64 weights costs 0.5 bit/parameter.
An 8-bit scale per block and one FP32 scale per 256 blocks cost 0.127
bit/parameter. The computed saving is 0.373 bit/parameter, about 31 GB on
671B parameters.

**Follow-up selection.** The rule minimized the absolute difference from
DeepSeek-V3's full-sample median of `8.598e-06` with double quantization on.
The stop condition required every candidate to fall below one tenth of that
median (`8.598e-07`). Mistral and Qwen exceeded the stop threshold.

| Candidate | Median `frac_over`, on | Absolute difference from DeepSeek-V3 |
|---|---:|---:|
| Mistral-7B-v0.1 | 6.104e-05 | 5.244e-05 |
| Qwen2.5-7B | 1.438e-04 | 1.352e-04 |
| OLMoE-1B-7B | 0 | 8.598e-06 |

**Verdict — positive. Double quantization produced bound violations in all five samples, and the selection rule chose OLMoE-1B-7B for follow-up.**

## 4. Limitations and superseded runs

The giant-model results describe sampled tensors. The survey measured weight
reconstruction and left the effect on training quality open. DeepSeek-V3 and
Kimi K2 include tensors with substantial violation fractions even though their
all-tensor medians are small. On OLMoE, 27 tensors had a worst block error ratio
above 10, up to `1.5e30` against near-zero reference absmax. The per-tensor
rows lack the block magnitudes needed to recover those absolute errors.

Each selected tensor produced one finite result with double quantization on.
The historical fetcher allowed unlogged HTTP retries. The runs lack the
required transport audit, and the failed-request count is unknown.
Byte counts establish the expected span length. No full survey
run was superseded. The earlier three-tensor smoke checks for each of
DeepSeek-V3 and OLMoE are outside the five-run dataset.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
