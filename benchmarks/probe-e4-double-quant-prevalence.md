# E4 probe, part 1: how widespread is the double-quant absmax defect?

Card: [E4] Проверить влияние double-quant NF4 на качество. Follows B3 (PR #2, `benchmarks/gate-b3-fp8-source.md`, G3).

## 1. Rule

Agreed with @nurikw3 on 2026-10-12 and committed here before any download or run. The text in 1.1 is the agreed rule, verbatim. Section 1.2 only pins definitions the agreed text leaves open; it was written at commit time, with no data in hand.

### 1.1 Agreed rule (verbatim)


## What B3 established, and what it did not

- Established, on ONE tensor (`model.layers.1.mlp.gate_proj` of DeepSeek-V3 file 1/163): with bitsandbytes double-quant, ~6.4% of NF4 blocks exceed `0.17 x absmax`; worst 2.15; mechanism is small block absmax values stored sign-flipped when a 256-block group spans a wide range (345x there). Whole-tensor relative error 0.0872 -> 0.0880.
- In that one file, 123 of 126 FP8 weights failed the bound with double-quant, 0 without.
- Not established: how common this is across the model, whether Kimi K2 shows it, and whether it affects training quality.

Part 1 answers the first two questions. It also picks the model for part 2 (training quality), which has its own rule, committed before any training run.

## Sample (fixed before any download)

Read through HTTP range requests on the official checkpoints, only the bytes of the selected tensors; nothing else is downloaded.

| | DeepSeek-V3 (`deepseek-ai/DeepSeek-V3`) | Kimi K2 (`moonshotai/Kimi-K2-Instruct`) |
|---|---|---|
| Format | e4m3, 128x128 blocks | e4m3, 128x128 blocks |
| Every attention weight | 310 (62 layers incl. MTP) | 305 (61 layers) |
| Every dense MLP weight | 9 | 3 |
| Every shared-expert weight | 177 | 180 |
| Routed experts | 2 per MoE layer, all 3 projections: 354 | 2 per MoE layer, all 3 projections: 360 |
| Total | 850 | 848 |

Routed experts are chosen with `random.Random(20261008)`, independently per layer, before any tensor is read; the selected keys are written to the record first. The index revision (commit hash of each repo) is recorded.

## Per tensor, both double-quant settings

Reference `r`: FP8 decoded with B3's independent decoder (bit-field table, fp32 multiply, one bf16 rounding). Each tensor's decode is also checked against B3's `dequantize_fp8_blockwise`; any mismatch voids the tensor.

Then `quantize_4bit` / `dequantize_4bit` (NF4, blocksize 64), double-quant off and on, CPU.

| Metric | Definition |
|---|---|
| `frac_over` | share of 64-element blocks with `max|n - r| > 0.17 x max|r|` |
| `worst` | max over blocks of `max|n - r| / max|r|` |
| `frac_flipped` | share of blocks whose stored absmax (after double-quant decode) is <= 0 while the true absmax is > 0 |
| `rel_fro` | `||n - r|| / ||r||` |
| `group_spread` | per 256-block group, max/min true absmax; reported as the tensor's median and max |

## Questions and how each is answered

**Q1, prevalence.** Per model and per tensor kind (attention, dense MLP, shared expert, routed expert): median, p90 and max of `frac_over` and `frac_flipped`, double-quant on vs off. Descriptive; no threshold.

**Q2, mechanism.** Is `frac_flipped` near zero wherever `group_spread` is small? Reported as a table of `frac_flipped` by `group_spread` bucket. Descriptive.

**Q3, cost of the alternative.** Storage saved by double-quant, computed from bitsandbytes' layout, not measured: fp32 absmax per 64 weights = 0.5 bit/param without; uint8 per 64 + fp32 per 256 blocks = 0.127 bit/param with; saving 0.373 bit/param, about 31 GB on 671B params. Labelled COMPUTED.

**Q4, model for part 2 (decides part 2's design).** The same metrics on candidate single-GPU models: Mistral-7B-v0.1, Qwen2.5-7B, OLMoE-1B-7B (bf16 sources, no FP8 step). Pick the candidate whose median `frac_over` (double-quant on, all linear weights) is closest to DeepSeek-V3's. If every candidate's median is below one tenth of DeepSeek-V3's, the defect does not reproduce at small scale: part 2 is not run as planned, and the record says so and goes to the owner.

## Validity

- V1: every tensor's fetched byte count equals its header's `data_offsets` span; otherwise void.
- V2: B3's decoder check (above) per tensor.
- V3: box stamped (CPU, RAM, library versions, tree). These are numerical results; timing is not a goal and is not reported as a result.
- V4: a failed fetch, OOM or killed run is kept as void and reported, not silently retried.
- Sampled tensors are reported as a sample: no statement about "all weights" is made from them.

## Deliverable

`benchmarks/probe-e4-double-quant-prevalence.md` with this rule verbatim, the selected keys, raw per-tensor JSON for both models and the candidates, the void list, and the Q1 to Q4 answers.

### 1.2 Definitions pinned at commit time (before any run)

**Revisions.** Every tensor is read from these exact repo revisions:

| Repo | Revision |
|---|---|
| `deepseek-ai/DeepSeek-V3` | `e815299b0bcbac849fa540c768ef21845365c9eb` |
| `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` |
| `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` |
| `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |

**Expert draw.** One `random.Random(20261008)` per model. MoE layers are visited in ascending index (DeepSeek-V3: 3 to 61, the MTP layer 61 included; Kimi K2: 1 to 60), and each draws `rng.sample(range(n_routed_experts), 2)`. The drawn keys are written to `results/e4-double-quant-prevalence/selected-keys.json` before any tensor is read.

**Tensor kinds.** attention = every `*.self_attn.*.weight` stored as FP8; dense MLP = `mlp.{gate,up,down}_proj` of dense layers; shared expert = `mlp.shared_experts.*`; routed expert = `mlp.experts.<i>.*`. Router `mlp.gate.weight`, norms, embeddings and heads are not FP8 and are excluded.

**Candidate linear weights (Q4).** Every 2-D `*.weight` inside `model.layers.*` of the candidate, except norms and MoE router gates. Embeddings and `lm_head` are excluded. bf16 as stored, no FP8 step.

**Medians.** A "median" is over tensors, unweighted, within the stated set. For Q4 the DeepSeek-V3 comparator is the median over its 850 sampled tensors.

**Double-quant off and `frac_flipped`.** Without double-quant the stored absmax is the fp32 true value, so `frac_flipped` is 0 by construction; it is still computed and reported.

**Codec.** bitsandbytes `quantize_4bit` / `dequantize_4bit`, `quant_type="nf4"`, `blocksize=64`, `compress_statistics` False/True, on CPU, same library versions for every model (stamped).

## 2. Box

_Filled from the harness JSON after the runs._

## 3. Results

_Filled after the runs. Void runs and void tensors are listed, not dropped._
