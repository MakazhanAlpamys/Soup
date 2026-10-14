# E4 probe, part 1: how widespread is the double-quant absmax defect?

Card: [E4] Проверить влияние double-quant NF4 на качество. Follows B3 (PR #2, `benchmarks/gate-b3-fp8-source.md`, G3).

## 1. Rule

Agreed with @nurikw3 on 2026-10-12 and committed here before any download or run. The text in 1.1 is the agreed rule, verbatim. Section 1.2 only pins definitions the agreed text leaves open; it was written at commit time, with no data in hand.

### 1.1 Agreed rule (verbatim)

**Provenance.** The rule was supplied as a separate file, `e4-rule-part1.md`, which is not kept in this repository. Its title and status lines, verbatim:

> `# E4 rule, part 1: how widespread is the double-quant absmax defect?`
>
> `Status: DRAFT, 2026-10-08. Committed to the repo before the first run; nothing below has been measured yet.`

| | |
|---|---|
| Git blob hash of the supplied file (`git hash-object`) | `f7059463b3349d311198c4a2d5194521f931cab2` |
| Source commit of the rule in this record | `df3dae1` |
| Source path | `benchmarks/probe-e4-double-quant-prevalence.md` (blob `f196bcf10d7d063c29ef4ef487f32e96054b5a8a` at that commit) |
| SHA-256 of the rule body | `6a0053d36568a025112a451ad875db6faba737047e98b6622577596b39ff6aa5` |

The rule body is the text from "What B3 established" through "Deliverable". The supplied file's body and the body committed in `df3dae1` are byte-identical. Transform applied below, and nothing else: every heading of the body is demoted two levels (`##` to `####`) so that it nests under 1.1. `tests/test_e4_double_quant_prevalence.py` promotes the headings back and asserts byte equality with the body in `df3dae1`.

<!-- e4-rule-body:begin -->
#### What B3 established, and what it did not

- Established, on ONE tensor (`model.layers.1.mlp.gate_proj` of DeepSeek-V3 file 1/163): with bitsandbytes double-quant, ~6.4% of NF4 blocks exceed `0.17 x absmax`; worst 2.15; mechanism is small block absmax values stored sign-flipped when a 256-block group spans a wide range (345x there). Whole-tensor relative error 0.0872 -> 0.0880.
- In that one file, 123 of 126 FP8 weights failed the bound with double-quant, 0 without.
- Not established: how common this is across the model, whether Kimi K2 shows it, and whether it affects training quality.

Part 1 answers the first two questions. It also picks the model for part 2 (training quality), which has its own rule, committed before any training run.

#### Sample (fixed before any download)

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

#### Per tensor, both double-quant settings

Reference `r`: FP8 decoded with B3's independent decoder (bit-field table, fp32 multiply, one bf16 rounding). Each tensor's decode is also checked against B3's `dequantize_fp8_blockwise`; any mismatch voids the tensor.

Then `quantize_4bit` / `dequantize_4bit` (NF4, blocksize 64), double-quant off and on, CPU.

| Metric | Definition |
|---|---|
| `frac_over` | share of 64-element blocks with `max|n - r| > 0.17 x max|r|` |
| `worst` | max over blocks of `max|n - r| / max|r|` |
| `frac_flipped` | share of blocks whose stored absmax (after double-quant decode) is <= 0 while the true absmax is > 0 |
| `rel_fro` | `||n - r|| / ||r||` |
| `group_spread` | per 256-block group, max/min true absmax; reported as the tensor's median and max |

#### Questions and how each is answered

**Q1, prevalence.** Per model and per tensor kind (attention, dense MLP, shared expert, routed expert): median, p90 and max of `frac_over` and `frac_flipped`, double-quant on vs off. Descriptive; no threshold.

**Q2, mechanism.** Is `frac_flipped` near zero wherever `group_spread` is small? Reported as a table of `frac_flipped` by `group_spread` bucket. Descriptive.

**Q3, cost of the alternative.** Storage saved by double-quant, computed from bitsandbytes' layout, not measured: fp32 absmax per 64 weights = 0.5 bit/param without; uint8 per 64 + fp32 per 256 blocks = 0.127 bit/param with; saving 0.373 bit/param, about 31 GB on 671B params. Labelled COMPUTED.

**Q4, model for part 2 (decides part 2's design).** The same metrics on candidate single-GPU models: Mistral-7B-v0.1, Qwen2.5-7B, OLMoE-1B-7B (bf16 sources, no FP8 step). Pick the candidate whose median `frac_over` (double-quant on, all linear weights) is closest to DeepSeek-V3's. If every candidate's median is below one tenth of DeepSeek-V3's, the defect does not reproduce at small scale: part 2 is not run as planned, and the record says so and goes to the owner.

#### Validity

- V1: every tensor's fetched byte count equals its header's `data_offsets` span; otherwise void.
- V2: B3's decoder check (above) per tensor.
- V3: box stamped (CPU, RAM, library versions, tree). These are numerical results; timing is not a goal and is not reported as a result.
- V4: a failed fetch, OOM or killed run is kept as void and reported, not silently retried.
- Sampled tensors are reported as a sample: no statement about "all weights" is made from them.

#### Deliverable

`benchmarks/probe-e4-double-quant-prevalence.md` with this rule verbatim, the selected keys, raw per-tensor JSON for both models and the candidates, the void list, and the Q1 to Q4 answers.
<!-- e4-rule-body:end -->

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

One machine for all five runs: a MacBook Air (Apple M5, model `Mac17,3`, 10 cores, 24 GB RAM), CPU only. The harness stamps one `<model>.box.json` per run; the fields below are identical in all five.

| Field | Value |
|---|---|
| CPU (harness stamp) | `unknown` |
| Logical CPUs | 10 |
| torch threads | 4 |
| Platform | `macOS-26.5-arm64-arm-64bit` |
| Python | 3.12.12 |
| torch | 2.14.1 |
| bitsandbytes | 0.50.2 |
| Tree | `afdb84ee` |

The harness does not read the CPU name or the RAM size on macOS, so the stamp says `unknown` and has no RAM field. The chip, model identifier and RAM above come from `sysctl` (`machdep.cpu.brand_string`, `hw.model`, `hw.memsize`) on the same machine, read after the runs. The raw box files are left as the harness wrote them.

| Run | Started (UTC, from the box stamp) |
|---|---|
| DeepSeek-V3 | 2026-10-13T04:03:29Z |
| Kimi K2 | 2026-10-13T04:32:21Z |
| Mistral-7B-v0.1 | 2026-10-13T04:55:35Z |
| Qwen2.5-7B | 2026-10-13T05:04:28Z |
| OLMoE-1B-7B | 2026-10-13T05:12:34Z |

Runs were sequential, one model at a time, in the order DeepSeek-V3, Kimi K2, Mistral-7B-v0.1, Qwen2.5-7B, OLMoE-1B-7B. No run was interrupted or resumed.

## 3. Results

All figures are from `results/e4-double-quant-prevalence/summary.json`, which the harness's `summarize` step builds from the per-tensor rows in `<model>.jsonl`.

### Voids

No `*.void.jsonl` file was written for any model: every selected tensor ended with a row, with no V1 byte-count mismatch, no V2 decoder mismatch and no crashed or killed run. No row has a non-finite value with double-quant on.

**V4 was not met at the transport level in these runs.** The main runs fetched through `_get`, which retried a failed HTTP request up to three times without logging it. Zero `*.void.jsonl` files show that every selected tensor ended with a row. They do not show that no request failed, and V4's "not silently retried" was not met at the transport level in these runs. V1 confirms each row's fetched byte count equals its header span. V1 does not verify byte content. The records are kept unchanged. From commit `06b8f24` on, transport retries are removed and every attempt is journaled (`<model>.attempts.jsonl`); `summarize` reports these five runs as `provenance: no per-attempt log` and gives no count of failed requests for them.

Whether a `*.void.jsonl` ever existed, checked on 2026-10-13: for the five main runs, no. There is none on the run machine's disk or in any git object, the five run logs each start at `0 done` and contain no `VOID` line, and each model was run once. For the smoke runs made before the handoff on another machine (3 DeepSeek-V3 and 3 OLMoE tensors, not part of this record): unknown, that directory is not available.

| Model | Selected | Rows | Voids |
|---|---:|---:|---:|
| DeepSeek-V3 | 850 | 850 | 0 |
| Kimi K2 | 848 | 848 | 0 |
| Mistral-7B-v0.1 | 224 | 224 | 0 |
| Qwen2.5-7B | 196 | 196 | 0 |
| OLMoE-1B-7B | 3136 | 3136 | 0 |

### Q1, prevalence

Each cell is median / p90 / max over the tensors of that kind. DQ = double-quant. `frac_flipped` with DQ off is 0 for every tensor of every model (by construction, section 1.2) and is not given a column.

**DeepSeek-V3** (850 tensors)

| Kind | Tensors | `frac_over`, DQ off | `frac_over`, DQ on | `frac_flipped`, DQ on |
|---|---:|---|---|---|
| attention | 310 | 0 / 0 / 0 | 2.25e-05 / 0.000923 / 0.316 | 0 / 0 / 3.39e-06 |
| dense MLP | 9 | 0 / 0 / 0 | 0.0248 / 0.181 / 0.181 | 0 / 0.000185 / 0.000185 |
| shared expert | 177 | 0 / 0 / 0 | 0.000183 / 0.00154 / 0.0144 | 0 / 0 / 0 |
| routed expert | 354 | 0 / 0 / 0 | 0 / 4.8e-05 / 0.157 | 0 / 0 / 0.00187 |
| all | 850 | 0 / 0 / 0 | 8.6e-06 / 0.000959 / 0.316 | 0 / 0 / 0.00187 |

**Kimi K2** (848 tensors)

| Kind | Tensors | `frac_over`, DQ off | `frac_over`, DQ on | `frac_flipped`, DQ on |
|---|---:|---|---|---|
| attention | 305 | 0 / 0 / 0 | 3.1e-05 / 0.000687 / 0.0835 | 0 / 0 / 1.53e-05 |
| dense MLP | 3 | 0 / 0 / 0 | 0.00028 / 0.000286 / 0.000286 | 0 / 0 / 0 |
| shared expert | 180 | 0 / 0 / 0 | 3.49e-05 / 0.000401 / 0.00111 | 0 / 0 / 0 |
| routed expert | 360 | 0 / 0 / 0 | 0 / 0 / 8.72e-06 | 0 / 0 / 0 |
| all | 848 | 0 / 0 / 0 | 0 / 0.000231 / 0.0835 | 0 / 0 / 1.53e-05 |

**Mistral-7B-v0.1** (224 tensors)

| Kind | Tensors | `frac_over`, DQ off | `frac_over`, DQ on | `frac_flipped`, DQ on |
|---|---:|---|---|---|
| attention | 128 | 0 / 0 / 0 | 8.01e-05 / 0.000343 / 0.00354 | 0 / 0 / 0 |
| dense MLP | 96 | 0 / 0 / 0 | 3.54e-05 / 0.000128 / 0.000189 | 0 / 0 / 0 |
| all | 224 | 0 / 0 / 0 | 6.1e-05 / 0.000214 / 0.00354 | 0 / 0 / 0 |

**Qwen2.5-7B** (196 tensors)

| Kind | Tensors | `frac_over`, DQ off | `frac_over`, DQ on | `frac_flipped`, DQ on |
|---|---:|---|---|---|
| attention | 112 | 0 / 0 / 0 | 0.000209 / 0.00101 / 0.00223 | 0 / 0 / 0 |
| dense MLP | 84 | 0 / 0 / 0 | 9.94e-05 / 0.000469 / 0.0524 | 0 / 0 / 5.09e-05 |
| all | 196 | 0 / 0 / 0 | 0.000144 / 0.00101 / 0.0524 | 0 / 0 / 5.09e-05 |

**OLMoE-1B-7B** (3136 tensors)

| Kind | Tensors | `frac_over`, DQ off | `frac_over`, DQ on | `frac_flipped`, DQ on |
|---|---:|---|---|---|
| attention | 64 | 0 / 0 / 0 | 0.0048 / 0.139 / 0.347 | 0 / 0.0104 / 0.017 |
| routed expert | 3072 | 0 / 0 / 0 | 0 / 6.1e-05 / 0.0239 | 0 / 0 / 0.00397 |
| all | 3136 | 0 / 0 / 0 | 0 / 9.16e-05 / 0.347 | 0 / 0 / 0.017 |

### Q2, mechanism

Blocks are counted over all tensors of the model, double-quant on, and bucketed by the `group_spread` of the 256-block group they belong to. Shares are of the bucket's blocks.

**DeepSeek-V3**

| `group_spread` | Blocks | Flipped blocks (share) | Over blocks (share) |
|---|---:|---:|---:|
| 1 to 10 | 289,736,960 | 0 (0) | 806,033 (0.00278) |
| 10 to 100 | 25,482,752 | 0 (0) | 438,994 (0.0172) |
| 100 to 1000 | 6,416,640 | 1,283 (0.0002) | 898,608 (0.14) |
| 1000 and above | 0 | 0 (no blocks) | 0 (no blocks) |

**Kimi K2**

| `group_spread` | Blocks | Flipped blocks (share) | Over blocks (share) |
|---|---:|---:|---:|
| 1 to 10 | 221,853,184 | 0 (0) | 15,855 (7.15e-05) |
| 10 to 100 | 4,547,072 | 0 (0) | 98,328 (0.0216) |
| 100 to 1000 | 37,888 | 2 (5.28e-05) | 5,505 (0.145) |
| 1000 and above | 0 | 0 (no blocks) | 0 (no blocks) |

**Mistral-7B-v0.1**

| `group_spread` | Blocks | Flipped blocks (share) | Over blocks (share) |
|---|---:|---:|---:|
| 1 to 10 | 107,905,024 | 0 (0) | 3,995 (3.7e-05) |
| 10 to 100 | 1,146,880 | 0 (0) | 5,390 (0.0047) |
| 100 to 1000 | 0 | 0 (no blocks) | 0 (no blocks) |
| 1000 and above | 0 | 0 (no blocks) | 0 (no blocks) |

**Qwen2.5-7B**

| `group_spread` | Blocks | Flipped blocks (share) | Over blocks (share) |
|---|---:|---:|---:|
| 1 to 10 | 94,875,136 | 0 (0) | 6,873 (7.24e-05) |
| 10 to 100 | 6,135,296 | 0 (0) | 130,469 (0.0213) |
| 100 to 1000 | 947,200 | 55 (5.81e-05) | 79,140 (0.0836) |
| 1000 and above | 0 | 0 (no blocks) | 0 (no blocks) |

**OLMoE-1B-7B**

| `group_spread` | Blocks | Flipped blocks (share) | Over blocks (share) |
|---|---:|---:|---:|
| 1 to 10 | 103,676,160 | 0 (0) | 43,091 (0.000416) |
| 10 to 100 | 680,192 | 0 (0) | 26,134 (0.0384) |
| 100 to 1000 | 366,336 | 3,757 (0.0103) | 107,132 (0.292) |
| 1000 and above | 134,912 | 6,760 (0.0501) | 29,697 (0.22) |

Reading only these tables: in all five models the number of flipped blocks is exactly 0 in both buckets below a spread of 100, and every flipped block sits in a bucket of spread 100 or more.

### Q3, cost of the alternative (COMPUTED)

COMPUTED from bitsandbytes' layout, not measured. Without double-quant: one fp32 absmax per 64 weights = 32 / 64 = 0.5 bit/param. With double-quant: one uint8 per 64 weights plus one fp32 per 256 blocks = 8 / 64 + 32 / (64 x 256) = 0.127 bit/param. Saving: 0.373 bit/param, about 31 GB on 671B params (671e9 x 0.373 / 8 bytes).

### Q4, model for part 2

DeepSeek-V3's median `frac_over` with double-quant on, over its 850 sampled tensors: **8.598e-06**. One tenth of it: 8.598e-07.

| Candidate | Linear weights | Median `frac_over`, DQ on | Below one tenth of DeepSeek-V3's | Absolute difference from DeepSeek-V3's |
|---|---:|---:|---|---:|
| Mistral-7B-v0.1 | 224 | 6.104e-05 | no | 5.244e-05 |
| Qwen2.5-7B | 196 | 1.438e-04 | no | 1.352e-04 |
| OLMoE-1B-7B | 3136 | 0 | yes | 8.598e-06 |

**Q4 owner resolution (Andi, 2026-10-13).** Under the rule as committed in `df3dae1`, distance is the absolute difference of medians (`frac_over`, double-quant on, all linear weights). The stop clause is false: Mistral-7B-v0.1 (6.104e-05) and Qwen2.5-7B (1.438e-04) are both above one tenth of DeepSeek-V3's median (8.598e-07). **Selected: OLMoE-1B-7B.** Absolute distances: OLMoE 8.598e-06, Mistral 5.244e-05, Qwen 1.352e-04.

*Post-hoc alternative, not a decision:* ratio distance was considered after the data were seen. It would select Mistral-7B-v0.1 (7.1x), and OLMoE has no finite ratio. It is recorded for transparency only.

Part 2 (training quality) has its own preregistration, which is committed before any training run.

For reference, not part of Q4: Kimi K2's median `frac_over` with double-quant on, over its 848 sampled tensors, is 0.

### Extra metric (descriptive)

`frac_absmax_off10` was added to the harness after the 3-tensor smoke test and feeds no decision. Median over all tensors of the model:

| Model | DQ off | DQ on |
|---|---:|---:|
| DeepSeek-V3 | 0 | 0 |
| Kimi K2 | 0 | 0 |
| Mistral-7B-v0.1 | 0 | 0 |
| Qwen2.5-7B | 0 | 0 |
| OLMoE-1B-7B | 0 | 0 |

### OLMoE-1B-7B: extreme `worst` values

**Observation, not a preregistered result.** `worst` is not part of Q1 to Q4 and is not tabulated above. With double-quant on, 27 of the 3136 OLMoE-1B-7B tensors (17 attention weights and 10 routed-expert weights) have `worst` above 10, up to 1.5e30. `worst` is a ratio to the block's true absmax, and these values come from blocks whose true absmax is close to zero: the largest `group_spread` among those tensors is 3.6e32. All values are finite, and with double-quant off the same tensors have `worst` 0.153. The absolute error `e_b` of those blocks was not recorded: the raw rows hold only the ratio, not the block's absmax or error, so it cannot be derived without a new measurement. It is noted here because OLMoE-1B-7B is the model selected for part 2.

### Not covered

- The DeepSeek-V3 and Kimi K2 tensors are a sample (section 1.1). Nothing here is a statement about all weights of either model.
- Training quality is not measured. That is part 2, which has its own rule, committed before any training run.
