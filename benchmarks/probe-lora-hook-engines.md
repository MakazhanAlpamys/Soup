<!--
Working measurement record. §2, the decision rule, was written and committed
before any engine run; results are appended below it as they arrive and the
rule is not edited afterwards.

Box for Part A: Windows 11 Pro 26200, AMD Ryzen 5 8645HS (6 cores / 12 threads),
15.3 GB RAM, CPU only. The box has an RTX 4050 Laptop GPU; this probe does not
use it.
-->

# Probe — does a MoE serving engine apply a Soup LoRA adapter to MLA attention and the shared expert?

**Status: Part A measured 2026-10-08, three runs. Run 3 gives the verdict
HOOK NEEDED: llama.cpp b11476 silently drops a LoRA on `q_a_proj`, `q_b_proj`
and `kv_a_proj_with_mqa`, and cannot convert one on `kv_b_proj`; `o_proj` and
the shared expert are applied exactly. The positive control (Qwen3.5-shaped)
passed. Runs 1 and 2 were void or without verdict, for instrument reasons
recorded in §3, and are kept. Run 2 also found that PEFT, on Soup's pinned
stack, cannot build the shared-expert adapter at all (§3.2). Part A', a draft
hook of 18 added and 13 removed lines, makes every one of those modules APPLIED
on the same tiny models: SUFFICIENT under its own rule (§4a-§4b). Part B0, a
SYNTHETIC adapter on the real DeepSeek-V3 and Kimi K2 configs: with the hook's
converter it exports from the config alone, every factor pair carries the
adapter's rank on both halves, and every layer's `k_b` and `v_b` pairs have the
outer dimensions the config gives the base tensors; the other pairs' outer
dimensions are not checked. The stock converter fails on `kv_b_proj` (§4d-§4e).
Part B, the real Qwen3.5-35B-A3B on a cloud CPU box: run 1, bf16 on both
sides, is VOID (the two bf16 base models differ by 6.6% against a 5% line);
run 2, f32 on both sides, records both adapter variants APPLIED, `r` < 1.5e-5
with a base gap of 1.3e-6 (§4e-§4h). Qwen3.5 has no MLA: this measures the
standard attention and shared-expert paths at real scale. The historical
real-model runs did not repeat adapted logits; that determinism control is
unmeasured, so they do not establish that every registered control passed.**

This record answers one question behind the serving-scope decision in
[`scope-moe-serving.md`](scope-moe-serving.md): if an external engine serves the
giant MoE and Soup only trains, how much has to be added to that engine so that
a Soup adapter works there, unmerged?

---

## 0. The question, and why it decides the serving scope

For a giant MoE (DeepSeek-V3 671B, Kimi K2 ~1T) Soup adapts the attention
projections and, optionally, the shared expert; the routed experts stay frozen.
Soup's `target_modules: auto` policy for `deepseek_v3`, `kimi_k2` and
`kimi_k25` is exactly the five MLA projections `q_a_proj`, `q_b_proj`,
`kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`
([`peft_wiring.py:81-87,126,139,144`](../src/soup_cli/utils/peft_wiring.py)).

Merging the adapter into the base is not a way around the engine. `soup export`
merges through a full fp16 load of the base on the CPU
([`export.py:409-421`](../src/soup_cli/commands/export.py)), about 1.3 TB for
671B parameters. So the engine has to load the adapter as an adapter.

Reading llama.cpp at tag `b11476` (commit `988190680d5a`) says it would not,
for the DeepSeek-V3 architecture:

- the MLA query and compressed-KV projections are plain `ggml_mul_mat` calls,
  not the LoRA-aware `build_lora_mm`:
  [`deepseek2.cpp` L502, L508, L518](https://github.com/ggml-org/llama.cpp/blob/b11476/src/models/deepseek2.cpp#L496-L520);
- the absorbed key projection `wk_b` is a plain `ggml_mul_mat`
  ([L561](https://github.com/ggml-org/llama.cpp/blob/b11476/src/models/deepseek2.cpp#L556-L566)),
  and so is the value projection `wv_b` inside the attention helper
  ([`llama-graph.cpp` L2744-L2754, L2811-L2812](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-graph.cpp#L2744-L2812));
- the converter splits `kv_b_proj` into `k_b` and `v_b` and transposes `k_b`
  ([`conversion/deepseek.py` L434-L449](https://github.com/ggml-org/llama.cpp/blob/b11476/conversion/deepseek.py#L434-L449)),
  and the adapter converter's tensor wrapper supports that transpose only in 2-D
  ([`convert_lora_to_gguf.py` L168-L179](https://github.com/ggml-org/llama.cpp/blob/b11476/convert_lora_to_gguf.py#L168-L179));
- `o_proj` goes through `build_lora_mm` in the attention output helper, and the
  shared expert through `build_ffn`, which uses it
  ([`llama-graph.cpp` L1837-L1977, L2896](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-graph.cpp#L1837-L1977)).

Code reading is not execution, and a dropped adapter is silent: the engine loads
it and answers. This probe runs the path end to end on tiny models and names,
tensor by tensor, what a hook has to add.

## 1. The instrument

**Models — SYNTHETIC weights.** Random initialisation with a fixed seed; the
shapes follow the real architectures, scaled down. No result here is a
statement about a trained model.

| label | transformers class | layers | attention | experts | vocabulary / tokenizer |
|---|---|---|---|---|---|
| `dsv3-tiny` | `DeepseekV3ForCausalLM` | 3 (layer 0 dense, 1-2 MoE) | MLA, 4 heads, `q_lora_rank` 32, `kv_lora_rank` 16, `qk_nope` 16, `qk_rope` 8, `v_head` 16 | 8 routed, top-2, 1 shared; sigmoid scoring, `n_group` 1 as in Kimi K2 | 129,280, tokenizer of `deepseek-ai/DeepSeek-V3@e815299b` |
| `qwen35moe-tiny` | `Qwen3_5MoeForCausalLM` | 4 (3 Gated DeltaNet, 1 full attention) | full: 4 heads, 2 KV heads, head dim 64, output gate | 8 routed, top-2, 1 shared with gate | 248,320, tokenizer of `Qwen/Qwen3.5-35B-A3B@59d61f3c` |

Deliberate deviations from the real configs, each orthogonal to the adapter
path: no YaRN rope scaling, no multi-token-prediction layer, and
`dsv3-tiny` routes without expert groups (DeepSeek-V3 uses `n_group` 8,
Kimi K2 uses 1).

**Adapters — SYNTHETIC.** PEFT LoRA, `r` 8, `lora_alpha` 16, dropout 0.
`lora_A` keeps PEFT's initialisation; `lora_B` is drawn from N(0, 0.02) with a
seeded generator, because PEFT's zero `lora_B` would make every adapter a no-op
and every engine trivially correct. One adapter per target set:

| model | variant | target modules |
|---|---|---|
| `dsv3-tiny` | `soup-auto` | `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` (Soup's policy) |
| `dsv3-tiny` | `all` | `soup-auto` + `shared_experts.{gate,up,down}_proj` |
| `dsv3-tiny` | `all-but-kv_b` | `all` without `kv_b_proj` |
| `dsv3-tiny` | one per module | each of the five MLA projections alone; `shared` = the three shared-expert projections |
| `qwen35moe-tiny` | `soup-auto` | `q_proj`, `v_proj`, `in_proj_qkv`, `out_proj` (Soup's policy, [`peft_wiring.py:25-30`](../src/soup_cli/utils/peft_wiring.py)) |
| `qwen35moe-tiny` | `all` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `shared_expert.{gate,up,down}_proj` |
| `qwen35moe-tiny` | one per module | each module of `all` alone; `shared` = the three shared-expert projections |

**Reference.** transformers + PEFT, fp32, CPU.

**Engine.** llama.cpp `b11476`: the official Windows CPU build
(`llama-b11476-bin-win-cpu-x64.zip`, sha256 `a23e548c…0100e5483`), base and
adapter converted to f32 GGUF by `convert_hf_to_gguf.py` and
`convert_lora_to_gguf.py` from the same tag, run in llama.cpp's own pinned
converter environment. Logits come from `llama-results`, which writes the
prompt's token ids and the logits of every position to a GGUF file
([`tools/results/results.cpp`](https://github.com/ggml-org/llama.cpp/blob/b11476/tools/results/results.cpp)).
The reference consumes the engine's token ids, so both sides see identical
tokens. Adapter off = no `--lora`; adapter on = `--lora <adapter.gguf>`.

**Strata** (`Niko1221/Strata`) is not run. At `d5ea7133` its README names one
model family, Qwen3.8-Flash-Next, and requires an NVIDIA or AMD card with at
least 12 GB of VRAM
([README L12, L58-L61](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/README.md#L58-L61)),
so it can load neither tiny model on this CPU-only box.

## 2. The decision rule, written before any run

For each model and adapter variant, over every prompt position and the whole
vocabulary, with `z_ref⁰`, `z_ref¹` the reference logits without and with the
adapter and `z_eng⁰`, `z_eng¹` the engine's:

- `e_base = max|z_eng⁰ − z_ref⁰| / max|z_ref⁰|`
- `Δ_ref = z_ref¹ − z_ref⁰`, `Δ_eng = z_eng¹ − z_eng⁰`
- `s_ref = ‖Δ_ref‖ / ‖z_ref⁰‖` (Frobenius norms), the adapter's effect on the reference
- `ρ = ‖Δ_eng‖ / ‖Δ_ref‖`, how much of that effect the engine reproduces
- `r = ‖Δ_eng − Δ_ref‖ / ‖Δ_ref‖`, how wrong the engine's effect is

The verdict is the first row that matches:

| condition | verdict |
|---|---|
| token ids differ, or `e_base > 1e-3` | **VOID**: the base model itself does not match, so nothing about adapters can be read. Re-run the model with the next seed, at most three seeds; every void attempt is kept |
| `s_ref < 1e-2` | **TOO WEAK**: redraw `lora_B` once with 4x the standard deviation, then report whatever it gives |
| `convert_lora_to_gguf.py` exits non-zero | **CONVERT-FAILED** |
| `llama-results` exits non-zero with the adapter but not without it | **LOAD-FAILED** |
| `r ≤ 1e-2` | **APPLIED** |
| `ρ ≤ 1e-2` | **DROPPED**: the adapter loads and changes nothing. This is the negative control: with the adapter off the engine's output must differ from its output with the adapter on |
| otherwise | **WRONG**: the adapter changes the output, but not the way the reference does |

**Why these thresholds.** Both sides run f32 on the same CPU, so base
differences come only from kernel accumulation order and should sit orders of
magnitude below `1e-3`. A dropped adapter gives `ρ = 0` and `r = 1`, and a
scale error of even 10% gives `r = 0.1`. The APPLIED and DROPPED bands are each
at least 10x from the other outcomes.

**Controls.**

- *Positive control.* On `qwen35moe-tiny` the variants `q_proj`, `v_proj` and
  `shared` go through llama.cpp's LoRA helper and must be APPLIED. If any of
  them is not, the instrument is broken and the probe has **NO VERDICT** on
  `dsv3-tiny`.
- *Reference sanity.* PEFT with the adapter disabled must reproduce `z_ref⁰`
  bit for bit.
- *Determinism.* One variant per model runs twice on each side; logits must be
  bitwise identical, otherwise that model is VOID.

**Expected from the code reading in §0, stated before the run so the run can
contradict it.**

- `dsv3-tiny`: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` DROPPED; `kv_b_proj`,
  `soup-auto` and `all` CONVERT-FAILED; `o_proj` and `shared` APPLIED;
  `all-but-kv_b` WRONG.
- `qwen35moe-tiny`: every variant APPLIED.

**What the result decides.** The hook's content is every `dsv3-tiny` module
whose single-module variant is not APPLIED, together with the converter or
graph code path that drops it. If that list is empty, Soup adapters on MLA need
no engine change at all. If the positive control fails, nothing is concluded.

## 3. Results

Captured results, including VOID runs, are in
[`results/probe-lora-hook-engines/`](results/probe-lora-hook-engines/).
The JSON records failed-step stderr, binary SHA-256 values, Python environments,
harness fingerprints and logit-array digests. Numerical thresholds were fixed
before collection.

### 3.1 Run 1: VOID on both models

- `qwen35moe-tiny`: the base model did not convert at any of the three seeds.
  `convert_hf_to_gguf.py` asserts that a Qwen3.5 config declares a
  multi-token-prediction block unless it is called with `--no-mtp`
  ([`conversion/qwen.py` L298-L305](https://github.com/ggml-org/llama.cpp/blob/b11476/conversion/qwen.py#L298-L305)),
  and the tiny model has none.
- `dsv3-tiny`: `e_base` = 0.0099, 0.0089 and 0.0110 at seeds 17, 18 and 19,
  above the `1e-3` line, so VOID by the rule. Runs 2 and 3 use explicit
  `"scoring_func": "sigmoid"` and `"topk_method": "noaux_tc"`, as in the real
  DeepSeek-V3 config, and give `e_base` 5.2e-7 (§3.2–§3.3).

The measured f32 configuration uses those DeepSeek routing keys, `--no-mtp`
for Qwen3.5 base conversion, and `-fa off -ctk f32 -ctv f32` for the engine.

### 3.2 Run 2: NO VERDICT, and a PEFT finding

Both bases were valid (`e_base` 4.8e-7 and 5.2e-7). Every Qwen3.5 adapter
conversion failed the converter's MTP-config assertion, so no positive-control
verdict was available. The successful configuration declares
`mtp_num_hidden_layers: 1`, matching the real config; Transformers constructs
no MTP block, so this does not add base-model weights.

The `dsv3-tiny` variant `shared` failed to convert with "Unprocessed experts",
and its adapter file shows why: it held LoRA factors for
`model.layers.{1,2}.mlp.experts` (the routed experts) and none for
`mlp.shared_experts`, although the targets were
`shared_experts.{gate,up,down}_proj`. **This is a finding about the training
side, not the engine.** In peft 0.21.2, `_convert_peft_config_moe`
(`peft/utils/transformers_weight_conversion.py`, L444-L482) rewrites every LoRA
target that equals or ends in `gate_proj`, `up_proj` or `down_proj` into
`target_parameters` `gate_up_proj` / `down_proj`, the fused parameters of the
routed experts, for every model type whose transformers v5 conversion pattern is
`qwen2_moe`. transformers 5.19.0 maps `deepseek_v3`, `deepseek_v2`, `qwen3_moe`,
`qwen3_next`, `olmoe`, `glm4_moe` and `glm_moe_dsa` to that pattern. A target
prefix does not protect the shared expert, and a regex target is resolved to
the same leaf names first (L355-L399). So on Soup's pinned stack an "attention +
shared expert" adapter for DeepSeek-V3 silently becomes "attention + every
routed expert", an unrequested target expansion whether or not it fits in memory. The
same code is in peft 0.20.0, Soup's floor (read from the published wheel, not
run). Soup's own `target_modules: auto` for `deepseek_v3` is attention only and
is not affected.

**Upstream status, inspected 2026-10-08.** The same failure is reported in
[PEFT issue #3711](https://github.com/huggingface/peft/issues/3711), which
remains open. Maintainer BenjaminBossan's
[PR #3715](https://github.com/huggingface/peft/pull/3715) is open and unmerged
at head `be54773d7ac4ee7ce3f889846354ffdb8985d464` [DOC].
The [changed source and tests](https://api.github.com/repos/huggingface/peft/pulls/3715/files)
were read, not run here [CODE]:

- `_convert_peft_config_moe` preserves a module target only when it matches at
  least one module and **all** matching modules are `nn.Linear`.
- The DeepSeek-V3 tests include qualified `shared_experts.down_proj` and
  `mlp.shared_experts.{gate,down}_proj` targets. Without explicit routed-expert
  `target_parameters`, they expect shared-expert LoRA and no routed-expert
  wrapper; with explicit parameter targets, only those are added.
- This is not a general fix for every mixed, fused or layer-specific target:
  xfailed tests remain for broad targets mixing dense modules with expert
  parameters, loss of legacy layer scope, and unrepresentable selections of
  fused experts. A fully qualified shared-expert list and a broad regex must
  not be assumed equivalent after conversion.

The latest [release is 0.21.2, published 2026-10-01](https://github.com/huggingface/peft/releases/tag/v0.21.2)
[DOC]; its [conversion source](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/utils/transformers_weight_conversion.py#L401-L488)
does not contain the PR's linear-module preservation check [CODE].
Soup declares `peft>=0.20.0,<1.0.0`
([`pyproject.toml`](../pyproject.toml)) [CODE]; this is a dependency range,
not a fixed installed version.

**Required training-side guard (prototype P3), not measured here.** At
configuration load, before loading checkpoint weights and before
`get_peft_model`, construct the real config's model on `meta`, as in
[`meta_model`](harness/lora_export_real_configs.py). Apply Soup's normal
target resolution, preserve the requested `LoraConfig`, and run
`convert_peft_config_for_transformers` on a copy: it mutates the config in
place [CODE]. Compare the exact tensor paths selected by the original
request and by the converted module/parameter targets. Refuse any mismatch
with the requested and actual names, including shared-expert or dense-MLP
targets redirected to `mlp.experts.*`. A nonempty `target_parameters` is not
itself an error: the same mapping converts a `gate` target to the router's
`gate.weight`, not a routed expert. Checking only after `get_peft_model`
can be too late, because unintended routed-expert adapters allocate before
inspection. Memory-budget checking alone cannot establish intended placement;
both checks may share a single meta-model traversal [INFERENCE]. Raise
Soup's PEFT floor to the actual release containing #3715 when it is published,
and retain the path guard afterwards.

Instrument change after run 2: the harness switches that rewrite off while it
builds and loads adapters (`targets_as_named`), so the engine is tested on the
adapter Soup's policy intends, and every variant now records whether its file
touches the routed experts.

The `dsv3-tiny` single-module rows of run 2 already read as in run 3, but they
carry no verdict: the positive control had failed.

### 3.3 Run 3: the verdict

Both bases valid at seed 17: `e_base` 4.8e-7 (`qwen35moe-tiny`) and 5.2e-7
(`dsv3-tiny`), 22 prompt tokens, engine and reference bitwise deterministic.
For all 21 variants PEFT with the adapter disabled reproduced the base bit for
bit, the engine saw the same token ids, and no adapter touched a routed expert.

`qwen35moe-tiny`, the positive control:

| variant | modules adapted | `s_ref` | `ρ` | `r` | verdict |
|---|---|---|---|---|---|
| `soup-auto` | 8 | 0.226 | 1.000 | 1.8e-6 | APPLIED |
| `all` | 31 | 0.305 | 1.000 | 1.3e-6 | APPLIED |
| `q_proj` | 1 | 0.121 | 1.000 | 2.8e-6 | APPLIED |
| `k_proj` | 1 | 0.124 | 1.000 | 2.7e-6 | APPLIED |
| `v_proj` | 1 | 0.215 | 1.000 | 1.5e-6 | APPLIED |
| `o_proj` | 1 | 0.115 | 1.000 | 2.2e-6 | APPLIED |
| `in_proj_qkv` | 3 | 0.076 | 1.000 | 5.3e-6 | APPLIED |
| `in_proj_z` | 3 | 0.041 | 1.000 | 1.0e-5 | APPLIED |
| `in_proj_a` | 3 | 1.5e-4 | 1.000 | 2.6e-3 | TOO WEAK |
| `in_proj_b` | 3 | 0.013 | 1.000 | 3.1e-5 | APPLIED |
| `out_proj` | 3 | 0.042 | 1.000 | 9.4e-6 | APPLIED |
| `shared` | 12 | 0.072 | 1.000 | 5.5e-6 | APPLIED |

`in_proj_a` feeds the per-head decay of the Gated DeltaNet and moves the logits
by 1.5e-4 even with `lora_B` at 4x; the engine reproduces that effect too
(`ρ` 0.9999), but the rule calls it TOO WEAK and it stays so. The positive
control (`q_proj`, `v_proj`, `shared`) is APPLIED, so the instrument reads.

`dsv3-tiny`:

| variant | modules adapted | `s_ref` | `ρ` | `r` | verdict | expected (§2) |
|---|---|---|---|---|---|---|
| `q_a_proj` | 3 | 0.022 | 0 | 1.000 | DROPPED | DROPPED |
| `q_b_proj` | 3 | 0.025 | 0 | 1.000 | DROPPED | DROPPED |
| `kv_a_proj_with_mqa` | 3 | 0.241 | 0 | 1.000 | DROPPED | DROPPED |
| `kv_b_proj` | 3 | 0.468 | - | - | CONVERT-FAILED | CONVERT-FAILED |
| `o_proj` | 3 | 0.205 | 1.000 | 1.5e-6 | APPLIED | APPLIED |
| `shared` | 6 | 0.105 | 1.000 | 2.6e-6 | APPLIED | APPLIED |
| `soup-auto` | 15 | 0.559 | - | - | CONVERT-FAILED | CONVERT-FAILED |
| `all` | 21 | 0.450 | - | - | CONVERT-FAILED | CONVERT-FAILED |
| `all-but-kv_b` | 18 | 0.283 | 0.808 | 0.719 | WRONG | WRONG |

`ρ` is exactly 0 for the three dropped projections: the engine's logits with the
adapter are bit-identical to its logits without it, while the reference moves
by 2-24%. `kv_b_proj` fails with `NotImplementedError` in the 3-D transpose of
`k_b` (`convert_lora_to_gguf.py` L179, reached from `conversion/deepseek.py`
L446). `all-but-kv_b` is the case a user would actually hit if the converter
skipped `kv_b`: the server loads the adapter, the output changes, and 72% of
the change is wrong.

**Verdict under §2: HOOK NEEDED**, for exactly the four modules §2 predicted
from the code reading: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` (DROPPED)
and `kv_b_proj` (CONVERT-FAILED).

## 4. What the hook has to cover

llama.cpp `b11476`; Kimi K2 runs through the same `deepseek2` graph
(`DeepseekV3ForCausalLM`), so the list is the same for it.

| Soup module | GGUF tensor | today | change needed |
|---|---|---|---|
| `q_a_proj` | `blk.N.attn_q_a` | plain `ggml_mul_mat`, [`deepseek2.cpp` L502](https://github.com/ggml-org/llama.cpp/blob/b11476/src/models/deepseek2.cpp#L502); MTP path L266 | call `build_lora_mm` instead |
| `q_b_proj` | `blk.N.attn_q_b` | plain, L508; MTP L272 | same |
| `kv_a_proj_with_mqa` | `blk.N.attn_kv_a_mqa` | plain, L518; MTP L288 | same |
| `kv_b_proj` | `blk.N.attn_k_b`, `blk.N.attn_v_b` | the converter cannot transpose the per-head `k_b` factor in 3-D; the absorbed `wk_b` (L561; MTP L330) and `wv_b` (attention helper, [`llama-graph.cpp` L2744-L2754 and L2811](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-graph.cpp#L2744-L2812), with and without flash attention) are plain matmuls | converter: split `lora_B` per head into K and V parts and swap the factor roles for the transposed `k_b`; graph: low-rank terms at `wk_b` and both `wv_b` sites |
| `o_proj` | `blk.N.attn_output` | APPLIED | none |
| `shared_experts.*` | `blk.N.ffn_{gate,up,down}_shexp` | APPLIED | none |

By reading, not by running: the adapter loader's shape check compares only the
first two dimensions and the rank
([`llama-adapter.cpp` L358-L371](https://github.com/ggml-org/llama.cpp/blob/b11476/src/llama-adapter.cpp#L358-L371)),
so per-head 3-D factors for `k_b` and `v_b` would pass it unchanged. Run 3
exercised the path without flash attention; the flash-attention `wv_b` site is
the same plain matmul by reading, not by measurement.

Outside llama.cpp, two more pieces are needed: Soup has to export an adapter on
its own (`soup export` merges today), and PEFT has to be stopped from rewriting
shared-expert targets (§3.2) before a shared-expert adapter can be trained at
all.

## 4a. Part A': is a small hook enough? Rule written before the run

A draft hook, written against `b11476` after run 3: 3 files, 18 lines added and
13 removed (`convert_lora_to_gguf.py` +8/-3, `src/models/deepseek2.cpp` +8/-8,
`src/llama-graph.cpp` +2/-2). It routes `wq_a`, `wq_b`, `wkv_a_mqa` and the
absorbed `wk_b` through `build_lora_mm` in the main and MTP graphs, does the
same for `wv_b` at both attention-helper sites, and teaches the adapter
converter the batched transpose of the last two dimensions that `k_b` needs:
`(B·A)ᵀ = Aᵀ·Bᵀ` per head, so the two factors swap roles. The patch under test
is fixed by its SHA-256,
`80851e0bcb5ac775e7918799333eb6e0210948132e9f8d18238caec0d5fe1c09`, and is not
edited between the control and the patched run.

Both builds come from the same local toolchain (MinGW-w64 GCC 15.2.0, CMake
4.4.4, Ninja, CPU only, `GGML_NATIVE=ON`) out of the same `b11476` checkout, and
run through the unchanged harness with §2's rule.

| condition | outcome |
|---|---|
| the stock local build does not reproduce run 3's verdict for every variant of both models | **NO VERDICT**: the local build changes behaviour, so nothing about the patch can be read |
| with the patched build and converter, every `dsv3-tiny` variant (the five MLA projections, `shared`, `soup-auto`, `all`, `all-but-kv_b`) is APPLIED and every `qwen35moe-tiny` verdict equals run 3's | **SUFFICIENT**: this hook alone serves the adapter shape on these tiny models |
| otherwise | **INSUFFICIENT**: the variants that are not APPLIED name what is still missing |

A SUFFICIENT draft says nothing about upstream acceptance, quantised base
tensors, the GPU backends or a real checkpoint; it measures whether the change
is a few dozen lines or a project.

## 4b. Part A': results

Both local builds report `version: 0.6.0-dev (build 1, commit 9881906)`, GNU
15.2.0; the harness is unchanged from run 3 (fingerprint `b0b17e1ea00dcbc4`).
The patch's SHA-256 was checked again before the patched run and matched.

- **Control, stock local build:** every verdict of both models equals run 3's,
  with `e_base` 4.8e-7 (`qwen35moe-tiny`) and 4.5e-7 (`dsv3-tiny`). The local
  toolchain does not change behaviour.
- **Patched build and converter:** every `dsv3-tiny` variant is APPLIED:
  `q_a_proj` r = 1.4e-5, `q_b_proj` 1.2e-5, `kv_a_proj_with_mqa` 1.4e-6,
  `kv_b_proj` 7.4e-7, `o_proj` 1.5e-6, `shared` 2.6e-6, `soup-auto` 6.6e-7,
  `all` 9.2e-7, `all-but-kv_b` 1.2e-6, all with `ρ` = 1.000. Every
  `qwen35moe-tiny` verdict equals run 3's.

**Verdict under §4a: SUFFICIENT.** On these tiny models the four failures of
run 3 are closed by 18 added and 13 removed lines in three files. The patch is
kept as [`mla-lora-hook-draft.patch`](results/probe-lora-hook-engines/mla-lora-hook-draft.patch)
(llama.cpp code, MIT; its licence travels with it in
[`LICENSE-llama.cpp.txt`](results/probe-lora-hook-engines/LICENSE-llama.cpp.txt)).
It is a measurement instrument, not a proposed upstream change: a pull request
would need llama.cpp's own tests, review of the DeepSeek-V2-Lite and legacy
`wkv_b` paths it leaves alone, and the GPU backends.

**An extra arm, decided after the verdict and not part of it.** Parts A and A'
run without flash attention, so the flash-attention `wv_b` site of the patch was
not exercised. The same patched build was run once more on `dsv3-tiny` with
llama.cpp's CPU defaults restored (flash attention, f16 KV cache;
`part-a2-hook-local-fa-on.*`). Base gap 3.0e-4, inside §2's line. `kv_b_proj`
is APPLIED (r 7.0e-4), and so are `kv_a_proj_with_mqa`, `o_proj`, `shared`,
`soup-auto`, `all` and `all-but-kv_b` (r 6.5e-4 to 1.7e-3). `q_a_proj` and
`q_b_proj` come out WRONG by §2's f32 line, at r = 0.012 and 0.011 with
`ρ` = 1.0003 and 1.0004 and a cosine of 0.99993 between the two effects. These
are the two smallest effects in the table (`s_ref` 0.022 and 0.025), and with
f16 storage in the KV cache the base gap is 600 times the f32 arm's, so an
error of about 1% of so small an effect is the size the noise predicts. That
reading is an inference, not a measurement, and the verdicts stand as printed.
The harness's overall line for this arm reads NO VERDICT only because the
positive-control model was not part of it.

## 4c. Part B: the real Qwen3.5-35B-A3B in the cloud. Rule written before the run

Part A's verdicts come from tiny random models. Part B asks whether the same
engine path holds on the real test model, with the adapter exported on its own
and the routed experts served from host RAM. Qwen3.5 has no MLA, so Part B
covers the adapter-only export and llama.cpp's standard adapter paths for
attention and the shared expert at real scale, not the tensors the hook
changes; MLA on real weights is prototype task P4
([`scope-moe-serving-prototype.md`](scope-moe-serving-prototype.md)).

- **Model:** `Qwen/Qwen3.5-35B-A3B@59d61f3c` (bf16, 71.9 GB, Apache-2.0). For
  this config transformers builds `Qwen3_5MoeForCausalLM`, the text tower.
- **Adapters, SYNTHETIC:** Part A's recipe (PEFT LoRA, `r` 8, `lora_alpha` 16,
  `lora_B` ~ N(0, 0.02) from a seeded generator, one 4x redraw if `s_ref` <
  0.01). Two variants: `soup-auto`, Soup's Qwen3.5 policy (`q_proj`, `v_proj`,
  `in_proj_qkv`, `out_proj`), and `soup-auto+shared`, which adds
  `shared_expert.{gate,up,down}_proj`.
- **Export:** adapter only, through `convert_lora_to_gguf.py --base <checkpoint>`,
  which reads the base config and not its weights; f32.
- **Engine:** llama.cpp `b11476`, built from the tag on the instance (Linux,
  CPU). Base converted with `--outtype bf16 --no-mtp`, weights memory-mapped
  from the local disk, so the routed experts are served from host RAM;
  `-fa off -ctk f32 -ctv f32`.
- **Reference:** transformers 5.19.0 + PEFT 0.21.2 on the CPU in bf16. An fp32
  copy (about 140 GB) does not fit in the box's 128 GB.
- **Box:** NVIDIA Brev, GCP `n2d-highmem-16` (16 vCPU, 128 GB RAM, no GPU),
  disk of at least 300 GB.
- **Harness:** [`lora_hook_real_model.py`](harness/lora_hook_real_model.py),
  which reuses Part A's engine runner and metrics.

Both sides run bf16, so §2's f32 lines do not apply. The tolerance is set by
how much the two base models disagree:

- `n_base = ‖z_eng⁰ − z_ref⁰‖ / ‖z_ref⁰‖`, and top-1 agreement over positions;
- the floor `f = ‖z_eng⁰ − z_ref⁰‖ / ‖Δ_ref‖`, the base gap in units of the
  adapter's effect, and the tolerance `t = 3f + 0.02`.

| condition | verdict |
|---|---|
| top-1 agreement < 0.9, or `n_base` > 0.05, or either side not deterministic | **VOID** |
| `s_ref` < 0.01 after the redraw | **TOO WEAK** |
| the converter fails / the engine refuses the adapter | **CONVERT-FAILED** / **LOAD-FAILED** |
| `t` > 0.5 | **TOO NOISY**: the allowed effect-error band exceeds the conservative 0.5 limit |
| `r` ≤ `t` | **APPLIED** |
| `ρ` ≤ 0.1 | **DROPPED** |
| otherwise | **WRONG** |

Why `3f`: an engine that applies the adapter correctly still differs from the
reference by about one base gap with the adapter and one without, so its error
on the effect is about `2f`. `3f + 0.02` leaves room for that. A dropped
adapter (`r` = 1) cannot pass while `t ≤ 0.5`; a half-applied adapter
(`r` ≈ 0.5) can pass at the upper boundary, not below it.
Expected from Part A: both variants APPLIED. The preregistered inequalities
and thresholds are unchanged.

The stage's cloud budget is $10. The instance is deleted after the run and
the record gives the instance type, versions, wall times and the cost
(hours times list price, plus disk). If the reference cannot finish inside the
budget, Part B records how the engine takes the adapter (steps, memory,
versions) and correctness stays with Part A.

## 4d. Part B0: exporting the adapter alone for the real configs. Rule written before the run

Soup's export has to work on a laptop for a 671B or 1T base whose weights are
not on it. Part B0 checks the converter half of that, at real dimensions, with
a directory that holds only the model's `config.json`.

- **Configs:** `Qwen/Qwen3.5-35B-A3B@59d61f3c`, `deepseek-ai/DeepSeek-V3@e815299b`,
  `moonshotai/Kimi-K2-Instruct@fd1984e2`.
- **Adapters, SYNTHETIC:** factors drawn from a seeded generator for every layer
  of Soup's `target_modules: auto` plus the shared expert, `r` 8. Shapes come
  from a meta-device model of each config (no weights): 200 modules for
  Qwen3.5, 479 for DeepSeek-V3, 485 for Kimi K2.
- **Converters:** the stock `b11476` tree, and the Part A' hook tree (patch
  SHA-256 `80851e0b…`), both run in the same converter environment.
- **Harness:** [`lora_export_real_configs.py`](harness/lora_export_real_configs.py).

| condition | verdict |
|---|---|
| the converter exits 0; every `lora_a` has its `lora_b` and both carry rank 8; and for the `deepseek2` configs, every layer has `attn_k_b` and `attn_v_b` pairs whose outer dimensions match the base tensors (`k_b`: ne0 = `qk_nope_head_dim`, ne1 = `kv_lora_rank`; `v_b`: ne0 = `kv_lora_rank`, ne1 = `v_head_dim`), which is llama.cpp's own loader check | **PASS** |
| otherwise | **FAIL** |

Expected: stock, Qwen3.5 PASS and both DeepSeek-family configs FAIL at
`kv_b`; hook, all three PASS. This says whether an adapter-only export works
without the base weights, and whether the converter half of the hook holds at
real dimensions. It does not run the engine: the graph half is measured only
on the tiny models.

## 4e. Part B0 results, and where Part B stands

**Part B0, 2026-10-08.** Runs 2 and 3 agree on every measured verdict and
count. Run 2's fingerprint `fef0de9e9bcad8e4` identifies its imported
`lora_hook_parity.py`, not the export harness. Run 3 records both identities:
`harness_sha256_16` = `f38f10f090c03add` and
`parity_sha256_16` = `fef0de9e9bcad8e4`.

| config | modules | stock `b11476` | hook tree |
|---|---|---|---|
| Qwen3.5-35B-A3B | 200 | PASS, 200 pairs | PASS, 200 pairs |
| DeepSeek-V3 | 479 | FAIL: `NotImplementedError` transposing `k_b` out of `kv_b_proj` | PASS, 540 pairs |
| Kimi K2 | 485 | FAIL: the same | PASS, 546 pairs |

All six as expected. The stock converter fails at `k_b.transpose(1, 2)`
(`conversion/deepseek.py` L446), which its LoRA tensor wrapper does not
implement. With the hook, each `kv_b_proj` becomes a `k_b` and a `v_b` pair, so
the pair counts are the module counts plus one per layer (61 layers in both
configs). Every pair carries rank 8 on both halves, and every layer's `k_b` and
`v_b` pairs have the outer dimensions the config gives the base tensors
(DeepSeek-V3: 128 heads, `kv_lora_rank` 512; Kimi K2: 64 heads, the same rank).
The outer dimensions of the other pairs were not checked: §4d's rule compares
only the two tensors the hook writes. The meta-device model's `in_features` and
`out_features` would have been enough to check the rest; that check was not
added, and llama.cpp's loader checks them when the adapter is
loaded on a real base. With the hook's converter, then, an adapter for a 671B
or 1T base exports on a machine that holds only the base's `config.json`.

One scratch check ran before §4d's rule was committed (`6b3584fa`, 03:35,
UTC+5): at 03:31-03:32 a throwaway script, not committed, had the stock
converter export a single `q_proj` factor pair for layer 3 of the Qwen3.5
config from a directory holding only `config.json`, to see whether an
adapter-only export runs at all. It wrote the pair (`blk.3.attn_q.weight`,
rank 8). It did not touch DeepSeek-V3 or K2, and it is not part of B0's result.

**Part B access.** On 2026-10-07 (UTC) two Brev instances were created and
deleted without ever being reached; on 2026-10-08 a probe of two cheap
instances reached both. Lifecycle, capacity and cost evidence is retained in
[`part-b-access-attempts.evidence.log`](results/probe-lora-hook-engines/part-b-access-attempts.evidence.log).

| date (UTC) | instance | create command | delete command | list price | cost at list price |
|---|---|---|---|---|---|
| 2026-10-07 | GCP `n2d-highmem-16` (16 vCPU, 128 GB RAM); disk size not recorded, probably 129 GB | 22:08:45 | 22:22:10 | $0.72/h, disk $0.16 per GB-month | $0.16, disk about $0.006 at 129 GB |
| 2026-10-07 | AWS `m8a.medium` (1 vCPU, 4 GB RAM), SSH probe; disk size not recorded | 22:23:08 | 22:29:56 | $0.08/h, disk $0.14 per GB-month | $0.009, disk under $0.003 at 129 GB |
| 2026-10-08 | GCP `n2d-highcpu-2` (2 vCPU, 2 GB RAM), SSH probe; disk size not recorded | 04:42:52 | 04:48:11 | $0.05/h, disk $0.16 per GB-month | $0.004, disk about $0.002 at 129 GB |
| 2026-10-08 | Crusoe `c1a.2x` (2 vCPU, 8 GB RAM, 128 GB disk), SSH probe | 04:48:56 | 04:54:12 | $0.10/h, disk included | $0.009 |
| 2026-10-08 | GCP `n2d-highmem-16` (16 vCPU, 128 GB RAM), 129 GB disk; deleted unused (below) | 04:57:19 | 05:05:44 | $0.72/h, disk $0.16 per GB-month | $0.10, disk $0.004 |

About $0.30 in all, counted from each create command to its delete command; the
bill itself was not seen. Disk sizes were looked at only on the 2026-10-08
`n2d-highmem` instances (129 GB each); the rows without a recorded size are
costed at that size [INFERENCE]. The instance Part B runs on is costed with its
results. On 2026-10-07 Brev reported both instances `READY`, but neither was
ever reached, so no driver or software version was recorded and no model was
downloaded. Brev's SSH gateway closed every connection before the
SSH banner (`kex_exchange_identification: Connection closed by remote host`),
and port 22 of both instances timed out. `brev refresh` changed nothing;
`brev enable-ssh` asks for `brev register`, which makes the local machine a Brev
node and does not bear on reaching an instance. Outbound SSH from the same
machine worked (`github.com:22`). On 2026-10-08, from the same machine and CLI
version, `brev exec` and a plain `ssh` reached both probes once Brev showed
`BUILD COMPLETED` and `SHELL READY`, on the provider that had failed (GCP) and
on one not tried before (Crusoe). Nothing was changed locally in between, so
the failure of 2026-10-07 was on Brev's side or transient, not tied to one
provider [INFERENCE].

**Deviation from §4c, recorded before the results.** §4c asks for GCP
`n2d-highmem-16` (128 GB RAM) with at least 300 GB of disk. The same command as
on 2026-10-07, `brev create --type n2d-highmem-16 --min-disk 300`, gave a 129 GB
disk (`/` 125 GB, 118 GB free;
[`part-b-cloud-run.evidence.log`](results/probe-lora-hook-engines/part-b-cloud-run.evidence.log)):
Brev ignores `--min-disk` when `--type` is given
([brev-cli #380](https://github.com/brevdev/brev-cli/issues/380), open; the CLI
here is v0.6.335). The checkpoint (71.9 GB) and its bf16 GGUF (about 70 GB) do
not fit on that disk together, and with 128 GB of RAM the GGUF cannot sit in RAM
next to the bf16 reference (about 70 GB). That instance was deleted unused.
Part B runs on GCP `n2d-highmem-32` (32 vCPU, 256 GB RAM, the same 129 GB disk)
instead, with the checkpoint on disk and the GGUF and every work file on a
RAM-backed tmpfs (`/dev/shm`, 126 GB). The rule, the precision of both sides
(bf16) and the engine flags are as §4c states; the engine's thread count follows
the box (32).

On timing: this paragraph was committed (`c7bb202b`, 05:26:54 UTC) within
seconds of the run's end. The harness was launched between 05:13:29 and
05:13:49 and ran 791 s by its own clock, so it ended between about 05:26:40 and
05:27:05. The deviation itself was decided before the launch: the
`n2d-highmem-16` was deleted at 05:05:44 and the `n2d-highmem-32` created at
05:06:40. The result was first seen at the status check of 05:29:10; the one
before, at 05:24:04, still showed the base conversion running.

## 4f. Part B run 1: VOID

**Run.** 2026-10-08 (UTC), on the box of §4e's deviation: GCP `n2d-highmem-32`
through Brev, AMD EPYC 7B13, 32 vCPU, 251 GiB of RAM, no GPU, Ubuntu 22.04.5,
kernel 6.8.0-1069-gcp. The checkpoint sat on the 129 GB disk; the 69.4 GB bf16
GGUF and the work files on `/dev/shm`. The repository went to the box as
`git archive` of `f4375fa3` with LF line endings, so the fingerprints are those
of the committed files: `fca602f5484081b8` (`lora_hook_real_model.py`) and
`fef0de9e9bcad8e4` (`lora_hook_parity.py`). llama.cpp `b11476` (`98819068`) was
built on the box with GNU 11.4.0. The reference ran torch 2.14.1+cpu,
transformers 5.19.0 and peft 0.21.2 on Python 3.12.15; the converter torch
2.11.0+cpu, transformers 4.57.6 and gguf 0.19.0; 32 threads. The harness took
13.2 minutes, 7.0 of them for the download and 4.4 for the base conversion.
Captured result: `part-b-real-model-run1.json`; source-lined output and runtime
identity: `part-b-real-model-run1.evidence.log`, `part-b-setup-run1.evidence.log`.
Provisioning, copy and teardown:
[`part-b-cloud-run.evidence.log`](results/probe-lora-hook-engines/part-b-cloud-run.evidence.log).

**Result: VOID, at the base check.** Both sides were deterministic (two engine
runs and two reference forwards bitwise equal); the engine tokenised the prompt
the same way in every run (22 tokens), and the reference was given those tokens.
Top-1 agreement was 21 of 22 (0.955, above 0.9), but `n_base` = 0.066 is above
the 0.05 line (`e_base` 0.105), so by §4c's table the run is VOID and neither
variant has a verdict. The per-variant numbers were recorded all the same; the
rule does not read them once the run is VOID:

| variant | `s_ref` | `ρ` | `r` | cos | `f` | `t` = 3f + 0.02 |
|---|---|---|---|---|---|---|
| `soup-auto` | 0.118 | 0.92 | 0.75 | 0.69 | 0.56 | 1.70 |
| `soup-auto+shared` | 0.241 | 0.98 | 0.37 | 0.93 | 0.27 | 0.84 |

The `t` column is derived from the recorded `f`; this capture has no
per-variant `t`. Its `meta.rule` contains Part A constants, while its verdicts
use `verdict_b` and §4c's Part B constants: top-1 ≥ 0.9, `n_base` ≤ 0.05,
`t = 3f + 0.02 ≤ 0.5`, and DROPPED at `ρ ≤ 0.1`. Recomputing `verdict_b`
from the captured numbers reproduces the recorded verdicts.

With a valid base both rows would still be TOO NOISY (`t` > 0.5): the two bf16
base models differ by 27-56% of the adapters' own effect, so in bf16 this model
cannot tell a correct adapter from a wrong one under this rule. Why the two
bf16 implementations differ by 6.6% was not diagnosed. Different bf16 rounding
(transformers keeps bf16 activations through every layer, llama.cpp keeps f32
activations and rounds only the matmul inputs) and the expert-routing flips it
can cause are the likely reasons [HYPOTHESIS].

For both variants `r`/`f` is 1.35 and 1.34, close to √2 ≈ 1.41: what an exactly
applied adapter gives when the base gap with the adapter and the one without it
are independent and of equal size, since then `r` ≈ ‖gap₁ − gap₀‖ / ‖Δ_ref‖ ≈
√2 `f` [INFERENCE]. This is not a verdict: a systematic adapter error of the
order of `f` would hide inside the same noise.

Outside the rule, the run shows two things [RUN], neither of them a correctness
result. Both adapters exported adapter-only (19.8 MB and 29.7 MB of GGUF) and
llama.cpp loaded both onto the real bf16 base, so every factor pair passed the
loader's rank and shape checks against the real base tensors. And the engine's
logits moved under each adapter by about as much as the reference's (`ρ` 0.92
and 0.98), where a dropped adapter gives `ρ` = 0 on this deterministic engine.

| date (UTC) | instance | create command | delete command | list price | cost at list price |
|---|---|---|---|---|---|
| 2026-10-08 | GCP `n2d-highmem-32` (32 vCPU, 256 GB RAM), 129 GB disk | 05:06:40 | 05:32:42 | $1.44/h, disk $0.16 per GB-month | $0.62, disk $0.012 |

With §4e's rows, about $0.94 for the stage so far; the bill itself was not seen.
The archive of the outputs had the same SHA-256 on the box and after the copy,
both printed before the delete command; `brev ls` was empty at 05:34:06. Run 2,
in f32, is in §4g-§4h.

## 4g. Part B run 2: f32 on both sides. Rule written before the run

Run 1 was VOID in bf16 (§4f). Run 2 repeats §4c with one change, the
precision, so that the base gap is no longer two different bf16 roundings.

- **As in §4c:** the model, the SYNTHETIC adapters' recipe and seeds, the redraw
  rule, the adapter-only export, llama.cpp `b11476` with
  `-fa off -ctk f32 -ctv f32`, the prompt, and the verdict table with its
  thresholds: top-1 agreement ≥ 0.9, `n_base` ≤ 0.05, `t` = 3f + 0.02 ≤ 0.5,
  DROPPED at `ρ` ≤ 0.1.
- **Changed:** the base GGUF is converted with `--outtype f32` (about 139 GB),
  and the reference is transformers + PEFT in fp32 (about 139 GB of RAM). Both
  sides hold the checkpoint's bf16 weights, upcast exactly. The harness takes
  this as `--precision f32` and records it in its JSON; its default, bf16, runs
  run 1's command in §5 unchanged.
- **Box:** GCP `n2d-highmem-48` through Brev (48 vCPU, 384 GB RAM, and the
  129 GB disk Brev gives), the checkpoint on disk, the GGUF and the work files
  on `/dev/shm`.
- **Expected [ESTIMATE]:** `n_base` of the order of 1e-5 to 1e-4, and both
  variants APPLIED.
- **Outcomes named in advance:**
  - VOID or TOO NOISY in f32 as well: the gap between the two base models is
    not just bf16 rounding. That is a finding in itself. There is no run 3 in
    this stage, and the comparison's precision goes to P4.
  - WRONG or DROPPED for either variant: a defect in llama.cpp's standard
    adapter paths (attention, shared expert) on the real model, reported at
    once.
  - APPLIED for both: llama.cpp `b11476` applies these two Soup-shaped
    adapters, unmerged, to the real Qwen3.5-35B-A3B in f32 on CPU within the
    rule's tolerance. Not MLA (P4), not quantised bases, not GPU backends.
- **Deadline:** 90 minutes from the create command, a cap of about $3.3
  ($2.17/h plus disk). At the deadline whatever exists on the box is copied off,
  each copy time-limited, and the instance is deleted even if the harness is
  still running; the run is then recorded as incomplete. Two mechanisms hold
  the deadline without the agent's session: a Windows scheduled task on the
  laptop that copies, deletes and logs `brev ls` at the deadline (removed after
  a normal deletion), and a `shutdown` on the instance 10 minutes after the
  deadline, set right after the first successful `brev exec`. Whether Brev stops
  billing for a stopped instance was not checked, so the instance is deleted in
  every case.

## 4h. Part B run 2: APPLIED

**Run.** 2026-10-08 (UTC), under §4g's rule (`c46ab90b`), on GCP
`n2d-highmem-48` through Brev: AMD EPYC 7B13, 48 vCPU, 377 GiB of RAM, no GPU,
the 129 GB disk, `/dev/shm` 189 GB. Harness fingerprints `65161ef16679e79b`
(`lora_hook_real_model.py` at `c46ab90b`) and `fef0de9e9bcad8e4`, with
`"precision": "f32"` in the JSON; llama.cpp, reference and converter versions
as in run 1; 48 threads. The harness took 11.4 minutes, 7.0 of them for the
download and 1.8 for the f32 conversion. Captured result:
`part-b-real-model-run2.json`; source-lined output and runtime identity:
`part-b-real-model-run2.evidence.log`, `part-b-setup-run2.evidence.log`.
Provisioning, copy and teardown:
[`part-b-cloud-run2.evidence.log`](results/probe-lora-hook-engines/part-b-cloud-run2.evidence.log).

**Result: both variants APPLIED.** Both sides were deterministic and read the
same 22 tokens as in run 1. Top-1 agreement 22 of 22; `n_base` = 1.3e-6,
`e_base` 2.4e-6.

| variant | `s_ref` | `ρ` | `r` | cos | `f` | `t` = 3f + 0.02 | verdict |
|---|---|---|---|---|---|---|---|
| `soup-auto` | 0.111 | 1.000 | 1.4e-5 | 1.000 | 1.2e-5 | 0.020 | **APPLIED** |
| `soup-auto+shared` | 0.237 | 1.000 | 8.3e-6 | 1.000 | 5.5e-6 | 0.020 | **APPLIED** |

As in run 1 (§4f), the capture's `meta.rule` contains Part A constants;
its verdicts use `verdict_b` and §4c's Part B constants. Per-variant `t`
is derived from the recorded `f`. Recomputing the rule from those numbers
reproduces the recorded verdicts.

This is §4g's APPLIED outcome: llama.cpp `b11476` applies these two
Soup-shaped adapters, exported adapter-only, unmerged, to the real
Qwen3.5-35B-A3B in f32 on CPU, with `r` about a thousandth of the tolerance. It
covers the standard attention and shared-expert paths only, not MLA (P4),
quantised bases or GPU backends. The base gap came out below the expected 1e-5
to 1e-4 [ESTIMATE]. With the same checkpoint and tokens agreeing to 1.3e-6 in
f32, run 1's 6.6% gap came from the two bf16 implementations, not from a
mismatch in the model or its conversion [INFERENCE]. The two runs share the
adapters' recipe and seeds, not the adapters bit for bit: every `lora_B` is
equal, and every `lora_A` of run 1 is run 2's rounded to bf16 (280 of 280
tensors; per tensor the largest |ΔA| is 2.8e-3 of the largest |A|), because
PEFT casts a new adapter to its base layer's dtype, bf16 in run 1 (peft 0.21.2
`tuners/lora/layer.py:312-314`, `tuners_utils.py:2210`); compared on the runs'
archives, which are kept outside the repository [RUN].

**Deadline.** The create command ran at 06:01:12 and the deadline was 07:30.
The Windows scheduled task for it was armed at 06:01, after a test task had run
WSL and `brev` through the same mechanism. The instance's shutdown was set at
06:08:18 for 07:40:23. The harness finished at about 06:21:30, and the result
was first seen at the status check of 06:22:34. The outputs' archive matched
the box's SHA-256 before the delete command (06:26:44); `brev ls` was empty at
06:29:06 and the scheduled task was removed at 06:29:23. Neither mechanism had
to act.

| date (UTC) | instance | create command | delete command | list price | cost at list price |
|---|---|---|---|---|---|
| 2026-10-08 | GCP `n2d-highmem-48` (48 vCPU, 384 GB RAM), 129 GB disk | 06:01:12 | 06:26:44 | $2.17/h, disk $0.16 per GB-month | $0.92, disk $0.012 |

With the rows of §4e and §4f, about $1.87 for the stage; the bill itself was
not seen.

## 5. Reproducing

Two Python 3.12 environments: the reference one is Soup's
(`pip install -e ".[dev]"`; run 3 used torch 2.14.1+cpu, transformers 5.19.0,
peft 0.21.2), and the converter one is llama.cpp's own at the tag:

```bash
git clone --depth 1 --branch b11476 https://github.com/ggml-org/llama.cpp <llama-src>
python -m venv <convert-venv>
<convert-venv>/bin/python -m pip install -r <llama-src>/requirements/requirements-convert_lora_to_gguf.txt
<convert-venv>/bin/python -m pip install <llama-src>/gguf-py
```

Engine: the release asset `llama-b11476-bin-win-cpu-x64.zip` (sha256
`a23e548c6b3525c38bcfeceaff919786ae06741857043cb670279b70100e5483`) unpacked to
`<llama-bin>`, or any `b11476` build that has `llama-results`. Then, from the
repository root:

```bash
python benchmarks/harness/lora_hook_parity.py \
  --llama-bin <llama-bin> --llama-src <llama-src> \
  --convert-python <convert-venv>/bin/python \
  --work-dir <scratch> --models qwen35moe-tiny,dsv3-tiny --seed 17 --threads 4 \
  --out benchmarks/results/probe-lora-hook-engines/part-a-cpu-run3.json \
  --log benchmarks/results/probe-lora-hook-engines/part-a-cpu-run3.log
```

The tokenizers are fetched from the Hub at the revisions in §1; the models and
adapters are built from seeds, so nothing else is downloaded. Wall time on this
box: about 6.5 minutes.

Run 3 and every original Part A' run used the harness in `4552ce82`
(fingerprint `b0b17e1ea00dcbc4`). `5e1c34a7` adds `convert_base`'s `outtype`
argument with the same f32 default. The current instrument enforces the
controls in §6 and repeats adapted logits; historical captures do not
establish those controls. The historical source is
`git show 4552ce82:benchmarks/harness/lora_hook_parity.py`.
Use fresh work/output paths rather than overwrite retained evidence.

**Part A'.** A second checkout of the same tag with the draft patch applied,
and a CPU-only build of both trees (run here with MinGW-w64 GCC 15.2.0, and
CMake 4.4.4 and Ninja 1.13.2 installed with pip):

```bash
git clone --depth 1 --branch b11476 https://github.com/ggml-org/llama.cpp <hook-src>
git -C <hook-src> apply "$PWD/benchmarks/results/probe-lora-hook-engines/mla-lora-hook-draft.patch"
# once for <llama-src> into <stock-build>, once for <hook-src> into <hook-build>
cmake -S <tree> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build <build> --target llama-results
python benchmarks/harness/lora_hook_parity.py --llama-bin <stock-build>/bin \
  --llama-src <llama-src> --convert-python <convert-venv>/bin/python \
  --work-dir <scratch-stock> --models qwen35moe-tiny,dsv3-tiny --seed 17 --threads 4 \
  --out benchmarks/results/probe-lora-hook-engines/part-a2-stock-local.json \
  --log benchmarks/results/probe-lora-hook-engines/part-a2-stock-local.log
python benchmarks/harness/lora_hook_parity.py --llama-bin <hook-build>/bin \
  --llama-src <hook-src> --convert-python <convert-venv>/bin/python \
  --work-dir <scratch-hook> --models qwen35moe-tiny,dsv3-tiny --seed 17 --threads 4 \
  --out benchmarks/results/probe-lora-hook-engines/part-a2-hook-local.json \
  --log benchmarks/results/probe-lora-hook-engines/part-a2-hook-local.log
```

The flash-attention arm (`part-a2-hook-local-fa-on.*`) is the same harness
with `ENGINE_F32_FLAGS` replaced by `("-fa", "on")`, which also leaves the KV
cache at llama.cpp's default f16. It ran on `dsv3-tiny` only:

```bash
python - --llama-bin <hook-build>/bin --llama-src <hook-src> \
  --convert-python <convert-venv>/bin/python --work-dir <scratch-fa> \
  --models dsv3-tiny --seed 17 --threads 4 \
  --out benchmarks/results/probe-lora-hook-engines/part-a2-hook-local-fa-on.json \
  --log benchmarks/results/probe-lora-hook-engines/part-a2-hook-local-fa-on.log <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "lora_hook_parity", "benchmarks/harness/lora_hook_parity.py")
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)
harness.ENGINE_F32_FLAGS = ("-fa", "on")
raise SystemExit(harness.main(sys.argv[1:]))
EOF
```

**Part B0.** No engine is needed; only the three configs are downloaded. Wall
time on this box: about 100 seconds.

```bash
python benchmarks/harness/lora_export_real_configs.py \
  --convert-python <convert-venv>/bin/python \
  --llama-src stock=<llama-src> --llama-src hook=<hook-src> --work-dir <scratch> \
  --out benchmarks/results/probe-lora-hook-engines/part-b0-export-real-configs-run3.json \
  --log benchmarks/results/probe-lora-hook-engines/part-b0-export-real-configs-run3.log
```

Run 3 used the harness as committed in `c4c09b43` (fingerprint
`f38f10f090c03add`). Its later changes touch only the docstring's account of
the checks; the steps are as they were, so only the fingerprint differs.
The file run 3 used is
`git show c4c09b43:benchmarks/harness/lora_export_real_configs.py`.

**Part B.** A Linux x86-64 box with 256 GB of RAM (run 1: GCP `n2d-highmem-32`
through Brev), `<repo>` the repository at the commit under test (run 1:
`git archive` of `f4375fa3`). Runtime and build evidence for run 1 is in
`part-b-setup-run1.evidence.log`.

```bash
sudo apt-get install -y build-essential cmake git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth 1 --branch b11476 https://github.com/ggml-org/llama.cpp <llama-src>
cmake -S <llama-src> -B <build> -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build <build> --target llama-results -j "$(nproc)"
uv venv --python 3.12 <convert-venv>
uv pip install --python <convert-venv>/bin/python --index-strategy unsafe-best-match \
  -r <llama-src>/requirements/requirements-convert_lora_to_gguf.txt
uv pip install --python <convert-venv>/bin/python <llama-src>/gguf-py
uv venv --python 3.12 <ref-venv>
uv pip install --python <ref-venv>/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
uv pip install --python <ref-venv>/bin/python transformers==5.19.0 peft==0.21.2 \
  safetensors huggingface_hub numpy psutil
uv pip install --python <ref-venv>/bin/python <repo>
HF_XET_CHUNK_CACHE_SIZE_BYTES=0 <ref-venv>/bin/python \
  <repo>/benchmarks/harness/lora_hook_real_model.py \
  --llama-bin <build>/bin --llama-src <llama-src> \
  --convert-python <convert-venv>/bin/python \
  --model-dir <checkpoint-dir> --work-dir /dev/shm/<scratch> --threads 32 \
  --box "<box>" --out part-b-real-model-run1.json --log part-b-real-model-run1.log
```

Run 1 used `f4375fa3` (fingerprint `fca602f5484081b8`), in bf16.
Its historical source is
`git show f4375fa3:benchmarks/harness/lora_hook_real_model.py`.
Run 2 used `--precision f32` and `-run2` names on a 384 GB box (§4g), with
`c46ab90b` (fingerprint `65161ef16679e79b`). The current instrument records
Part B's constants and each variant's `t`, enforces reference/target controls,
and repeats adapted logits. Historical captures do not establish these
additional controls.

## 6. Controls, provenance and repeat protocol (2026-10-09)

### 6.1 Validity controls

Historical captures report disabled-adapter/base and target metadata, but
their base-only repeats do not establish adapted-path determinism.

Both harnesses reject a missing/failed reference sanity, frozen-target or
reference-determinism control with VOID before conversion. Every final adapter
variant repeats its reference logits; every successfully loaded engine adapter
repeats its logits and tokens. Comparison is bitwise, including signed zero,
not approximate numeric equality. Failed engine repeats or token changes give
VOID; a model with a VOID adapter cannot support a hook recommendation.
Part B also requires base-token stability.
The original signal, error and noise thresholds are unchanged.

The combined LoRA/holdout checks pass **106 cases**, including fourteen
variant-boundary cases. Running those files before the eleven historical
full-suite failures passes **117 cases** in one native-locale process.
`ruff check src/soup_cli/ scripts/ tests/ benchmarks/` passes.
The historical full-suite result, its coverage and the paired baseline
comparison are reported separately; these targeted checks do not replace it.

### 6.2 Actual local adapter controls

Fresh CPU/f32 runs used seed 17, four engine threads, the original prompt,
`-fa off -ctk f32 -ctv f32`, and the existing stock/hook binaries. Every binary
SHA-256 matches the corresponding original local Part A' run. Current host:
Windows 11 build 26300, Python 3.12.10, torch 2.14.1+cpu,
transformers 5.19.0, PEFT 0.21.2. Parity harness fingerprint:
`7e1378a789ce32b1`.

| Fresh run | Reference adapter repeats | Engine adapter repeats | Result |
|---|---:|---:|---|
| [`part-a-controls-stock.json`](results/probe-lora-hook-engines/part-a-controls-stock.json), [log](results/probe-lora-hook-engines/part-a-controls-stock.log) | 21/21 bitwise equal | 18/18 bitwise equal; three adapters cannot convert | All 21 historical variant verdicts reproduced; HOOK NEEDED on the same four MLA projections |
| [`part-a-controls-hook.json`](results/probe-lora-hook-engines/part-a-controls-hook.json), [log](results/probe-lora-hook-engines/part-a-controls-hook.log) | 21/21 bitwise equal | 21/21 bitwise equal | All nine DeepSeek-V3 variants APPLIED, maximum `r = 1.3779691662976642e-5`; Qwen3.5 unchanged: 11 APPLIED, `in_proj_a` TOO WEAK |

All reference-disabled/base, frozen-routed-expert and token controls passed.
Thus the patched probe reports NO HOOK NEEDED *with the draft already applied*;
the stock comparison still identifies the hook's necessity. The Part A'
SUFFICIENT conclusion holds for these tiny f32 CPU paths. This is not a
quantised, GPU or real-MLA result.

Reproduce with §5's stock/hook build commands and the current harness, fresh
work directories, and output stems `part-a-controls-stock` /
`part-a-controls-hook`; do not overwrite historical files. These runs used
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` with the pinned tokenizers already cached.

The Part B `run` path was also exercised on the saved SYNTHETIC
`qwen35moe-tiny-seed17` checkpoint, in f32 with the stock converter/engine,
the default prompt and four engine threads. Both `soup-auto` and
`soup-auto+shared` were APPLIED with all reference/engine controls passed:
`r = 1.8357667556328084e-6` / `1.3608183473605723e-6`.
The [tiny integration capture](results/probe-lora-hook-engines/part-b-controls-tiny-run2.json)
([log](results/probe-lora-hook-engines/part-b-controls-tiny-run2.log))
records Part B's `rule_constants()` and f32 precision. Both reference and
engine repeats are bitwise equal. This is tiny-model integration evidence,
separate from the complete real-model repeat in §6.5.
The Part B harness fingerprint is `53c5a250b078968f`; its parity dependency
is `7e1378a789ce32b1`. To repeat this integration check, pass the saved Part A
tiny checkpoint to `lora_hook_real_model.run(ctx, model_dir, Clock(), "f32")`
with a stock-engine `Context`; do not invoke `main`, which downloads the real
35B model. Record `rule_constants()` rather than the parity utility's rule.

The historical cloud runs lack repeated adapted logits. Their numerical
observations remain available as historical evidence; §6.5 supplies the
registered real-model repeat with the complete controls.

### 6.3 Artifact provenance

[`artifact-provenance.json`](results/probe-lora-hook-engines/artifact-provenance.json)
records source and published SHA-256 values. Numerical values, flags, verdicts,
array digests and binary fingerprints are preserved. The seven holdout JSON
files and original coverage record/harness are unchanged. Evidence extracts
identify their source files and retained line ranges; full source captures
remain archived separately. Historical revision IDs identify measurement
sources and are not all ancestors of this branch.

### 6.4 Real-model repeat preregistration

The scope is correctness and serving architecture, not a throughput SLA or
delivery of prototype P1–P7. The `src/soup_cli` Git subtree is identical at
baseline `9aa43bd71afe12d3724f196202f1140e7dc409dc`, measured holdout revision
`011309792e989fcdb37cdf12eb7e8dc661e62ebd`, and the research tree used for the
baseline comparison: `beca7e706f4bee8c96d617221b0d1451357dd3d6`.

Invalid tokens or adapted repeats yield VOID before effect arithmetic.
After the registered retry, zero reference effect yields TOO WEAK; failed
controls take priority. Invalid/weak runs retain digests and step outcomes
but have no effect-error or noise-floor ratios.

**Baseline attribution [RUN].** The same eleven historical failing cases ran
sequentially with Python 3.12.10, the same dependencies, seed and environment,
native Windows cp1251 decoding, UTF-8 mode off, and verified tree-specific
imports. Their seven test files were byte-identical. Both baseline and the
pre-fix research tree produced 9 passed / 2 failed. The reproducible baseline
failures were the plugin test's missing UTF-8 subprocess decoder and TRL PPO's
omitted Transformers save-state initialization. The nine other failures did
not reproduce in either arm; overload is a hypothesis, not established cause.
After the narrow fixes, the benchmark tests followed by those eleven cases
passed together: **117 passed**, still in the native locale. The real PPO
case additionally saved checkpoints at steps 1 and 2 and reloaded every final
adapter tensor exactly. None of those eleven tests had an assertion,
timing bound or checkpoint disabled.
The [paired record](results/probe-lora-hook-engines/baseline-comparison.json)
retains every case, both import paths, environment, dependency versions and
test-file hashes; the [benchmark-first record](results/probe-lora-hook-engines/baseline-cases-after.json)
retains all 117 outcomes and the exercised source hashes.

**Boundary and CLI smokes [RUN].** [Actual local engines](results/probe-lora-hook-engines/harness-smoke.json)
returned APPLIED for positive tiny-model controls, VOID for changed prompt
tokens, and TOO WEAK for actual zero-B adapters in Part A and both Part B
variants. These were real PEFT, converter and `llama-results` executions,
not mocked engine responses; they do not replace real 35B evidence.
The [PPO CLI smoke](results/probe-lora-hook-engines/ppo-cli-smoke.json) ran
`soup train` for two local CPU steps, saved optimizer/scheduler/RNG checkpoints,
and reloaded the final adapter for finite inference logits of shape `[1, 2, 64]`.
Full PPO resume, including critic state, was not tested.

**Real-model repeat rule (unchanged).** Run 3 must use
`Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`,
CPU f32 in both runtimes, stock llama.cpp `b11476`, and the same Part B prompt,
seed, adapter recipe, two variants and numerical thresholds as §4g. It must
record base token stability, base and adapted bitwise repeats in both runtimes,
adapter-disabled base equality, actual adapted module names excluding routed
experts, and token equality on both engine adapter passes. No tiny-model result
substitutes for this run. All verdicts and failed attempts are retained; no
threshold is recalibrated from the repeat.

**Registered admission and teardown.** Before the repeat, the list-price
compute estimate was $2.547243 against the authorized $10 stage ceiling. The observed
`n2d-highmem-48` quote is $2.166624/hour: 130 minutes reserves $4.694352, plus
$2 for non-compute charges. Recheck the quote and empty inventory before create;
refuse a higher total rather than substitute a machine or silently reduce
controls. Use one instance, original-create-anchored Windows +120-minute and
server +130-minute guards, with work ending by +105 minutes. Copy complete or
partial evidence and match its fresh remote/local SHA before deletion; failed
copy cannot be replaced by a stale matching file. Verify absence before
removing the local deadline. Catalog pricing is not an invoice.

The committed source archive, exact revision, individual shell syntax checks,
LF checks, extraction and Git-blob hashes are verified before create. Public
LoRA and holdout evidence paths disable Git newline conversion so published
SHA-256 values identify the same bytes on Windows and Linux. The preregistered
[protocol](results/probe-lora-hook-engines/part-b-run3-protocol.json) is retained;
the completed measurement follows.

### 6.5 Real-model repeat with complete controls

**Both variants APPLIED [RUN].** The registered CPU/f32 repeat used the full
`Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754` checkpoint,
stock llama.cpp `b11476` (`988190680d5a89fce97de3c20df2c2813731fd61`),
48 threads and the same 22 prompt tokens. Source revision:
`3e75c3af1d7aab7431c5ca9785dbf1fe0808bb30`. The reference used
torch `2.14.1+cpu`, transformers `5.19.0` and PEFT `0.21.2`.
Seed 17, rank 8, alpha 16 and B standard deviation 0.02 were unchanged;
neither variant needed the registered weak-effect retry.

| Variant | Adapted modules | `s_ref` | `r` | Frozen-rule `t = 3f + 0.02` | Verdict |
|---|---:|---:|---:|---:|---|
| `soup-auto` | 80 | 0.111113 | 1.4406289065e-5 | 0.020035206742 | APPLIED |
| `soup-auto+shared` | 200 | 0.237401 | 8.2624312617e-6 | 0.020016478156 | APPLIED |

Base top-1 agreement was **1.0**, with `n_base = 1.3039755553902343e-6`.
Base and adapted repeats were bitwise equal within each runtime. Both
disabled-reference/base checks were exact; both engine adapter passes kept
the same tokens. Actual module lists contain no routed experts; saved PEFT
configs match the requested targets and contain no routed parameter targets.
Conversion, initial engine evaluation and repeated evaluation all exited 0.
This closes the real-Qwen adapted-determinism limitation, not real-MLA parity.

The [result](results/probe-lora-hook-engines/part-b-real-model-run3.json)
retains all metrics, module names, step outcomes, repeat digests and binary
fingerprints. [Local verification](results/probe-lora-hook-engines/part-b-run3-verification.json)
checked every archived artifact's size and SHA-256, all six `[22, 248320]`
engine arrays against their GGUF captures, finite logits, matching tokens,
bitwise repeats, the committed source bytes and the unchanged protocol.
Reference controls and effect metrics remain in the original harness result;
raw reference logits were not included in the archive.

The complete external archive contains **36 audited artifacts plus its audit**
and is **330,921,646 bytes**, SHA-256
`2100922f0f85951392f8d8bb746673942d5fe35fd773a3b320207615652fa68e`.
It includes both saved adapters, converted adapter GGUFs, six engine outputs
in GGUF and NPZ form, runtime versions and source provenance.

**Cost and teardown [ESTIMATE, RUN].** The preceding provisioning-only
lifetime added $0.3770762912. The completed measurement's create anchor was
06:15:43 UTC; resource absence was confirmed at 06:47:44.300532 UTC on
2026-10-09. Its conservative 1,921.300532-second lifetime at $2.166624/hour
adds $1.1563155119. Stage compute is therefore **$4.080635 (~$4.08)**,
leaving **~$5.92 before non-compute charges** against the $10 ceiling.
The $2 reserve is a margin, not an observed charge or invoice.
[Lifecycle evidence](results/probe-lora-hook-engines/part-b-run3-lifecycle.json)
records both lifetimes and source receipts. Fresh copy and matching
remote/local SHA-256 preceded deletion; confirmed absence preceded removal
of the local deadline. No cloud resource remains allocated for this run.

The adapters are SYNTHETIC and Qwen3.5 has no MLA. No trained-adapter quality,
DeepSeek-V3/Kimi-K2 real-weight parity, quantised/GPU path, SSD behaviour or
fit on the 8 GB GPU / 32 GB RAM target follows from this result.
Generation speed remains outside the correctness gate; P1–P7 remain future work.

### 6.6 Repository verification and test isolation

The [pre-isolation full suite](results/probe-lora-hook-engines/pre-isolation-suite.json)
ran the default non-smoke selection on Windows/Python 3.12.10 with two xdist
workers, `--dist loadfile`, `PYTHONHASHSEED=0`, `NO_COLOR=1` and `TERM=dumb`:
**34,181 passed, 14 failed, 432 skipped, 2 xfailed**. Coverage was
**86.28% (75,652 / 87,678 lines)**, above the unchanged 77% gate.
This was a failed suite, not a passing result.

The failures exposed pre-existing test defects: un-restored working-directory
changes, repository-relative source assertions, a source-tree mutation raced
by another scanner, and Rich controls dependent on ambient terminal settings.
All affected files were unchanged from the paired-comparison baseline before
these test repairs. Seventy-two directory changes now use test-scoped
`monkeypatch.chdir`. Twenty source/AST-only test nodes were removed; the
remaining runtime CLI, sampler, lazy-import and terminal-safety assertions
were preserved. No global directory reset, skip or relaxed timing bound was added.

The leak scanner now has six deterministic controls covering complete and
truncated public, secret and Basic tokens split by ANSI highlighting and
panel folds. The actual colored-console sanitizer check explicitly requests
color with a test-scoped terminal setting.
[Targeted verification](results/probe-lora-hook-engines/isolation-verification.json)
ran seventeen affected/related selections: **812 passed, 15 skipped**.
All **827 test boundaries** restored the working directory and `TERM`/`NO_COLOR`;
no isolation leak was observed. These targeted results do not substitute
for the complete-suite result.

The subsequent [complete-suite result](results/probe-lora-hook-engines/full-suite-verification.json)
used the same environment and default non-smoke selection:
**34,180 passed, 432 skipped, 2 xfailed, 870 warnings; zero failures or errors**.
Coverage was **86.28% (75,648 / 87,678 lines)**, above the unchanged 77% gate.
The command was `python -m pytest tests/ -n 2 --dist loadfile --tb=short -q`
with external JUnit and coverage reports. JUnit's 434 skipped cases include
the two expected failures. The verification record includes the measured
source/test hashes and the hashes of the retained stdout, JUnit and coverage
captures. `ruff check src/soup_cli/ scripts/ tests/ benchmarks/` also passed.
