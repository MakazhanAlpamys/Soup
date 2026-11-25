# Gate: layer streaming admits Kimi K2 through transformers' own DeepseekV3Config

Status: **G1a fails as registered on transformers' rope_scaling normalisation; G1b, G1c, G2
and G3 pass** (CPU only; run 2026-10-21, re-run after the review round in section 7, which
hardened the guard). G1a: FAIL, 42 of 43 fields. Sections 1-3 are the rule
as committed in `a236a4e49` before any gate run, unchanged since; sections 4-6 are filled from the
run. No GPU, no K2 weight download, no K2 or 1T training run.

## 1. Question

Kimi-K2-Base and Kimi-K2-Instruct (moonshotai) publish a `config.json` with
`model_type: "kimi_k2"` and an `auto_map` onto the repository's own
`configuration_deepseek.DeepseekV3Config` and `modeling_deepseek.DeepseekV3ForCausalLM`.
Their weights are DeepSeek-V3's published FP8 block format. Layer streaming is Soup's
implementation of layer-to-layer streaming of a frozen base (L2L, Pudipeddi et al. 2020),
and since #1933 it builds the streamed model from transformers' own class with
`trust_remote_code=False`. A real K2 therefore never reaches the architecture allowlist:
`AutoConfig.from_pretrained(trust_remote_code=False)` raises transformers' "contains custom
code" error first (`trainer/stream_setup.py` and `build_meta_skeleton` in
`utils/layer_stream_runtime.py`).

The change under test loads the raw `config.json` dict into transformers' native
`DeepseekV3Config` (`model_type` rewritten to `deepseek_v3`, `auto_map` dropped), but only
after a guard finds no field whose value would make the native class compute differently
from K2's own modelling code. Every K2 variant that fails the guard is refused by name,
naming the field and its value; Kimi-K2-Thinking (compressed-tensors INT4) is one of them.
transformers does the same mapping itself for the text tower of Kimi K2.5
(`Kimi_K25Config.__post_init__` maps a `kimi_k2` text config onto `deepseek_v3`).

Does the rewritten config carry every K2 field (G1)? Does the streamed native model then train
on exactly the numbers a resident native model on the same weights does (G2)? Does the disk
pre-flight size a real K2 cache from its real index (G3)?

## 2. Decision rules (written before any run)

Every tensor comparison is `torch.equal` (bit-exact). No tolerance is substituted after the
fact; a failing check is reported with its maximum absolute difference.

Inputs, read as JSON data only; no K2 Python file is imported or executed anywhere in this
gate. The files were fetched (text only) into the session scratchpad:

| file | revision | sha256 |
|---|---|---|
| Kimi-K2-Base `config.json` | `ce72df012259dcc55d945e890f815fe7ef69159c` | `8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439` |
| Kimi-K2-Instruct `config.json` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | same bytes as Base |
| Kimi-K2-Thinking `config.json` | `a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55` | `666ee58fc3871ecc369a22bc8c100075c2ad24ccbe8b3e336fa5f726e6111c15` |
| Kimi-K2-Base `model.safetensors.index.json` | as Base | `c1f1d16c853f20467ae81361d2a92223650d39efa005f9c872a7cc14425ddcbc` |
| `modeling_deepseek.py` (all three) | as above | `ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4` |

### G1, config (CPU, no model)

**G1a, mapping.** Soup's loader (`load_stream_model_config`) is run on a temporary folder
holding the real Base `config.json`, and again on one holding the Instruct file. It must
return a `DeepseekV3Config` whose `model_type` is `deepseek_v3` and that has no `auto_map`.
For every key of the raw JSON except `model_type` and `auto_map`, the loaded config must hold
the same value: `getattr(config, key) == raw[key]`. Two transformers normalisations are
named here in advance; an exploratory probe during design (a scratch script, before this
rule was written) showed that transformers applies them to any `config.json`, and they are
the only ones admitted:

- `rope_theta` is read as `config.rope_parameters["rope_theta"]`;
- `torch_dtype` is compared as `str(config.torch_dtype) == "torch." + raw["torch_dtype"]`.

Also: every key of the raw `rope_scaling` equals `config.rope_parameters[key]`, and
`config.rope_parameters["rope_type"] == "yarn"`.

Controls (each must hold, or G1a pins nothing):

- today's failure: `AutoConfig.from_pretrained(folder, trust_remote_code=False)` on the same
  folder raises `ValueError`;
- AutoConfig equivalence: the same dict with `model_type: deepseek_v3` and no `auto_map`,
  written to a second folder and loaded by `AutoConfig.from_pretrained`, gives a `to_dict()`
  equal to the K2 path's `to_dict()` except `_name_or_path`;
- the loader never calls transformers' dynamic-module loader
  (`get_class_from_dynamic_module` replaced by a function that fails the test), and the raw
  dict it reads is not mutated.

Reported, no threshold: every warning or log record at WARNING level or above emitted
during the load.

**G1b, the guard.** The guard checks, in this order, and a failure names the field and its
value:

| field | admitted | why the native class would compute differently otherwise |
|---|---|---|
| `architectures` | exactly `["DeepseekV3ForCausalLM"]` | another head is not the decoder this mapping was checked on |
| `topk_method` | `"noaux_tc"` | K2's router implements only this; the native router always runs it |
| `scoring_func` | `"sigmoid"` | K2's router implements only this; the native router always uses it |
| `n_group` | `1` | K2 masks unselected groups with 0.0, the native router with -inf; with one group nothing is masked |
| `topk_group` | `1` | as `n_group` |
| `num_experts_per_tok` | an int >= 2 when `norm_topk_prob` is true or absent | K2 renormalises the top-k weights only when top-k > 1; the native router always does |
| `moe_layer_freq` | absent or 1 | K2 makes only every n-th layer an MoE layer; the native class makes every layer past `first_k_dense_replace` one |
| `ep_size` | absent or an int <= 1 | K2 splits the experts across processes |
| `n_routed_experts` | an int >= 1 | `None` makes K2 all-dense; the native class cannot build it |
| `n_shared_experts` | an int >= 0 | `None` drops K2's shared expert; the native class always builds one |
| `num_key_value_heads` | absent or equal to `num_attention_heads` | K2's MLA ignores it; the native class divides by it |
| `rope_interleave` | absent or true | K2 always applies the interleaved rotary layout; the native class reads this flag |
| `rope_scaling` | absent, or a yarn dict as below | |
| `quantization_config` | absent, or a spelling `utils/fp8_block.py`'s `fp8_block_size` accepts | Soup dequantises only DeepSeek-V3's block FP8 |

A `rope_scaling` dict is admitted only when: its `type` is `"yarn"` (and `rope_type`, if
present, is too); it has no key outside `type`, `rope_type`, `factor`,
`original_max_position_embeddings`, `beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim` (the
keys K2's yarn reads; the native yarn reads more, such as `attention_factor` and `truncate`);
`factor` is a number; `original_max_position_embeddings` is present (K2 defaults it to 4096,
the native class to `max_position_embeddings`); `beta_fast` and `beta_slow` are not 0 where
present (the native yarn replaces a 0 with its default); and `mscale` is consistent with
`mscale_all_dim` (if `mscale_all_dim` is non-zero, `mscale` must be non-zero; otherwise
`mscale` must be absent or 1), the cases where the native attention factor equals K2's.

- **PASS (guard)**: the unmodified Base and Instruct dicts pass. For every row above (and for
  each `rope_scaling` condition), a copy of the real Base dict with exactly that one field
  changed to a failing value is refused with `UnsupportedKimiK2Error`, whose message names the
  field and the value. The FP8 spellings `fp8_block_size` accepts (with `scale_fmt: "float"`,
  without `fmt`) pass. Thinking is refused; the record names the first failing field
  (expected, by reading: `quantization_config`, `compressed-tensors`). A refusal never prints a
  raw control byte from the config.

**G1c, the guard is necessary on the numeric rows (reference transcribed by hand).** K2's
remote code does not import on transformers 5.17 (the investigation), and this gate does not
try to run it. Instead the router (`MoEGate.forward`) and the yarn rotary embedding plus its
softmax scale are transcribed from `modeling_deepseek.py`, read as text, into the test file,
and compared with the native `DeepseekV3TopkRouter`, the native rotary embedding and the
native attention's `scaling`:

- equal (`torch.equal` / `==`) on K2's own values and on every admitted value tested;
- different on each refused variant tested: `n_group` 2 with a negative router bias,
  `num_experts_per_tok` 1 with `norm_topk_prob`, no `original_max_position_embeddings`,
  `mscale_all_dim` without `mscale`, `mscale` != 1 without `mscale_all_dim`, a 0 `beta_fast`,
  and an `attention_factor` key. A transcription that raises where the native module computes
  counts as different (by reading, a 0 `beta_fast` divides by zero in K2's yarn).

A transcription can be wrong; this is evidence for the guard, not vendor-code parity.

### G2, streamed-vs-resident parity on a tiny K2-like checkpoint (CPU fp32)

Model: a seeded random native `DeepseekV3ForCausalLM`: vocab 128, hidden 64, 3 layers with
`first_k_dense_replace=1`, 8 routed experts plus a shared expert, top-k 2, `n_group=1`,
`topk_group=1`, `q_lora_rank` 24, `kv_lora_rank` 16, `qk_nope_head_dim` 16,
`qk_rope_head_dim` 8, `v_head_dim` 16, 4 heads, yarn rope (`factor` 32,
`original_max_position_embeddings` 64, `max_position_embeddings` 2048, `beta_fast` 1,
`beta_slow` 1, `mscale` 1, `mscale_all_dim` 1, `rope_theta` 50000), `routed_scaling_factor`
2.827, and `e_score_correction_bias` set to large random values. Weights saved bf16 (the
bias float32). `config.json` is the real Base dict with these dimensions substituted: it
keeps `model_type: kimi_k2`, `auto_map`, `architectures`, `topk_method`, `scoring_func`,
`moe_layer_freq`, `seq_aux`, `aux_loss_alpha` and `num_nextn_predict_layers: 0`. Every layer
also carries a stale BF16 `model.layers.N.self_attn.rotary_emb.inv_freq` (shape [7], filled
with a constant no rope uses), as the real checkpoint carries 61 of them. The folder holds no
`.py` file. The FP8 copy is written by `benchmarks/harness/fp8_block_quantize.py` (block
16 x 32) from the bf16 checkpoint.

Resident control: `DeepseekV3ForCausalLM` from a `DeepseekV3Config` built in the test from
keyword arguments (the native `rope_parameters` spelling, not through Soup's loader), fp32,
`load_state_dict(strict=True)` from the checkpoint's tensors minus the stale `inv_freq` keys
(the dropped set must be exactly those keys); for the FP8 cells, from the test's independent
block-loop dequantisation of the FP8 copy. `per_expert` cells set the resident experts
implementation to `eager`, as the FP8 gate does. LoRA: the targets Soup resolves for the
loaded config under `target_modules: auto`, rank 4, the same seeded adapters on both sides.

Cells, CPU, `dtype=float32`, `quantization: none`: {fused, per_expert} x {bf16 source, FP8
block source}.

- **PASS (model)**: in every cell the streamed model's logits, loss and every LoRA gradient
  equal the resident control's; both have the same set of LoRA gradients, and at least one
  layer-0 LoRA gradient is non-zero.
- **PASS (stale key)**: the runtime streams no `rotary_emb.inv_freq` key into any layer, and
  the streamed model has no per-layer `rotary_emb` module. Whether the sharder writes the key
  into the layer files is reported.
- **Controls** (each must hold, or the cells pin nothing): the resident model with the
  default rope instead of yarn gives different logits; zeroing the router bias changes the
  logits; the FP8-dequantised resident differs from the bf16 resident; the streamed model's
  config is a `DeepseekV3Config` with `model_type` `deepseek_v3`, no `auto_map` and (FP8
  cells) no `quantization_config`; no dynamic-module load happens.
- **FAIL**: any check false.

### G3, the real index (CPU, headers only, nothing downloaded)

No K2 shard is local, so the headers are synthesised: names from the real Base index,
shapes from the real config, and dtypes from the header facts the investigation recorded:
F8_E4M3 for every `*.weight` that has a `*.weight_scale_inv` beside it; F32 for those scales
(shape `[ceil(rows/128), ceil(cols/128)]`) and for `mlp.gate.e_score_correction_bias`; BF16
for the norms, the embeddings, `lm_head`, the router weight and the stale
`rotary_emb.inv_freq`. The stale key's shape is [56]: during design, a probe showed that it
is the shape for which the map's byte total equals the index's `total_size` exactly ([32],
`qk_rope_head_dim / 2`, leaves 2,928 B unaccounted for).

- **G3a, PASS**: the synthetic map's key set equals the index's key set (139,644 keys), and
  its byte total equals the index's `metadata.total_size`, 1,029,173,256,720 B.
- **G3b**: Soup's `estimate_shard_cache_bytes` runs over the map (only `read_header` is
  replaced, to serve it; every other step is Soup's), with `dtype=bfloat16`, the arch
  `stream_arch_of` returns for the config Soup's loader builds from the real `config.json`,
  `quant=nf4`, double quantisation on, the quant suffixes of
  `quantised_layer_suffixes(build_meta_skeleton(<folder with the real config>, quant="nf4"))`,
  `num_hidden_layers` 61, the config's `quantization_config`, in both layouts.
  **PASS**: no refusal, and in both layouts every layer's payload and the payload outside the
  layers equal an oracle computed in the gate script from the shapes alone (NF4 with double
  quantisation per weight of `n` elements: `ceil(n/2) + ceil(n/64) + 4*ceil(ceil(n/64)/256) + 4`
  bytes; everything else at its stored dtype).
- Reported: the total (payload plus the header and index allowances) and the RAM-tier bytes
  for each layout, compared with the investigation's figures (per_expert about
  533,246,623,322 B, fused about 2,039,152,149,798 B). An exact match is reported as one; a
  difference is reported with its size and explained, not adjusted, and is not a FAIL by
  itself.

### Mutation checks

Each guard row, each `rope_scaling` condition and each rewrite step (`model_type`, `auto_map`)
is altered in the source, a named test must fail, and the file is restored byte for byte
(sha256 checked). A mutation no test catches is reported and the test strengthened.

## 3. Not shown here

- **Parity with K2's own code.** Not run: the remote code does not import on transformers
  5.17, and Soup never executes it. G1c compares against a hand transcription only.
- **Known numeric differences from K2's own code**, not gated: the MoE combine (K2 sums the
  weighted expert outputs in fp32 and rounds once; the native class rounds each weighted
  output to the model dtype and accumulates with `index_add_`, bf16 rounding noise), eager
  versus sdpa attention, and the router bias held as a Parameter (K2) versus a float32
  buffer (native; both float32 in the checkpoint).
- **The tokenizer.** K2's tokenizer exists only as remote code and needs `tiktoken`; it is
  not exercised here.
- No GPU, no NF4 model parity (those tests run only on CUDA), no real K2 weights, no K2 or 1T
  training run, no throughput.

## 4. Box state

- The RTX 5070 Laptop box with its GPU unused: every process ran with `CUDA_VISIBLE_DEVICES=-1`
  (`torch.cuda.is_available()` False in the gate report). 32 logical CPUs, 31.7 GiB RAM
  (34,070,192,128 B); 11,084,537,856 B (10.3 GiB) available when the gate script started, and at
  least 6 GiB free before every pytest process. Other Claude sessions shared the machine.
- Software: Python 3.12.10, Windows 11 26200, torch 2.14.0+cu130, transformers 5.17.0,
  bitsandbytes 0.50.2 (CPU), safetensors 0.8.0. `tiktoken` 0.14.0 happens to be installed; no
  tokenizer was loaded.
- Code: the rule is `a236a4e49`. Every gate ran on `c5b723dfc` (code, tests and docs), with a
  clean working tree: G2, and the in-test halves of G1 and G3, as `tests/test_kimi_k2_streaming.py`
  (115 tests, all passed, 24.6 s); G1 and G3 on the real files as
  `benchmarks/harness/kimi_k2_gate.py --k2-dir <the scratchpad copies>` (peak RSS 1.0 GB).
- Nothing was downloaded. The real files were text-only copies fetched earlier into the session
  scratchpad (section 2's table; sha256 re-checked by the gate script). No K2 Python file was
  imported or executed: the loader's dynamic-module hook was replaced by a function that fails,
  and it was called 0 times.

## 5. Results

### G1, config: G1a FAIL; G1b PASS; G1c PASS

**G1a: FAIL, 42 of 43 fields.** On the real Base and Instruct files (byte-identical, sha256
`8c13ae10...7439`; the test file's `K2_BASE_CONFIG` equals the real Base dict), Soup's loader
returns a `DeepseekV3Config` with `model_type` `deepseek_v3` and no `auto_map`. Of the 43 raw keys
compared, 42 hold the same value under the rule, with its two named normalisations (`rope_theta`
and `torch_dtype`). The 43rd, `rope_scaling`, does not compare equal: transformers adds two keys to
it, `rope_theta: 50000.0` and `rope_type: "yarn"`. The rule admits no other normalisation, so G1a
fails as registered. The verdict stands; the rule is not amended.

Controls, all held, on both files:
- today's failure: `AutoConfig.from_pretrained(trust_remote_code=False)` raises `ValueError: The
  repository ... contains custom code which must be executed ...`;
- AutoConfig equivalence: the loader's `to_dict()` equals `AutoConfig`'s for the same dict spelled
  `deepseek_v3` without `auto_map`, except `_name_or_path`, and each `name_or_path` is its folder;
- no dynamic-module load (0 calls); `config.json` bytes unchanged; the in-memory raw dict is not
  mutated (deep copy, pinned by a test and by the `rewrite-deepcopy` mutation).

Reported: no Python warning and no `transformers` log record at WARNING or above during either
load.

#### G1a: analysis done after the verdict (not part of the registered rule)

- **The same additions on a native config.** `AutoConfig.from_pretrained` on the real Base dict
  spelled `deepseek_v3` without `auto_map` gives a `DeepseekV3Config` whose `rope_scaling` holds K2's
  seven keys plus the same two, `rope_theta: 50000.0` and `rope_type: "yarn"`. They come from
  transformers' rope standardisation, which writes into the `rope_scaling` dict it is handed, for any
  DeepSeek-V3 config.
- **Nothing K2 declares is lost or changed.** Each of K2's seven `rope_scaling` keys keeps its
  value, and `rope_parameters` holds all of them plus those two.
- **Why the rule missed it.** The exploratory probe before the rule handed transformers a shallow
  copy, so the raw dict it then compared against had received the same two keys.
- **Why it does not block admission.** G2 compares what the config computes. The streamed model,
  built from the `kimi_k2` folder through this loader, is bit-exact against a resident model whose
  config was built independently in the native spelling, with the same yarn rope, in every cell.
- `tests/test_kimi_k2_streaming.py` pins the observed behaviour exactly: the seven values equal,
  the two additions named.

**G1b.** Unmodified Base and Instruct pass. 39 refusal cases on the real Base dict (every row of the
table, each `rope_scaling` condition, absent spellings, non-int and non-mapping spellings) are each
refused with `UnsupportedKimiK2Error`, naming exactly that one field and its value; 21 admitted
variants pass (absent optional fields, the accepted FP8 spellings with `scale_fmt: "float"` and
without `fmt`, an `mscale` pair, top-1 without renormalisation, and Thinking's own non-quantisation
values). A refusal never carries a raw ESC, BEL or U+202E and is bounded in length.

Thinking (sha256 `666ee58f...1c15`) is refused. Its only guard failure, so the first, is
`quantization_config.quant_method='compressed-tensors'`, as expected by reading. It differs from
Base in 57 keys, most of them serialised generation defaults; the guard-relevant ones are
`quantization_config`, `rms_norm_eps` (1e-05), `rope_scaling.factor` (64.0),
`max_position_embeddings` (262144), `ep_size` (1) and the token ids, and all but
`quantization_config` are admitted values. With its `quantization_config` removed it passes the
guard (`thinking-unquantised`); that copy was not streamed.

**G1c.** The hand transcriptions of K2's `MoEGate.forward` and its yarn rotary embedding and
softmax scale agree with the native modules where the guard admits, and differ where it refuses:
- router at K2's values (384 experts, top-8, `n_group` 1, `topk_group` 1): indices and weights
  `torch.equal`. `n_group` 2 with an all-negative biased score: different experts (K2's 0.0 fill
  beats them). Top-1 with `norm_topk_prob`: same expert, different weight (K2 does not
  renormalise);
- rope over 4096 positions: cos, sin and the attention scale equal on 7 admitted variants (the
  published yarn, no scaling, `mscale` 1 alone, no `mscale` pair, an `mscale` pair of 0.707,
  default betas, factor 8); different on 4 refused ones (no `original_max_position_embeddings`,
  `mscale_all_dim` without `mscale`, `mscale` 0.707 alone, an `attention_factor`), and on a zero
  `beta_fast` the transcription raises `ZeroDivisionError` where the native class silently uses 32.

### G2, parity: PASS

- **Model, 4 of 4 cells** (`test_the_streamed_k2_matches_the_resident_native_model_bit_exactly`):
  fused / per_expert x bf16 source / FP8 block source (16 x 32). Logits, loss and every LoRA
  gradient of the streamed model, built from the `kimi_k2` folder through Soup's loader, are
  `torch.equal` to the resident control's; both have the same set of LoRA gradients (the targets
  Soup resolves: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), and each cell
  has a non-zero layer-0 LoRA gradient. No dynamic-module load in any cell.
- **Stale key**: the sharder writes `self_attn.rotary_emb.inv_freq` into every layer file as stored
  (pinned), the runtime streams it into no layer (no `rotary_emb` key in any layer's active keys),
  and the streamed model has no per-layer `rotary_emb` module, in both layouts.
- **Controls**, all held: the default-rope resident differs from the yarn resident; zeroing the
  router bias changes the logits; the FP8-dequantised resident differs from the bf16 resident; the
  streamed config is a `DeepseekV3Config`, `model_type` `deepseek_v3`, no `auto_map`, no
  `quantization_config` (both sources), and its decoder linears are plain `nn.Linear`.
- **Deviation from the rule's wording**: the resident control is loaded with
  `DeepseekV3ForCausalLM.from_pretrained(output_loading_info=True)` from a copy of the checkpoint
  without the stale keys, and the test requires empty `missing_keys`, `unexpected_keys` and
  `mismatched_keys`, instead of `load_state_dict(strict=True)` with the stale keys dropped.
  transformers 5.17 saves the fused experts as per-expert keys (the real index's layout), which
  only `from_pretrained`'s conversion maps back onto the fused module, so `load_state_dict` could
  not load them. The check is at least as strict.
- **Sensitivity** (section 6): the cells fail when the rewrite drops `rope_scaling` or changes
  `rms_norm_eps`, when the loader stops taking the K2 path, and when `build_meta_skeleton` stops
  using the loader.

### G3, the real index: G3a PASS, G3b PASS

**G3a.** The header map rebuilt from the real Base index names (sha256 `c1f1d16c...dcbc`) and the
config has the index's 139,644 keys and file assignments, and its byte total equals the index's
`metadata.total_size`, **1,029,173,256,720 B** (F8_E4M3 1,023,893,241,856; BF16 5,029,941,936; F32
250,072,928). It equals, key for key, the map `tests/test_kimi_k2_streaming.py` derives from the
config alone, which the test pins at the same count and total.

**G3b.** The meta probe of the real config (61 layers, NF4, 5.0 s on the meta device) reports 11
quantised suffixes: the five MLA projections, the dense MLP's three, and the shared expert's three;
the router weight is not among them. Over the map, with `dtype=bfloat16`, arch `deepseek_v3`,
double quantisation on and the 61 real source files, `estimate_shard_cache_bytes` refuses nothing
and equals the shape-only oracle in every layer and outside the layers, in both layouts:

| layout | layer payload | other payload | header + index allowance | total | RAM tier |
|---|---|---|---|---|---|
| per_expert | 528,527,512,320 | 4,697,635,904 | 21,460,490 + 25,216 | **533,246,633,930** | 9,779,755,776 |
| fused | 2,034,454,168,320 | 4,697,635,904 | 330,710 + 25,472 | **2,039,152,160,406** | 2,039,151,803,136 |

Layer 0 (dense) is 256,669,440 B in both; an MoE layer is 8,804,514,048 B per_expert (of which
the experts file holds 8,724,089,856) and 33,903,291,648 B fused. The fused cache is 3.82x the
per_expert one because the fused routed experts stay bf16 under NF4.

Against the investigation's figures (533,246,623,322 and 2,039,152,149,798 B) both totals are
**10,608 B higher**, the same in both layouts. Explained and reproduced: the investigation sized
the map as one source file and the stale `inv_freq` at [32]. One file instead of 61 removes 60
index entries (7,680 B of index allowance); [32] instead of [56] removes 61 x 24 x 2 = 2,928 B of
payload; with both, the estimator returns exactly the investigation's two figures (scratch probe,
`k2_g3_delta_probe.py`). The [56] here is the shape the index's `total_size` requires (G3a).

What this means for a real run (arithmetic, not measured): under `quantization: 4bit` a Kimi K2
layer cache needs about 533 GB of disk with `per_expert` and about 2.04 TB with `fused`; the
`per_expert` RAM tier holds 9.8 GB (the routed experts are read from disk on every tier).

## 6. Mutation run

49 mutations of `src/soup_cli/utils/kimi_k2.py`, `src/soup_cli/trainer/stream_setup.py` and
`src/soup_cli/utils/layer_stream_runtime.py`, one pytest process each over the named tests of
`tests/test_kimi_k2_streaming.py` (scratch driver `k2_mutate.py`). Each file was restored byte for
byte after its mutation (sha256 checked after each, and all three again at the end, equal to the
files committed in `c5b723dfc`).

- Every guard row: `architectures`, `topk_method`, `scoring_func`, `n_group`, `topk_group`,
  `num_experts_per_tok` (the top-k rule, its integer check, the `norm_topk_prob` default),
  `moe_layer_freq`, `ep_size`, `n_routed_experts`, `n_shared_experts`, `num_key_value_heads`,
  `rope_interleave`; every `rope_scaling` condition (not a dict, `type`, `rope_type`, extra keys,
  `factor`, `original_max_position_embeddings`, zero betas, the `mscale` rule and each of its two
  halves); `quantization_config` (the FP8 refusal, other methods, a non-mapping); a non-mapping
  config.
- Each rewrite step: `model_type`, `auto_map`, the deep copy; and two config changes that must
  break G2 (`rope_scaling` dropped, `rms_norm_eps` changed).
- The loader: the K2 path, the guard call, the `trust_remote_code` branch, the revision kwargs to
  each read, `name_or_path`; the message's truncation, `repr` and the rope key's rendering; the
  tokenizer hint (the tiktoken clause, the K2-only check, the print); the setup's loader call, hint
  call and re-raise; `build_meta_skeleton`'s loader call.

**48 caught on the first run, 1 survived:** `num_experts_per_tok-bool` (dropping the integer check
of the top-k rule). Its test case, `num_experts_per_tok: true`, was refused anyway, because
`True >= 2` is false. The case was replaced by `8.0` (a float that passes the comparison) and `true`
with `norm_topk_prob: false`; the mutation is then caught, and the re-run of it and of the
`num_experts_per_tok` mutation caught both. Tests added after the mutation run only add assertions
(the stale key in the layer files, the K2.5 refusal).

## 7. Review round (after the results; the rule in sections 1-3 is unchanged)

An independent review of `9d14ba561` returned CHANGES NEEDED. It found the loader itself safe:
it reads JSON only, gates on `model_type == "kimi_k2"` exactly, and has two streaming call sites;
the Moonlight cache fingerprint is unchanged. It then proved holes in the guard with tiny probes:

- **HIGH.** A transformers-5 `rope_parameters` (no `rope_scaling`) carrying `attention_factor` and
  `truncate` was admitted, and the native config uses both.
- **HIGH.** `num_key_value_heads` absent was admitted. The native default is 128 against K2's 64
  heads, so `num_key_value_groups` is 0 and the forward crashes. Section 5's G1b admitted list
  pinned that case as passing.
- **MEDIUM.** A top-level `partial_rotary_factor` or `head_dim` was admitted. The native rope reads
  them, K2 ignores them, and the forward crashes.
- **MEDIUM.** The tokenizer loaded before the guard ran, so a Kimi-K2-Thinking user was first told
  to pass `--trust-remote-code` and install `tiktoken`, and only then refused.
- **Surviving mutants of the review's own:**
  - the hint test raised a `ValueError` where a missing `tiktoken` really surfaces as an
    `ImportError`;
  - `partial_rotary_factor` inside `rope_scaling` was not pinned;
  - an empty `rope_scaling` had no defined path;
  - nothing pinned that a failure inside the hint never masks the tokenizer's own error.

### Changes (`cfe1699d6`, then `acad7a66e`)

- **The guard is also an allowlist of top-level keys.** It admits the 45 keys of the published
  Base / Instruct `config.json` plus three metadata keys, each justified in the code:
  - `_commit_hash`, which transformers' own reader adds to every dict it returns;
  - `_name_or_path`;
  - `dtype`.

  Any other key is refused by name, after the registered rows, with "training.stream_layers admits
  kimi_k2 only with the published K2 config fields".
- **The `ep_size` and `rope_interleave` rows are gone.** Neither key is in the published file, so
  both are now refused whenever present; before, the rows admitted `ep_size <= 1` and
  `rope_interleave: true`. Their registered refusal cases are still refused, naming the field and
  its value.
- **`num_key_value_heads` is required** and must equal `num_attention_heads`.
- **An empty `rope_scaling` is refused on its own.** K2 indexes `rope_scaling['type']` whenever
  `rope_scaling` is not null. `null` keeps its defined path: both classes use their plain rope
  (G1c "no-scaling").
- **The setup runs the guard before the tokenizer** by loading the model config first.
- **The loader import moved** (`acad7a66e`). It now sits beside the tokenizer import, because next
  to the `fp8_block` import it conflicted with `fix/stream-refuse-packed-quant` (tip `454e57113`).
  `git merge-tree` is clean now. On that merged tree, `test_kimi_k2_streaming` (127),
  `test_deepseek_v3_streaming` (30), `test_fp8_block_streaming` (157) and
  `test_stream_refuses_packed_quant` (211) pass.

### G1b cases added after the review (not part of the registered rule)

- **Refused**, each naming exactly its field and value:
  - `num_key_value_heads` absent, and 32 (which divides 64);
  - a top-level `rope_parameters` with `attention_factor` / `truncate`;
  - top-level `partial_rotary_factor` and `head_dim`;
  - `rope_scaling['partial_rotary_factor']`;
  - `rope_scaling: {}`;
  - `ep_size: 1` and `rope_interleave: true`;
  - `pad_token_id` and `sliding_window`;
  - outside the 50-case list, the control-bytes test adds a top-level key made of control bytes,
    which is refused and rendered escaped.
- **Removed from the admitted list:** `ep_size` 1 and 0, `num_key_value_heads` absent,
  `rope_interleave: true`, and Thinking without its quantization. The last is now refused naming
  `ep_size` and `pad_token_id`.
- **Added to the admitted list:** the three metadata keys.

G1b now has 50 refusal cases and 17 admitted variants. The allowlist is pinned to equal the real
Base key set.

### Re-run on `acad7a66e`

- **Tests.** `tests/test_kimi_k2_streaming.py`: 127 passed. `test_deepseek_v3_streaming` (30),
  `test_fp8_block_streaming` (157), `test_issue602_qwen4_ple_streaming` (33, 2 skipped) and
  `test_terminal_output_hygiene` (31) passed, and ruff is clean. So did the other files that drive
  the setup: `test_qwen35_streaming`, `test_issue803_*`, `test_stream_moe_lora_targets`,
  `test_mla_lora_targets`, `test_stream_expert_layout`, `test_stream_disk_preflight_moe` and
  `test_cli_startup_is_light`.
- **G1** (gate script on the real files):
  - G1a is unchanged: FAIL, 42 of 43 fields, `rope_scaling`.
  - The controls hold, with 0 dynamic-module loads.
  - Thinking's first failure is still `quantization_config.quant_method='compressed-tensors'`. The
    allowlist then names 50 more fields of its file: its serialised generation defaults, `ep_size`
    and `pad_token_id`. Section 5's "its only guard failure" describes the guard before this round.
- **G2** is unchanged: 4 of 4 cells bit-exact.
- **G3** is unchanged: G3a holds, and the estimate equals the oracle in both layouts, with the same
  totals (533,246,633,930 and 2,039,152,160,406 B).

### Mutation re-run

61 mutants were run on the review-round code:

- **46 of section 6's 49, unchanged.** The other 3 target code this round replaced (the `ep_size`
  row, the `rope_interleave` row, the old `num_key_value_heads` condition).
- **15 new ones:**
  - the allowlist: switched off; admitting `ep_size` / `rope_interleave`; admitting
    `rope_parameters` / `partial_rotary_factor` / `head_dim`; dropping `_commit_hash`; one extra
    metadata key; raw key names in the message; the reason text;
  - `num_key_value_heads`: absent admitted; any integer admitted;
  - `rope_scaling`: the empty-dict branch off; a falsy `rope_scaling` admitted; `partial_rotary_factor`
    added to the rope keys;
  - the setup's `except` narrowed to `ValueError`; the hint's own `except` narrowed;
  - the guard moved back after the tokenizer.

**All 61 caught.** Three of them (`hint-print`, `setup-hint-call`, `setup-reraise`) failed their
tests on the first pass, but the driver still named the test id from before it was parametrised.
With the ids corrected they were re-run and caught. After the import move, the five `stream_setup.py`
mutants were run again on `acad7a66e`: 5 of 5 caught. Each file was restored byte for byte, with its
sha256 checked.

### Correction: "does not import on transformers 5.17"

Sections 2 and 3 say K2's own code "does not import on transformers 5.17", quoting the
investigation. That was not observed here: K2's code was never run. It is an inference from
reading the code. Line 56 of `modeling_deepseek.py` imports `is_torch_fx_available` from
`transformers.utils.import_utils`, and transformers 5.17.0 does not define that name (checked by
`hasattr` and in its source); every other name the file imports from transformers exists. The docs
now state it this way.

### Round 2: review READY, survivors killed

The second review returned READY. It confirmed that the allowlist equals the union of the real Base
and Instruct keys (45 of 45), that all first-round holes are closed, and that the Moonlight cache
is unchanged. Under the house rule, each survivor it listed now has a test that kills it:

- **r2.** The setup could have handed the user's `trust_remote_code` to the config loader without
  any test noticing, because every test ran with the flag off. A K2 base would then run its
  repository's configuration code. The new test
  `test_trust_remote_code_reaches_the_tokenizer_but_never_the_streamed_config` drives the setup
  with the flag on: the loader gets `False`, and the tokenizer gets `True`.
- **r1.** A key outside the allowlist with a null value is refused like any other value: refusal
  case `head_dim: null`.
- **r3.** A non-ASCII key name was shown raw. Refusals now render key names and values with
  `ascii()`, which escapes every control and non-ASCII character, so a Cyrillic homoglyph cannot
  read as an ASCII name; the CLI's error printer then applies `for_terminal`. Pinned by
  `test_non_ascii_key_names_and_values_are_shown_escaped`.
- **LOW 1.** A 200,000-character identifier key produced a 400 KB message. Key names are now always
  bounded to 80 characters, and a refusal lists at most 10 fields, then "and N more". Pinned by
  `test_a_refusal_stays_bounded_whatever_the_key_names_and_their_number`.
- **LOW 2** (a known limit; no code change): `quantization_config` is checked only by
  `fp8_block_size`. Its sub-keys are not allowlisted against K2's published ones, so for example
  `weight_block_size: [64, 64]` or a `modules_to_not_convert` list is admitted. Nothing in Soup reads
  `modules_to_not_convert` today.

Mutants: r1, r2, r3 (on the key and on the value) and three for LOW 1 (unbounded name, uncapped list,
no "and N more"). All 7 caught, with sha256 verified. `tests/test_kimi_k2_streaming.py`: 131
passed. G1b now has 51 refusal cases and 17 admitted variants.

### Second review of the pushed PR (after the results; the rule in sections 1-3 is unchanged)

A review of the pushed head `050ca40b` found CI red and three guard or path findings. After a merge
of `origin/main` (with #2094, which pins `build_meta_skeleton`'s config read to the bound commit):

- **CI, contract test.** #1972's contract looked for `AutoConfig.from_pretrained` in
  `stream_setup.py`, and the read had moved into the loader. The contract now runs the setup's own
  `load_stream_model_config(...)` call through the real loader, with strict receivers on both of
  its reads (`AutoConfig`, and the raw `config.json` read of a K2 file).
- **CI, `_commit_hash`.** CI installs transformers 5.19.0, whose reader adds no `_commit_hash` at
  all, not even for a Hub snapshot folder (5.17 adds `None` for a plain folder and the commit for a
  snapshot), and whose config then has no `_commit_hash` attribute. The test runs on both folder
  shapes and asserts only what holds on both versions: the config keeps whatever the reader
  recorded, or nothing.
- **CI, Linux.** The NF4 suffix fixture built the full-size skeleton, and transformers allocates
  each MoE layer's fused experts on the CPU before they move to meta: 22,548,578,304 B at once,
  refused on a 16 GB runner. The probe now runs on a size twin of the published config (only size
  fields differ) and must report the same 11 suffixes as the full-size probe in G3b.
- **MEDIUM.** The streamed tokenizer load did not get `config=`, so with `--trust-remote-code`
  `AutoTokenizer` re-read the config with that flag and imported the repository's
  `configuration_deepseek.py` (traced by the review on 5.17). It now gets the loaded native config.
  A test runs the setup's tokenizer call with the real `AutoTokenizer` on a local K2 folder whose
  `configuration_deepseek.py` writes a sentinel file when imported; the file must not appear.
- **LOW.** Every published key must be present (only `auto_map` and `quantization_config` may be
  absent; `torch_dtype` may be spelled `dtype`), so a missing scalar cannot take the native
  default. `rope_scaling` absent and `moe_layer_freq` absent move from the admitted list to the
  refused one (`rope_scaling: null` is still admitted).
- **LOW.** Every refusal now says, before its field list, that other K2 releases
  (Kimi-K2-Instruct-0905, Kimi-K2-Thinking) are out of scope for now. The 0905 `config.json`
  (revision `ac6c49f0`, sha256 `4cd10b0f...07cc`; the review downloaded that file and nothing else)
  is refused naming exactly the keys it adds.

G1b now has 57 refusal cases, 16 admitted variants and 43 required-key cases.
`tests/test_kimi_k2_streaming.py`: 186 passed. 32 mutants (14 new, the review's 8 not run before,
10 earlier ones on code this round touched): all 32 caught, each file restored and its sha256
checked.
