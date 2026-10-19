# Gate: layer streaming shards a DeepSeek-V3 FP8 block-quantised checkpoint by dequantising it

Status: **G1 PASS, G2 PASS, G3 PASS** (CPU only; run 2026-10-19; review round in section 7). Sections 1-3 are the
rule as committed in `82abf7e6c` before any gate run, unchanged since (the branch was later
rebased onto main after #2039 merged; the gate ran with that commit as `cdfdf7546`); section 4 is filled
from the run, section 5 holds the results. No GPU run: the maintainer runs the GPU loss
check.

## 1. Question

DeepSeek-V3 is published as an FP8 block-quantised checkpoint, and Kimi K2 uses the same
format: each converted linear weight is `float8_e4m3fn` `[out, in]`, beside it sits a
float32 `<name>.weight_scale_inv` of shape `[ceil(out/bm), ceil(in/bn)]`, and `config.json`
carries `quantization_config: {quant_method: "fp8", fmt: "e4m3", activation_scheme:
"dynamic", weight_block_size: [bm, bn]}` (128 x 128 in the published checkpoints).
Embeddings, the head, the norms and the router stay bf16, the router correction bias
float32. Until this change Soup's layer streaming refused any `F8*` tensor or
`*_scale_inv` key ("Layer streaming has no FP8 path").

Layer streaming is Soup's implementation of layer-to-layer streaming of a frozen base
(L2L, Pudipeddi et al. 2020). The change dequantises each FP8 weight at shard time,
`w = w_fp8.float() * scale_inv` expanded over its 128 x 128 block and cropped at a ragged
edge, casts the fp32 result to the stream dtype, and then applies the existing storage
rules (NF4 under `quantization: 4bit` for what the meta probe converts). The scales are
consumed and never stored. Does the streamed model then train on exactly the numbers a
resident model holding the dequantised weights would, and does it work on a real
checkpoint?

Kimi K2 itself declares `model_type: kimi_k2`, which layer streaming still refuses by
name; this gate makes no claim about any K2 or any 1T run. Only `model_type: deepseek_v3`
is admitted for an FP8 source.

## 2. Decision rules (written before any run)

Every comparison below is `torch.equal` (bit-exact). No tolerance is substituted after the
fact; a failing check is reported with its maximum absolute difference.

### G1, dequantisation exactness (CPU)

Code under test: `soup_cli.utils.fp8_block.dequantize_fp8_block`. Reference: a dequant
written independently in the test, one Python loop per `(i, j)` block,
`ref[r0:r1, c0:c1] = w[r0:r1, c0:c1].float() * scale[i, j]` with `r1 = min(r0 + bm, rows)`.

Cases (shape, block):

| shape | block | why |
|---|---|---|
| (576, 2048) | (128, 128) | Moonlight's / DeepSeek-V3's `kv_a_proj_with_mqa` row count (576 = 4.5 x 128): ragged rows |
| (129, 257) | (128, 128) | ragged in both dimensions |
| (127, 1) and (1, 1) | (128, 128) | smaller than one block |
| (256, 384) and (2048, 1408) | (128, 128) | tileable (the second is Moonlight's routed `down_proj`) |
| (24, 64) and (100, 70) | (16, 32) | non-square block, ragged; catches a swapped `bm` / `bn` |
| (64, 96) | (16, 32) | non-square block, tileable |
| (16, 16), every finite e4m3 code point | (8, 8) | +-448, the subnormals, both zeros |

Weights are random normals with a different magnitude per block, quantised by
`benchmarks/harness/fp8_block_quantize.py` (one all-zero block included).

- **PASS**: Soup's fp32 output equals the reference in every case, and its bf16 and fp16
  casts equal the reference's casts.
- **Second oracle**: transformers 5.17's `Fp8Dequantize._dequantize_one` (CPU-callable,
  `output_dtype=float32`) must equal Soup's output on every tileable case. It derives the
  block size from the scale grid, so on a ragged shape it either refuses or uses a
  different block; the test records which, and that case is not an oracle case.
- **Converter check**: on the tileable cases the harness quantiser's FP8 bytes and scales
  equal transformers' own `Fp8Quantize`, so the converter writes the format transformers
  reads.

### G2, streamed-from-FP8 parity on a tiny `DeepseekV3ForCausalLM` (CPU)

Model: seeded random init, 3 layers with `first_k_dense_replace=1`, 8 routed experts plus a
shared expert, hidden 64, `e_score_correction_bias` set non-zero; saved bf16; then
FP8-block-quantised by the harness converter into the real format (config +
`weight_scale_inv`, the bf16-kept classes above kept bf16). The resident control is
`DeepseekV3ForCausalLM.from_pretrained` of a checkpoint holding the REFERENCE-dequantised
weights (the G1 reference), no `quantization_config`. LoRA targets: `q_proj` /
`q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, the same seeded adapters on both
sides.

Cells, CPU fp32, `quantization: none`:

| layout | q_lora_rank | block |
|---|---|---|
| fused | 24 | (16, 32) |
| fused | None | (16, 32) |
| per_expert | 24 | (16, 32) |
| per_expert | None | (16, 32) |
| fused | 24 | (128, 128) |

- **PASS (model)**: in every cell the streamed-from-FP8 logits, loss and every LoRA
  gradient equal the resident control's, and at least one LoRA gradient in layer 0 is
  non-zero.
- **PASS (cache)**: for `{fused, per_expert} x {none fp32, none bf16, nf4 bf16}` (NF4
  quantised on the CPU, double quantisation on), the cache sharded from the FP8 checkpoint
  is tensor-for-tensor equal, key for key, to the cache sharded from the
  reference-dequantised checkpoint with the same arguments, and its index agrees on every
  field except the source fingerprint and file list. No `*_scale_inv` key is in any shard.
  This is where NF4 is gated: the existing NF4 model-parity tests run only on CUDA, and a
  byte-identical cache hands the CUDA NF4 path exactly the bytes those tests already
  cover. The maintainer runs the GPU loss check.
- **Controls** (each must hold, or the cell pins nothing): the FP8 model's logits differ
  from the bf16 original's; the streamed model's decoder linears are plain `nn.Linear`,
  never transformers' `FP8Linear`, and its config carries no FP8 `quantization_config`.
- **FAIL**: any check false.

### G3, real data: Moonlight-16B-A3B (CPU only: header and tensor reads, no GPU)

1. `benchmarks/harness/fp8_block_quantize.py` converts
   `D:/soup-bench/models/Moonlight-16B-A3B` (bf16, `deepseek_v3`, 27 layers) tensor by
   tensor, one source file at a time, into the NEW folder
   `D:/soup-bench/models/Moonlight-16B-A3B-fp8block`, 128 x 128 blocks. FP8: every
   decoder `nn.Linear` weight (attention, dense MLP, shared and routed experts). Kept
   bf16: embeddings, `lm_head`, norms, `mlp.gate.weight`; float32 bias and the
   `rotary_emb.inv_freq` buffers unchanged.
   - Reported: the copy's size; per tensor, the relative Frobenius error
     `||deq - orig|| / ||orig||` and `max|deq - orig| / max|orig|` of the dequantised FP8
     weight against its bf16 original, as max and mean per tensor class (`q_proj`,
     `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, dense `gate/up/down`, shared
     `gate/up/down`, routed `gate/up/down`); wall time.
   - **Sanity bound (FAIL if broken)**: every tensor's relative Frobenius error is below
     0.1. e4m3 rounds to 3 mantissa bits (at most 6.25% per element), so a correct
     conversion sits well under it, and a transposed or mis-cropped scale grid lands near
     or above 1.
2. Soup's `shard_checkpoint` shards the FP8 copy into a NEW cache folder under
   `D:/soup-bench/` (cwd `D:/soup-bench`), `dtype=bfloat16`, `arch=deepseek_v3`,
   `num_hidden_layers` from the config, fused layout, `quant=nf4` with the meta probe's
   suffixes, `quant_device=cpu`.
   - **Time rule**: a first run with `num_hidden_layers=4` (the sharder's own
     multi-token-prediction drop path: layers 4-26 are dropped, embeddings, head and
     extras are written as in a full run) is timed. If 27/4 of its wall time exceeds 60
     minutes, the 4-layer cache is the result and the 27-layer time is the stated
     extrapolation, labelled as one. Otherwise all 27 layers are sharded.
   - Then `quant=none` into another new folder if the box has the time and D: keeps at
     least 100 GB free.
3. Checks on each cache written:
   - **Size**: `estimate_shard_cache_bytes` (fed the copy's `quantization_config`) must
     equal the cache's tensor payload exactly for every decoder layer and for the extras,
     embedding and head files together, and its total must be `>=` the cache's bytes on
     disk and `<= 1.05x` them.
   - **Real-data exactness**: for decoder layers 1, 13 and 26 (or 1-3 under the 4-layer
     result), every stored tensor that came from an FP8 weight equals the independent
     reference dequant of the copy's tensor, cast to bf16 (the fused routed experts:
     `cat(gate_e, up_e)` and `down_e` for every expert e); for the NF4 tensors of those
     layers, the packed bytes and every NF4 sidecar equal bitsandbytes' CPU quantisation
     of the reference-dequantised bf16 weight.
   - Reported, no threshold: shard wall time, peak RSS sampled every 2 s, the cache size.

- **PASS**: steps 1-3 complete and every check above holds.
- **FAIL**: a refusal, a crash, a sanity bound or an equality broken.

Box rules for G3: start only with at least 10 GB of RAM free; a watchdog stops the script
if free RAM falls below 3 GB (recorded as no verdict, not as a fail); D: must keep at least
100 GB free; the existing demo folders on D: are only read.

## 3. Not measured here

GPU anything: no CUDA run, no throughput, no loss curve (the maintainer runs the GPU loss
check). No quality claim: FP8 is lossy against the bf16 original by design, and G3's error
figures describe that loss, not a defect. No Kimi K2 or 1T checkpoint.

## 4. Box state

- The RTX 5070 Laptop box with its GPU unused: every process ran with `CUDA_VISIBLE_DEVICES=-1`
  (`torch.cuda.is_available()` False in every report). 32 logical CPUs, 31.7 GiB RAM.
- Software: Python 3.12.10, Windows 11 26200, torch 2.14.0+cu130, transformers 5.17.0,
  bitsandbytes 0.50.2 (CPU), safetensors 0.8.0.
- Code: G1 and G2 ran as `tests/test_fp8_block_streaming.py` at commit `1c1d855f4` (124
  tests; `fc2ef92fd` before the rebase, the same files). G3 ran `benchmarks/harness/fp8_shard_gate.py` from the same tree (its one-file
  index fallback was added before the run and is committed with this record).
- Other Claude sessions shared the machine. RAM, all in GiB (2^30 bytes), exact bytes from
  the reports: every G3 script was started only once at least 10 GiB were available (the
  check ran before torch was imported). The reports' box line, sampled just after torch
  loaded, reads 9.41-9.72 GiB (10,105,769,984-10,440,273,920 B). The lowest reading during any
  run was 4.98 GiB (5,346,910,208 B, the 27-layer NF4 shard), above the 3 GiB abort floor,
  so the watchdog never fired. (An earlier edit of this line said "9.4-10.2" and "5.35",
  mixing GB and GiB; corrected in the review round.) The laptop slept once (21:52-23:40 local)
  while a regression test file was running; that file was re-run afterwards, nothing in a
  gate ran across the sleep.
- D: had 189.8 GB free before G3 and 105.9 GB after it (decimal GB): the 100 GB floor held.
  The demo folders on D: were only read.

## 5. Results

### G1, dequantisation exactness: PASS

All in `tests/test_fp8_block_streaming.py`, every comparison `torch.equal`:

- `test_dequant_equals_the_reference_block_loop`: all 9 cases of section 2 (fp32, plus the
  bf16 and fp16 casts) equal the per-block reference loop.
- `test_every_finite_e4m3_code_point_dequantises_exactly`: the 254 finite e4m3fn code
  points (+-448, the subnormals, both zeros) under 2 x 2 random scales: equal.
- Second oracle, `test_transformers_dequant_agrees_on_tileable_shapes`: transformers'
  `Fp8Dequantize._dequantize_one` equals Soup's output on the 3 tileable cases
  (256 x 384, 2048 x 1408, 64 x 96 with a 16 x 32 block).
- Converter check, `test_the_harness_quantiser_writes_what_transformers_writes`: on the same
  3 cases the harness quantiser's FP8 bytes and scales equal transformers' `Fp8Quantize`.

Why transformers is not an oracle on ragged shapes (observed, informational): it derives
the block from the scale grid. On the real Moonlight tensor
`model.layers.0.self_attn.kv_a_proj_with_mqa.weight` (576 x 2048, scale 5 x 16) it raises
`ValueError: Weight shape (576, 2048) not divisible by scale grid (5, 16).` On a synthetic
200 x 256 weight with a 2 x 2 scale it accepts the input and uses 100-row blocks, which
differs from DeepSeek's 128 + 72 split by up to 45.4 in absolute value. (One direct call of
transformers 5.17.0 each; no whole-model resident load was tried.)

### G2, streamed-from-FP8 parity: PASS

- **Model, 5 of 5 cells** (`test_streamed_from_fp8_matches_the_dequantised_resident_bit_exactly`):
  logits, loss and every LoRA gradient are `torch.equal` to the resident control's in
  fused / per_expert x `q_lora_rank` 24 / None with a 16 x 32 block, and fused with a 128 x 128
  block. Each cell has a non-zero layer-0 LoRA gradient.
- **Cache, 6 of 6 cells** (`test_the_fp8_cache_is_the_dequantised_cache_byte_for_byte`):
  fused / per_expert x none fp32 / none bf16 / NF4 bf16 (CPU quantisation, double
  quantisation on). The FP8-sourced cache is tensor-for-tensor equal, key for key, to the
  cache of the reference-dequantised checkpoint. The index agrees on every field but the
  source fingerprint and file list, no `*_scale_inv` key is stored, and the disk estimate
  is the same for both sources.
- **Controls:**
  - `test_fp8_is_lossy_so_the_parity_pins_the_dequant`: the dequantised model's logits
    differ from the bf16 original's.
  - `test_the_streamed_model_is_built_bf16_not_with_transformers_fp8_layers`: no FP8
    module in the streamed model, `o_proj.base_layer` is `nn.Linear`, and the config has no
    `quantization_config`.
  - `test_the_meta_skeleton_never_hands_an_fp8_config_to_from_config`.
  - `test_the_resident_view_of_the_checkpoint_is_untouched`: `config.json` bytes are
    unchanged, and a fresh `AutoConfig` still reads the FP8 config.

Sensitivity (mutation run, section 6):
- the model cells fail when the dequant skips the scales, swaps `bm` / `bn`, crops the
  wrong edge, or reads experts without dequantising;
- the 128 x 128 cell is the one that does not see the swap or the crop, as expected: on
  hidden 64 every matrix is a single block.

### G3, Moonlight-16B-A3B on the CPU: PASS

Sizes and RSS below are in decimal GB (10^9 bytes); section 4's RAM figures are GiB.

Raw JSON: `benchmarks/results/probe-rtx5070/fp8-shard/`: `convert.json` (with every
tensor's errors), plus `shard-4L-nf4.json`, `shard-27L-nf4.json` and
`shard-27L-none.json`.

**1. Conversion** to `D:/soup-bench/models/Moonlight-16B-A3B-fp8block` (128 x 128 blocks,
DeepSeek-V3's `quantization_config` spelling):

| | value |
|---|---|
| source safetensors | 31,920,879,456 B (31.92 GB) |
| FP8 copy safetensors | 16,639,762,568 B (16.64 GB, 0.521x) |
| FP8 tensors (+ one `weight_scale_inv` each) | 5,181 |
| tensors kept bf16 (embeddings, head, norms, router, bias, `inv_freq`) | 163 |
| wall time | 76.2 s |
| peak RSS | 2.50 GB |

`kv_a_proj_with_mqa` is 576 x 2048 with a 5 x 16 scale, the ragged case of the real format.

Error of the dequantised FP8 weight against its bf16 original, per tensor, aggregated per
class. Relative Frobenius error is `||deq - orig|| / ||orig||`; relative max error is
`max|deq - orig| / max|orig|`.

| class | tensors | rel. Frobenius max | rel. Frobenius mean | rel. max-error max | rel. max-error mean |
|---|---:|---:|---:|---:|---:|
| q_proj | 27 | 0.02650 | 0.02645 | 0.03552 | 0.03146 |
| kv_a_proj_with_mqa | 27 | 0.02651 | 0.02645 | 0.03571 | 0.03313 |
| kv_b_proj | 27 | 0.02651 | 0.02646 | 0.03571 | 0.03364 |
| o_proj | 27 | 0.02648 | 0.02645 | 0.03571 | 0.03391 |
| dense gate_proj | 1 | 0.02644 | 0.02644 | 0.02679 | 0.02679 |
| dense up_proj | 1 | 0.02643 | 0.02643 | 0.01917 | 0.01917 |
| dense down_proj | 1 | 0.02642 | 0.02642 | 0.02632 | 0.02632 |
| shared gate_proj | 26 | 0.02648 | 0.02645 | 0.03456 | 0.01837 |
| shared up_proj | 26 | 0.02648 | 0.02645 | 0.03228 | 0.01602 |
| shared down_proj | 26 | 0.02646 | 0.02641 | 0.03250 | 0.01816 |
| routed gate_proj | 1,664 | 0.02652 | 0.02646 | 0.03571 | 0.03230 |
| routed up_proj | 1,664 | 0.02651 | 0.02645 | 0.03571 | 0.03378 |
| routed down_proj | 1,664 | 0.02650 | 0.02645 | 0.03571 | 0.02793 |

Sanity bound: the worst tensor (`model.layers.9.mlp.experts.44.gate_proj.weight`, 0.02652)
is far below 0.1. The ~2.6% is the same in every class, consistent with rounding to e4m3's
3-bit mantissa. The largest single error, 0.0357 x a tensor's max, is 1/28: half an e4m3
step at the top of the 448 range (step 32, so 16/448). These figures describe FP8's loss
against the bf16 original, not a defect of the sharder.

**2. Sharding** (`dtype=bfloat16`, fused, `quant_device=cpu`, NF4 suffixes from the meta
probe: the 10 decoder linears of attention, the dense MLP and the shared expert):

| run | layers | wall | peak RSS | cache on disk | estimate | estimate / disk | payload exact | real-data checks |
|---|---:|---:|---:|---:|---:|---:|---|---|
| NF4, time probe | 4 | 12.1 s | 4.40 GB | 4,755,794,622 B | 4,755,803,855 B | 1.0000019 | per layer and outside: yes | layers 1-3: 405 FP8-derived tensors (21 NF4), 0 mismatches |
| NF4 | 27 | 93.4 s | 4.63 GB | 30,598,520,564 B | 30,598,555,900 B | 1.0000012 | yes | layers 1, 13, 26: 405 (21 NF4), 0 mismatches |
| none | 27 | 73.2 s | 4.36 GB | 31,920,272,800 B | 31,920,287,732 B | 1.0000005 | yes | layers 1, 13, 26: 405, 0 mismatches |

- **Time rule.** 27/4 x 12.1 s = 82 s, far under 60 minutes, so all 27 layers were sharded.
  No extrapolation is used.
- **Disk.** There was room for the `quantization: none` run too, so it ran.
- **Real-data checks.** "FP8-derived" counts every stored tensor of the layer that came from
  an FP8 weight: 4 attention and 3 shared-expert linears, plus 64 experts x (`gate_up`,
  `down`) of the stacked routed experts.
  - Each equals the harness's own per-block reference dequant of the copy's tensor, cast to
    bf16.
  - The NF4 ones equal bitsandbytes' CPU quantisation of that reference, in the packed bytes,
    the absmax, the nested absmax and the offset.
- **Against the bf16 source.** The NF4 estimate is the same figure, to the byte, that #2039
  computed for the bf16 source (30,598,555,900 B), because the scales are charged nothing.
  The cache on disk is 26 bytes smaller than the bf16-sourced cache #2039 measured
  (30,598,520,590 B): its `index.json` records different source files.
- `total_params` is 15,960,111,936 in both 27-layer runs.

Folders written (new, left in place for the maintainer's GPU check):
- `D:/soup-bench/models/Moonlight-16B-A3B-fp8block` (16.6 GB);
- `D:/soup-bench/fp8-shard-4L-nf4` (4.8 GB);
- `D:/soup-bench/fp8-shard-27L-nf4` (30.6 GB);
- `D:/soup-bench/fp8-shard-27L-none` (31.9 GB).

## 6. Mutation check of the guards

48 guards were altered one at a time in the source, with `tests/test_fp8_block_streaming.py`
run after each and the file restored byte for byte (sha256 checked); the table is
`benchmarks/results/probe-rtx5070/fp8-shard/mutations.json`.

- 47 of 48 failed at least one named test on the first round.
- The one survivor was M17: dropping the dedicated compressed-tensors branch left the generic
  "declares quant_method 'compressed-tensors'" refusal, which the test's needle also matched.
  The two tests now assert the branch's own wording, and M17 then failed both.
- The mutations cover:
  - the dequant: scale skipped, `bm` / `bn` swapped, wrong crop, the non-finite, dtype and
    grid guards;
  - every config refusal and every tensor-layout refusal of section 1;
  - the arch gate;
  - the sharder's and the estimator's scale skips, FP8 plan and config-first refusal;
  - the expert and extras dequant paths, and the plain-read backstop;
  - the reader's F8 mapping;
  - the skeleton's config drop;
  - the setup's early refusal and the config it hands the pre-flight.

## 7. Review round (2026-10-20)

An independent review asked for changes. They were made in `34c9e98a0` (`9b9e51692` before the
rebase), after the gate run above; sections 1-6 describe the code at `1c1d855f4`.

What changed in the dequant:
- **Bounded work.** Rows now index their scale row (never expanded), and the column block is
  expanded as `min(block, cols)`. A `weight_block_size` past the matrix edge used to allocate
  its full size: a 4 x 4 weight with a [1, 1] scale cost about 1 GB at block 2^26.
- **Banding.** The fp32 product is formed one band of rows at a time (2^22 elements) and cast
  into an output preallocated at the stream dtype. A tensor now costs its output plus one band,
  not an fp32 copy plus expanded scales (about 9-11 bytes per element before).
- **New refusals.** A non-finite value after the dequant and cast is refused, naming the
  tensor: an e4m3fn NaN byte, or a float16 overflow, whose message says to stream in bfloat16.
  A negative scale is refused; a zero scale is kept.

New tests (`tests/test_fp8_block_streaming.py`, now 150):
- the huge block stays cheap and exact, at the dequant and through the sharder;
- the banded result is bit-equal to the unbanded reference for fp32, bf16 and fp16, with
  bands not aligned to blocks, and no fp32 intermediate exceeds one band;
- the NaN, overflow and negative-scale refusals, and the kept zero scale;
- a bf16 linear beside FP8 siblings is stored as it is;
- a non-FP8 `quantization_config` reaches `from_config` and the built model unchanged, which
  the code before this branch also does (checked by running the same probe on `387c758ca`, the head of #2039, merged as `d500ae0d`);
- the plain-read backstop refuses F8_E5M2, F8_E4M3FNUZ and a stray `*_scale_inv`.

**Effect on G3.** The 4-layer NF4 shard was re-run with the banded code into a new folder:
- verdict PASS, 14.7 s;
- cache 4,755,794,622 B, the same as before;
- all 7 safetensors files byte-identical (sha256) to the gate run's 4-layer cache;
- report: `shard-4L-nf4-review.json`.

The banded dequant writes the same bytes, so the 27-layer figures of section 5 stand. They were
not re-run, because D: had 101.2 GB free after this re-run.

**Units.** Section 4's RAM line mixed GB and GiB. It now gives GiB with exact bytes. The
"within 0.0001%" in the docs, the fragment and `benchmarks/README.md` is corrected to 0.0002%
(the 27-layer NF4 figure is 0.000115%, the 4-layer one 0.00019%).

**Mutations, review round.** 22 runs, each with `tests/test_fp8_block_streaming.py` and the
source restored byte for byte (sha256); all 22 killed. Table:
`benchmarks/results/probe-rtx5070/fp8-shard/mutations-review.json`.
- 11 are earlier mutants on the code that changed, re-spelled for the banded dequant: M01-M07,
  M35, M43, M45 and M46.
- 11 are new:
  - the column clamp;
  - the banding;
  - the stream-dtype output;
  - the non-finite check after the cast, and its overflow-versus-NaN message;
  - the negative-scale refusal, and keeping zero scales;
  - the skeleton dropping a non-FP8 config;
  - the backstop refusing only F8_E4M3, or missing a stray scale;
  - the sharder reading FP8 at fp32.
- The first version of the fix also clamped the row block. Rows are indexed, so that clamp
  changes nothing and no mutant could reach it; it was deleted rather than kept as dead code.
