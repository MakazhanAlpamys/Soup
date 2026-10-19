# Gate: layer streaming shards a DeepSeek-V3 FP8 block-quantised checkpoint by dequantising it

Status: **rule committed, not yet run.** Sections 1-4 are written before any gate run;
the results are appended below them and nothing above this line's sections is edited
after a run.

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

Filled at run time.
