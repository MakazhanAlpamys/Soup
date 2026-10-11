# B3 gate: block-scaled FP8 source checkpoints

Branch `b3-fp8-source` on `SoupTeam/soup-rnd`. Rule committed at `fcbd15a`, before any run; counted runs at `79a4d3f`. Base `9aa43bd`.
Harness: [`harness/fp8_source_probe.py`](harness/fp8_source_probe.py).
Card: B3 "Читать FP8 checkpoint и готовить NF4 веса" (rndTASK.pdf p. 5).

## 1. Rule, committed before the first real-shard run (2026-10-07)

Results are labelled **SYNTHETIC** (the CI fixtures in
`tests/test_fp8_source_checkpoints.py`) or **REAL** (DeepSeek-V3 source shards).
The two are never pooled.

**G1, bit-exact FP8 -> bf16.** For every float8 weight in the source, the new
path's output (`dequantize_fp8_blockwise`) must be `torch.equal` to an
independent decode: e4m3 read from its bit fields through a 256-entry table (no
`float8` cast), fp32 multiply by the block scale, one rounding to bf16. PASS iff
every tensor is equal. Any unequal tensor is FAIL, named.

**G3, NF4 reconstruction of what the sharder wrote.** For every float8 weight
the sharder quantised to NF4: read the packed nibbles and absmax back from the
layer shard, dequantise with bitsandbytes, and compare with the G1 reference.
Per NF4 codec block of 64 consecutive elements (not per FP8 128 x 128 block):
`e_b = max |n - r|`, `a_b = max |r|`, in fp32.
PASS iff `e_b <= 0.17 x a_b` in every block and every output is finite. A block
with `a_b = 0` passes only if every output in it is exactly zero.
Reported separately for double-quant on (Soup's default) and off.
**0.17 is fixed.** An exceedance is a finding, reported with tensor, block and
absmax spread, and the threshold is not raised. Rationale for the number:
design note section 3a (0.1519 NF4 half-gap + 0.0039 bf16 rounding, analytic
without double-quant; the double-quant remainder is an allowance, not a proof).

**G5, memory.** Peak RSS of the sharding process is recorded with the box. No
threshold: the per-chunk bound is asserted structurally in the tests; this row
records what the real run used.

**G6, sharding time.** Wall seconds for `shard_checkpoint(quant="nf4")` over
the source, divided by source GB (10^9 bytes), with the time split into FP8
dequantisation, NF4 quantisation and the rest. No threshold: a measurement,
valid only for the box it was taken on. Brief's bf16 baseline for scale only,
different box and source format: 138 GB -> 36.4 GB in 217 s.

**Validity.**
- V1: the box is stamped (CPU model, cores, RAM, disk, library versions, tree).
- V2: this run is on a shared cloud sandbox (2 vCPU, 7 GB RAM), not the team
  dev box. Its G6 seconds are not comparable to any dev-box number and say
  nothing about NVMe read rates. G1 and G3 are numerical and do not depend on
  the box.
- V3: a source file that splits a decoder layer across files yields a PARTIAL
  layer here; the record names it and does not count it as a full MoE layer.
- V4: a run that dies (OOM, killed) is kept and reported as void, not retried
  silently.

## 2. Box

| Field | Value |
|---|---|
| Machine | Shared cloud sandbox, not the team dev box (rule V2) |
| CPU | Intel Xeon @ 2.10 GHz, 2 logical CPUs, torch threads 2 |
| RAM | 8.42 GB total |
| Disk free at start | 17.8 GB (DQ on run), 9.0 GB (DQ off run) |
| OS | Linux 6.18.44, glibc 2.39 |
| Python / torch / bitsandbytes / safetensors | 3.12.3 / 2.14.1+cu130 (no GPU) / 0.50.2 / 0.8.0 |
| Tree | `79a4d3f`, `src/` clean in both runs |
| When | 2026-10-07 14:03 UTC (DQ on), 14:07 UTC (DQ off) |

**Source (REAL).** `deepseek-ai/DeepSeek-V3`, `model-00001-of-000163.safetensors`
(5,234,139,343 bytes, header and size checked complete) and its `config.json`
(`quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128, 128]`).
It holds layers 0 to 2 complete (dense) and layer 3 PARTIAL: attention, the
shared expert and 32 of 256 routed experts (rule V3).

**Staging.** No prefix of the DeepSeek-V3 files is closed: files 1, 1..2 and
1..3 each leave exactly one weight whose scale is in the next file (from the
official `model.safetensors.index.json`). For file 1 that weight is
`model.layers.3.mlp.experts.31.up_proj.weight`. The harness copies the file
byte for byte without that one tensor (`--exclude-key`, offsets rewritten,
every other tensor's bytes verified identical on a synthetic copy). 126 FP8
weights remain: 8 in each of layers 0 to 2, 102 in layer 3. 5,219,459,160
source bytes.

## 3. Results

### Void runs, kept (rule V4)

| # | Tree | What happened | Measured |
|---|---|---|---|
| void-01 | `354f204` | The sharder refused `--out` outside `$HOME` / `$CWD` / `$TMPDIR` (here `$HOME` is `/root`). Harness invocation error. | Nothing |
| void-02 | `354f204` | Refused: `experts.31.up_proj.weight` had no scale in file 1. Correct refusal of an incomplete source, and it exposed a design error (below). | Nothing |

**Finding from void-02, fixed before any counted run.** The official index puts
**155 of DeepSeek-V3's 45,808 scales in the file after their weight**, where a
layer crosses a file boundary. The design's "weight and scale in the same file"
refusal, copied from the oQ path, would have refused the full real checkpoint.
The synthetic tests could not have shown it. Dropped in `1aadd15`, with a test
that decodes a cross-file scale through a single live source handle.

### G1, bit-exact FP8 -> bf16 (REAL)

**PASS, both runs: 126 / 126 tensors `torch.equal`** to the independent decoder,
including all 102 expert and attention weights of the partial layer 3.

### G3, NF4 reconstruction of what the sharder wrote (REAL)

| Double-quant | Tensors passing | Worst `e_b / a_b` | Zero blocks | Verdict |
|---|---|---|---|---|
| off | **126 / 126** | **0.1531** | 0 | **PASS** |
| on (Soup's default) | **3 / 126** | **2.1503** (`layers.1.mlp.gate_proj`) | 0 | **FAIL** |

Without double-quant the worst block sits at 0.1531, inside the analytic
0.1558 (NF4 half-gap 0.1519 + bf16 rounding 0.0039). The bound holds where the
design proved it.

**The double-quant failure is the existing NF4 codec, not B3.** Checked on the
worst tensor, outside the sharder:
- The sharder's packed nibbles are **byte-identical** to `quantize_4bit` on the
  G1 reference, and its dequantised output is identical too (G2 on REAL data).
- `quantize_4bit` called directly on that reference gives the same 2.1503.
- Mechanism: double-quant stores each block's absmax as an 8-bit code
  relative to its 256-block group, after subtracting the group mean (0.0211
  here). The worst block's true absmax is 0.000584 in a group spanning
  0.000584 to 0.2012 (345x); it is stored as **-0.000673**, so the block comes
  back sign-flipped. On this tensor 9.27% of blocks have an absmax stored more
  than 10% off, and 6.39% of blocks exceed 0.17.
- Whole-tensor context, same tensor: relative Frobenius error 0.0872 without
  double-quant, 0.0880 with; maximum absolute error 0.03418 in both. The
  failing blocks are the near-zero ones.

Per the rule, **0.17 is not raised.** The finding is reported as (scoped after review, 2026-10-08): in one real DeepSeek-V3 file, the shipped double-quant NF4 path fails the per-block `0.17 x absmax` bound in 123 of 126 tensors; on the worst tensor (`layers.1.mlp.gate_proj`) about 6% of its blocks exceed it, through absmax sign flips in low-magnitude blocks. Other files and Kimi K2 are not measured. **Whether that affects fine-tuning quality is not
measured here.** It is a quality question (Track E style: double-quant on vs
off, same data and steps, scored on the gate suites), and it applies to every
NF4 streamed model with a wide absmax spread, not only FP8 sources. The Gaussian
fixtures the bound was chosen beside (0.1593 worst) did not show it.

### G5, memory (REAL)

| Double-quant | Peak RSS through sharding |
|---|---|
| on | 4.84 GB |
| off | 5.96 GB |

RSS counts the memory-mapped source pages the sharder has read, so these
figures overstate anonymous allocation. They are kept as measured; no
threshold (rule G5). The per-chunk bound itself is asserted structurally in
`test_g5_no_fp32_intermediate_exceeds_one_chunk`.

### G6, sharding time (REAL, this box only, rule V2)

| Double-quant | Wall | FP8 dequant | NF4 quantise | Other | s / source GB | Cache written |
|---|---|---|---|---|---|---|
| on | 134.21 s | 30.56 s (23%) | 95.43 s (71%) | 8.23 s | **25.71** | 3.59 GB |
| off | 122.11 s | 27.30 s (22%) | 84.78 s (69%) | 10.02 s | **23.39** | 3.75 GB |

NF4 quantisation ran on CPU (`quant_device="cpu"`; this box has no GPU). On a
CUDA box the sharder quantises on the GPU by default, so the 70% NF4 share does
not carry over. FP8 dequantisation is about 5.4 to 5.9 s per source GB here.
For scale only, different box and format: the brief's bf16 baseline was 138 GB
-> 36.4 GB in 217 s (1.6 s per source GB). A full DeepSeek-V3 (about 690 GB of source across 163 files) at this
box's rate would take about 4.9 hours; not a prediction for the dev box.

Raw JSON and void logs: [`results/b3-fp8-source/`](results/b3-fp8-source/). The sandbox working
directory is written as `$WORK/` in them; nothing else was edited. The
JSON `tree` stamp `f9b6c80` is the branch's hash at run time; the branch was
re-authored before it was pushed; the same code (`src/`, `tests/`, the
harness) is now `79a4d3f`.

## 3a. Amendment after review: G3 on the default path (rule committed 2026-10-11, before the run)

Review of PR #2 kept the G3 failure open: the gate applies to the default path,
and with double-quant on (the shipped default until then) it fails. The bound is
not changed. The default is: from `8e88b27`, a block-scaled FP8 source resolves an
unset `bnb_4bit_use_double_quant` to off, and an explicit `true` is refused by
name (`fp8_source.resolve_double_quant`).

**Rule for this run.** Same source file, same staging exclusion, same harness, same
G1 and G3 definitions and the same fixed `0.17` bound as section 1. The only
change is `--double-quant default`: the sharder is called with double-quant
unset, exactly as `soup train` does when the config does not set it. The record
stores both the requested setting and the value the sharder resolved.

- PASS iff G1 is `torch.equal` on every tensor, the resolved setting is off, and
  G3 holds in every 64-element block of every tensor.
- The double-quant-on results above stay in this record as the "before"
  evidence; nothing in sections 1 to 3 is edited.
- Validity rows V1 to V4 as in section 1.

### Result (REAL, 2026-10-11 16:33 UTC)

| Field | Value |
|---|---|
| Requested double-quant (`g6.double_quant_requested`) | `default` (unset) |
| Resolved by the sharder (`g6.double_quant`) | `false` (off) |
| G1 | **PASS, 126 / 126 tensors `torch.equal`** |
| G3 | **PASS, 126 / 126 tensors**, worst `e_b / a_b` **0.1531** (`layers.0.mlp.down_proj`), no failures, 0 zero blocks |
| **Verdict by the rule above** | **PASS** (G1 equal on every tensor, resolved setting off, G3 holds in every block) |

The per-tensor worst ratios are identical, all 126, to the double-quant-off run
in section 3, and the cache is the same size to the byte (3,748,021,400).

| | Wall | FP8 dequant | NF4 quantise | Other | s / source GB | Cache written | Peak RSS through sharding |
|---|---|---|---|---|---|---|---|
| default (resolved off) | 107.30 s | 28.78 s (27%) | 70.90 s (66%) | 7.62 s | **20.56** | 3.75 GB | 5.49 GB |

Box for this run (rule V1): tree `756ea99`, `src_dirty` false; same sandbox
class, source file, staging exclusion and library versions as section 2, 2
logical CPUs, 8.42 GB RAM, 17.7 GB disk free at start. The CPU stamp differs:
Intel Xeon @ 2.80 GHz here, @ 2.10 GHz in the section 3 runs.

Raw JSON and log: [`results/b3-fp8-source/dq-default.json`](results/b3-fp8-source/dq-default.json),
[`dq-default.log`](results/b3-fp8-source/dq-default.log); `$WORK/` convention as in section 3.

### Sharding time per GB (REAL)

| Run | Tree | CPU stamp | s / source GB |
|---|---|---|---|
| Double-quant on | `79a4d3f` | Xeon @ 2.10 GHz | 25.71 |
| Double-quant off | `79a4d3f` | Xeon @ 2.10 GHz | 23.39 |
| Default (resolved off) | `756ea99` | Xeon @ 2.80 GHz | 20.56 |

Box caveat as in section 3 (rule V2): a 2-vCPU CPU sandbox with NF4 quantised
on CPU, not comparable to the dev box. The default run does the same work as
the double-quant-off run and was stamped on a different CPU, so the 23.39 to
20.56 difference is the box, not the code.

## 4. Limits recorded for B2 (checked 2026-10-07, before the run)

- `soup train` refuses `model_type='deepseek_v3'` at the arch check, before
  sharding. End-to-end DeepSeek-V3 / K2 streaming needs B2.
- `shard_checkpoint` itself does not consult the arch; this record calls it
  directly.
- The MTP module (`model.layers.61.*`, `num_nextn_predict_layers: 1`) is sharded
  as an ordinary decoder layer (index `n_layers` 62 vs `num_hidden_layers` 61),
  carrying its own `embed_tokens` and `shared_head.head` into that layer shard.
- Dense layers (0 to 2) and MoE layers (3 to 60) produce different key sets.
- The pre-flight's shard-size estimate assumes a 16-bit source; for an FP8
  source it under-estimates the bf16 store by about 2x (planner, not B3).
- Sharding holds one decoder layer's blob in RAM before writing it. A full
  DeepSeek-V3 MoE layer is about 11 GB of FP8 source and about 6 GB as NF4, so
  sharding the real model needs a box with well over 8 GB of RAM (this one was
  8.4 GB and ran only the partial layer). Relevant to B1 and B6.
