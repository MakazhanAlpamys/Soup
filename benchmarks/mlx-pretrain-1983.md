# Native continued pretraining qualification (#1983)

Measured 2026-11-01 on Apple M4 Pro, 48 GiB unified memory, real Metal GPU.
Python 3.12.7; MLX 0.32.4; Transformers 5.22.1; tokenizers 0.23.3; NumPy 2.5.4.
Two isolated `.[mlx]` environments qualify MLX-LM **0.31.3** (declared floor)
and **0.32.2** (current installed compatible release). Neither environment
contains torch, TRL, PEFT or datasets.

Source base: `c38330cfae3c91e8d5ec880e8966a387ee025a74` plus the implementation
in this change. The exact working `src/**/*.py` content is identified by tree
SHA-256 `a42a175007fd15de04f69eabda1cfc492eeed11d10cbbc2e876e666ee0e057a5`;
the harness derives it from the sorted path-to-file-SHA256 manifest. This avoids
misrepresenting an uncommitted patch as the unmodified base SHA.

## Fixture and evidence boundary

A randomly initialized local Llama (one layer, hidden size 32, intermediate 64,
vocabulary 32, tied embeddings; base seed 13) and real saved WordLevel tokenizer.
The 4-bit base is quantized by MLX-LM, saved, and reloaded by the normal Soup
loader. Actual tensors are affine 4-bit `QuantizedLinear` (7) and
`QuantizedEmbedding` (1), group size 32; base tensors remain frozen. The tiny
group incurs scale/bias overhead: this is not a 0.5-byte-per-parameter memory
claim.

The dense base and LoRA adapters use float32; the quantized base stores uint32
packed weights with float32 scales/biases. Artifact dtype inspection is recorded
alongside model-file hashes in the JSON record.

The independent train/validation JSONL files have 3/2 accepted documents,
0 drops, 3/2 windows and 24/15 valid targets respectively. Tokenized counts
are 27/17, with one context-only token per document. PAD equals EOS.
Context limit is 64 (observed longest train window 18 tokens).
This lifecycle fixture does not exercise a long context or establish domain
quality. The portable exact-array tests separately pin long windows, tails,
overlap and document splitting.

Eight epochs, batch size 2, accumulation 2, one complete update per epoch,
192 supervised targets in total. AdamW, constant LR 0.03, zero weight decay
and warmup, gradient clip 1; LoRA rank 4, scale 2, dropout 0, resolved Q/V
targets; initialization seed 7, data seed 11. Evaluation/save every 3 completed
updates, with final evaluation at update 8. The last microbatch contains one
window. All losses below are actual held-out token-weighted causal losses.

| MLX-LM | Base | Initial → final held-out loss | Updates / targets | Max fresh-reload logit error | Max trained-vs-base logit change |
|---|---|---|---|---|---|
| 0.31.3 | dense | 4.666231 → 1.483473 | 8 / 192 | 0 | 5.286923 |
| 0.31.3 | affine 4-bit | 4.690506 → 1.503341 | 8 / 192 | 0 | 4.902772 |
| 0.32.2 | dense | 4.666231 → 1.483473 | 8 / 192 | 0 | 5.286923 |
| 0.32.2 | affine 4-bit | 4.690506 → 1.503341 | 8 / 192 | 0 | 4.902772 |

All dense base tensors (11) and quantized base tensors (27) remain bit-identical;
LoRA B moves from zero. A new Python process uses the shared native adapter
consumer to load the final artifact, with `atol=rtol=1e-5`. Measured maximum
absolute reload error is zero in all four runs. Model, tokenizer and corpus
hashes, effective settings, intermediate validation values, loading/preparation
and warmup times, tokens/s, end-to-end time and peak MLX allocation are in
[`evidence/mlx-pretrain-1983.json`](evidence/mlx-pretrain-1983.json).

Runs take about two seconds including a fresh-process reload. Their per-update
throughput is a tiny-fixture observation, not a stable speed comparison between
versions: eager runtime/kernel preparation, shape changes and shared host load
matter at this scale. One real inference-mode forward/backward warmup is measured
separately and does not advance dropout RNG or optimizer state. There is no
explicit `mx.compile` stage. Peak MLX allocation is not total process RSS and
excludes loading transients. No CUDA-parity or domain-adaptation claim is made.

## Reproduction

From this source checkout, in an Apple Silicon `.[mlx]` environment with pytest:

```bash
python -m pytest -o addopts= tests/test_mlx_pretrain.py tests/test_mlx_pretrain_native.py -q
```

`tests/test_mlx_pretrain_native.py::save_base` creates the local model and
tokenizer reproducibly; `loaded` writes the fixture corpora and returns a
public-configured wrapper. To reproduce the recorded eight-update run, use
`loaded(path, quantized=False, epochs=8, batch_size=2,
gradient_accumulation_steps=2, eval_steps=3, save_steps=3)` (and `quantized=True`
for the second base), serialize its config with
`model_dump(mode="json", exclude_unset=True)` to YAML, then:

```bash
python benchmarks/qualify_mlx_pretrain.py --config fixture/soup.yaml --report evidence.json
```

The harness itself only needs `.[mlx]`; pytest is needed by the test-fixture
factory. It also accepts another correctly configured local/hub MLX base and
corpus, but retains pre-update arrays for its freeze check, so use a bounded
model for qualification.

## Tests and mutation sensitivity

The original checkout failed all three public route tests (both dense/4-bit
YAML configurations and native trainer lookup) for the missing schema/route.
The implemented portable/native suite passes on both declared versions.
It includes independent float64 CE/analytic gradients, unequal-length
accumulation against a concatenated reference with binding/nonbinding clipping,
short evaluation tails, seeded repeatability, real optimizer counts/cadence,
explicit and `--resume auto` warm starts, strict tensor/identity checks and
failed writes.

Each deliberate mutation is applied to an isolated copy of the source, leaving
the tested worktree unchanged:

| Mutation | Failing test / evidence |
|---|---|
| Discard the final short window | Exact window list loses `(13, EOS)` |
| Carry two context tokens instead of one | Exact input/target windows duplicate a supervised pair |
| Mask by `target != pad_id` | Exact masks lose genuine EOS targets when PAD=EOS |
| Normalize gradients by microbatch count | Nonbinding-clip update differs from independent token-weighted SGD reference |
| Save unresolved `keys: [auto]` | Fresh-process native consumer refuses unique unresolved target keys |

Required project lint passes. The complete default non-smoke serial command
`pytest tests/ -v --tb=short` passes: **48,566 passed, 590 skipped, 10 smoke
deselected**, 87.70% coverage (77% required), 26m 15s. The 590 skips are optional
dependency/hardware cases; they are separate from the two clean-environment
native runs (47 tests passed per MLX-LM version). Native results do not substitute
for the complete suite.
