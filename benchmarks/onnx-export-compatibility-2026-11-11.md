# ONNX exporter compatibility qualification — 2026-11-11

This is Linux / Python 3.12.3 CPU evidence for #2676, using the validation implementation
at `bb3c83180a7b91f8445a7f608f441d1db254b2df`. No model was downloaded. No dependency bound,
installed training package or library function was changed to make an export pass.
The checked-in harness is `benchmarks/harness/onnx_export_compat.py`.
The [raw evidence](evidence/onnx-export-2676/) includes the input manifest, JSON
comparisons, real CLI logs, corruption controls and the export-only package freeze.

## What worked and what did not

| Method | Result | Limit |
|---|---|---|
| Released Optimum 2.3.0 / optimum-onnx 0.1.0 with Transformers 5.19.0 | Import fails on `get_parameter_dtype` | The ONNX package declares `transformers<4.58`; this diagnostic overlay is not a resolved stack |
| Old Optimum 1.16.0 with Transformers 5.19.0 | Import fails on `is_tf_available` | Its loose metadata does not establish runtime compatibility |
| Upstream ONNX PR 121 + upstream core, Transformers 5.19.0 | Import succeeds; GPT-2 export fails on missing `cache_position` | An unmerged Transformers-5.2 proposal does not qualify 5.19 |
| Released packages in a fresh export-only venv | Full `soup export --validate-onnx` passes for both fixtures | The checkpoint's config and tokenizer must be readable by Transformers 4.57.6 |
| Native `torch.onnx.export`, original Transformers 5.19.0 environment | Both original fixtures pass ONNX Runtime comparison | Lab prototype only; Soup's export backend/default has not changed |

The upstream candidate was pinned, not followed as a moving branch:

- [optimum-onnx PR 121](https://github.com/huggingface/optimum-onnx/pull/121),
  source `7c9ccd7331403147c6b2a9f9440cd31f470a08f4`;
- optimum core source `87ba58464371163e4f23d8fcc155141bab8c55de`.

The released venv was installed normally with `soup-cli[onnx]`, not with `--no-deps`, and
`pip check` returned **No broken requirements found**. Its key versions were torch
2.6.0+cpu, transformers 4.57.6, optimum 2.1.0, optimum-onnx 0.1.0, onnx 1.23.2,
onnxruntime 1.31.1. The training/native environment kept torch 2.14.1 and transformers
5.19.0. The package inventories are recorded with the evidence.
The native lane was also rerun with only normally installed ONNX/ONNX Runtime and
their dependencies added to the training environment's import path: no Optimum
distribution was present, none was imported, and `pip check` passed. Both native
numerical results remained exactly the same as the initial lab run.

## Inputs and interpretation

Two deterministic, randomly initialized **tiny** causal LMs, seed 2676: GPT-2
(one layer, hidden 32, four heads) and Llama (one layer, hidden 32, intermediate 64,
four query/two KV heads), vocabulary 8. Four explicit prompts tokenize to lengths
1, 2, 3 and 4: 80 logits elements per model. CPU float32, no padding, no KV cache,
no chat template. This is numerical smoke coverage, not language quality, large-model
feasibility, cached-generation or Windows/macOS qualification.

The fixtures are created in the Transformers-5.19 environment and store its reference
token IDs and logits independently in NPZ files. The harness also checks that the
export-side tokenizer reproduces those token IDs, then compares every ONNX logits
element against the saved training-side reference.

**Tokenizer control:** the original generic 5.x fixture saves `tokenizer_class` as
`TokenizersBackend`. The released 4.x loader cannot load it: the unchanged artifact's
full command exits 3 at validation. The `legacy-copy` lab lane makes a separate
copy with the historical generic name `PreTrainedTokenizerFast`; model weights and
`tokenizer.json` remain unchanged. This is an explicit synthetic-fixture serialization
control, not a migration recipe for arbitrary production tokenizers. The native lane
uses the untouched original fixture and needs no such change.

## Results

Errors below are against the stored **training-environment** reference, at `atol=1e-4`,
with no relative tolerance. The source weights remain unchanged.

| Path | Fixture | Max absolute error | Mean absolute error | Result |
|---|---|---:|---:|---|
| Released export-only venv, legacy-compatible fixture copy | GPT-2 | 5.9604645e-8 | 1.4662510e-8 | PASS; real CLI exit 0 |
| Released export-only venv, legacy-compatible fixture copy | Llama | 7.4505806e-8 | 1.5224214e-8 | PASS; real CLI exit 0 |
| Native prototype, unchanged 5.x fixture | GPT-2 | 5.9604645e-8 | 1.4941907e-8 | PASS |
| Native prototype, unchanged 5.x fixture | Llama | 7.4505806e-8 | 1.5224214e-8 | PASS |

Four copied graphs, one per successful path/model, then have their floating-point
initializers altered. All four are refused by the checker: max error about **8.376**
for GPT-2 and **24.360** for Llama. Originals are retained. Separately, the real
Optimum-to-Soup GPT-2 command at `--atol 0` exits **2**, prints FAIL and does not print
the completion panel. This is a measured strict-tolerance control, not an assertion
that every model/platform must have nonzero rounding error. Existing CLI unit tests
cover registry non-attachment and mutation of the refusal itself.

## Reproduce

From this checkout in a torch/Transformers-5.19 environment with ONNX and ONNX Runtime
available, use a fresh scratch directory; the harness refuses existing output trees:

```bash
python benchmarks/harness/onnx_export_compat.py create --root ./onnx-lab/fixtures
python benchmarks/harness/onnx_export_compat.py native --root ./onnx-lab/fixtures \
  --name gpt2 --graph ./onnx-lab/native-gpt2 --report ./onnx-lab/native-gpt2.json
python benchmarks/harness/onnx_export_compat.py native --root ./onnx-lab/fixtures \
  --name llama --graph ./onnx-lab/native-llama --report ./onnx-lab/native-llama.json
```

Use the separate venv setup in `docs/serving-and-export.md`, then prepare only the
explicit compatibility fixture copy and run the actual command:

```bash
python benchmarks/harness/onnx_export_compat.py legacy-copy --root ./onnx-lab/fixtures \
  --name gpt2 --output ./onnx-lab/gpt2-compatible
.venv-onnx/bin/python -m soup_cli.cli export --model ./onnx-lab/gpt2-compatible \
  --format onnx --validate-onnx --prompts ./onnx-lab/fixtures/prompts.txt \
  --output ./onnx-lab/optimum-gpt2
.venv-onnx/bin/python benchmarks/harness/onnx_export_compat.py inspect \
  --root ./onnx-lab/fixtures --name gpt2 --source ./onnx-lab/gpt2-compatible \
  --graph ./onnx-lab/optimum-gpt2 --report ./onnx-lab/optimum-gpt2.json
```

Repeat that lane with `llama`. To inject a numerical error into a copied native graph:

```bash
python benchmarks/harness/onnx_export_compat.py corrupt --root ./onnx-lab/fixtures \
  --name gpt2 --graph ./onnx-lab/native-gpt2 --output ./onnx-lab/corrupted-gpt2
python benchmarks/harness/onnx_export_compat.py inspect --root ./onnx-lab/fixtures \
  --name gpt2 --graph ./onnx-lab/corrupted-gpt2 --report ./onnx-lab/corrupted-gpt2.json
# The inspect command must exit 2, with pass:false in its report.
```

Set `PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1` for the recorded CPU/offline protocol. No pretrained model
cache is needed. Install/build dependencies before switching network access off.

## Support decision — 2026-11-12

The maintainer [confirmed the separate export-only path](https://github.com/MakazhanAlpamys/Soup/issues/2676#issuecomment-6816046820):
keep training and merging in the training environment, and install `soup-cli[onnx]`
in its own environment without `[train]`. The `[onnx]` requirement remains
`optimum[onnxruntime]>=1.16.0`; dependency bounds and the export backend do not change.
The opt-in check in #2695 remains scoped to that export-only environment.

The guide documents the two-environment workflow and the known limit of
Transformers-5-saved tokenizers that the 4.x loader cannot read. The measurements
above remain author-reported qualification records; the support decision is not
a claim that maintainers reran them or that every architecture is compatible.

A native exporter for Transformers 5 is a separate question requiring its own
issue and optional-dependency decision. It is not requested for #2676, and the
prototype in this report remains a lab harness rather than a Soup export backend.
