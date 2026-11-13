# Native MLX dense Llama SFT qualification (#1992)

Source: `345c1f6105b8bd3bf3004f9c575b1517075080e9`, based on upstream
`9a8a5f39a632742f5abce7f9209d91223ba7638a`. This is a local qualification record;
no large-model memory or throughput claim follows from this tiny fixture.
The branch contains upstream `8b80c2db`, the helper-module registry fix, and
publication integration with upstream `1f1a3aaf`. The measured native trainers,
loader and native fixture remain byte-identical to the measured source. Later
upstream schema/CLI changes are covered by 442 publication integration checks;
the earlier 432-check integration result is retained in the verification record.

## Agreed scope and the four plan corrections

The [issue](https://github.com/MakazhanAlpamys/Soup/issues/1992),
[claim plan](https://github.com/MakazhanAlpamys/Soup/issues/1992#issuecomment-6848439995),
and [maintainer confirmation](https://github.com/MakazhanAlpamys/Soup/issues/1992#issuecomment-6857335834)
were read in full, including the timeline. The confirmed slice is dense native text
SFT, rank zero, no quantization, one native Llama layout, existing freeze semantics,
complete standalone checkpoints and the named unsupported-mode refusals. The
v0.78.0 release window is open; no broad schema or architecture support is implied.

1. **Memory:** trainable master weights use float32. AdamW's moments inherit that
   dtype, rather than being assumed to be float32 for every input dtype. Gradients
   are included in the formula below. The instrumented native peaks are measured
   at the shape below; 8B and 96/128 GB figures remain projections.
2. **Tests:** real parameter deltas, frozen tensor equality and optimizer-state
   coverage; tied weights; an independent summed-token cross-entropy/SGD reference
   with unequal supervised-token counts, accumulation and clipping; exact logits
   before/after fresh-process reload; exact saved tensor coverage at intermediate
   and final saves. The original fixture is deleted before any reload child starts.
3. **Floors:** all 46 feature checks pass on both the declared MLX floors and the
   implementation environment. No narrower version guard or dependency change is
   needed. Native tests are included in the existing `mlx-smoke` CI selection.
4. **Release:** implementation targets the confirmed v0.78.0 window. The changelog
   fragment lives under the latest released version, 0.77.2. Its provisional
   filename is the issue number; replace it with the actual PR number and append
   the PR credit when a PR exists. No PR number has been invented.

## Implementation choices supported by the controls

The adapter loop averages microbatch means and saves only trainable parameters.
The dense loop instead differentiates each supervised-token loss sum, accumulates
those gradients, divides by the group's supervised-token count, and applies the
existing clipping optimizer once. Complete saves use `save_model`, `save_config`
and `tokenizer.save_pretrained`; the effective config comes from the real model
loader. Reconstructing it from MLX dataclass defaults lost tokenizer-compatible
configuration and failed the fresh-process test.

Trainable parameters are promoted to float32. Disabling this promotion on the fp16
fixture produces a non-finite loss under AdamW; the mutation is retained. Frozen
parameters keep their input dtype and have no Adam moments. Tied embeddings/output
use the native single parameter entry, tested against a reference update.

## Hardware, fixture and versions

Apple M4, 16 GiB unified memory, macOS arm64, CPython 3.12.13. All data, weights and
tokenizer files are generated locally; there is no Hub model revision or download.
The fixture constructor is `tests/test_mlx_dora_native.py::_checkpoint`, seed 7:
Llama, 2 decoder layers, hidden size 64, intermediate size 128, 4 attention/KV
heads, model vocabulary 32. Four chat rows use batch size 1, maximum length 64
(actual padded width 33), accumulation 2 and two AdamW updates. The token-weighted
reference uses SGD to distinguish a correctly scaled gradient from Adam's first
step normalization. Checkpointing is exercised both enabled and disabled. Additional tied/untied
frozen-bottom-layer runs with checkpointing enabled pass on both environments;
the assert-based check is retained in `verification.json`.

- Current: MLX 0.32.4, MLX-LM 0.32.2, Transformers 5.22.2.
- Floor: MLX 0.31.2, MLX-LM 0.31.3, Transformers 5.16.1.
- Complete versions, source/test hashes, commands and exit codes:
  [current environment](evidence/issue1992/current/environment.json),
  [floor environment](evidence/issue1992/floor/environment.json).
- Each case JSON retains the fixture safetensors SHA-256, exact data/config,
  per-tensor maximum absolute deltas, actual moment dtype, counts and reload error.
  Default-valued config fields are omitted from committed case records; exact
  schema round-trip equality was asserted and each resolved config hash retained.

## Native observations

Each row is one instrumented lifecycle, not a performance distribution. Peak is
`mx.get_peak_memory()` after `mx.reset_peak_memory()` immediately before training;
it includes the retained initial weights and save-time logit observations. It is
MLX's active-allocation peak, not whole-process RSS or host memory. Prior test
allocations/cache and test order affect these tiny peaks, so they cannot establish
a memory speedup from freezing. The exact optimizer-state tensor membership does
establish that frozen parameters allocate no moments. State bytes include scalar
optimizer state as well as moments. Times cover the training loop and intermediate
saves, excluding model loading, the final root save and fresh-process reload.

| Runtime | Tied | Freeze selector | Trainable | Frozen | Optimizer state bytes | Peak MB | Loop ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| current | no | none | 86,336 | 0 | 690,700 | 3.610 | 19.65 |
| current | no | layers: 1 | 45,248 | 41,088 | 361,996 | 3.885 | 16.96 |
| current | no | ratio: 0.5 | 45,248 | 41,088 | 361,996 | 4.500 | 14.97 |
| current | yes | none | 84,288 | 0 | 674,316 | 4.069 | 14.28 |
| current | yes | layers: 1 | 43,200 | 41,088 | 345,612 | 4.113 | 15.06 |
| current | yes | ratio: 0.5 | 43,200 | 41,088 | 345,612 | 4.302 | 15.16 |
| floor | no | none | 86,336 | 0 | 690,700 | 3.811 | 27.87 |
| floor | no | layers: 1 | 45,248 | 41,088 | 361,996 | 2.520 | 17.47 |
| floor | no | ratio: 0.5 | 45,248 | 41,088 | 361,996 | 3.228 | 16.36 |
| floor | yes | none | 84,288 | 0 | 674,316 | 4.535 | 17.67 |
| floor | yes | layers: 1 | 43,200 | 41,088 | 345,612 | 2.776 | 18.53 |
| floor | yes | ratio: 0.5 | 43,200 | 41,088 | 345,612 | 3.483 | 18.01 |

All fresh-process reload logits are bit-identical to the observed logits before
saving, including the intermediate model, with every frozen tensor present.

## Memory projection, not a device-fit promise

For `P` total and `T` trainable parameters, persistent storage projects to:

```
trainable weights:       4 * T
frozen weights:         sum(frozen_parameter.nbytes)
gradients:              4 * T
AdamW first+second:      8 * T
persistent subtotal:   16 * T + frozen_weight_bytes
```

All-trainable 8B projects to 128 GB decimal before activations, accumulation,
transient buffers, caches and the operating system. With bf16 frozen parameters,
the subtotal is `16*T + 2*(P-T)`. Freezing 75% of decoder blocks does not freeze
75% of model parameters because embeddings, final norm and head remain trainable.
Neither a 96 GB nor a 128 GB device has been qualified here, nor has an 8B model.

## Verification and sensitivity

- Original public-config control: three cases failed on upstream's explicit
  `lora.r=0`/MLX refusal before implementation.
- Feature suite: 46 passed on current and floor versions at the source SHA above.
- Native acceptance uses real outputs and parameters; portable config/CLI-refusal
  tests run without MLX. No native results are attributed to mocks.
- [Six mutation controls](evidence/issue1992/mutations.json) fail for the intended
  reason: last-block-only training, missing freezing, equal microbatch weighting,
  partial saving, disabled float32 masters, and duplicate tied updates.
- Ruff, generated-reference structure, recipe snapshot regeneration and package
  listing checks pass. The recipe snapshot has no diff; no defaults/fields changed.
- Pinned full non-smoke run: **56,771 passed, 391 skipped, 37 failed** in 1049.98 s.
  One failure was the missing helper declaration in the backend-support registry;
  it is fixed and the 432 post-integration checks pass. The other 36 reproduce on
  unchanged upstream: 20 directly, 16 with long temporary paths that make Rich
  truncate `prompts.jsonl` from a diagnostic. The full suite was not repeated after
  the registry fix; it is not reported as green. Exact cases, baseline conditions
  and commands are in [verification.json](evidence/issue1992/verification.json).

## Reproduction

From the source revision above, on Apple Silicon with the MLX extra installed:

```bash
PYTHONPATH=src pytest tests/test_mlx_full_finetune.py tests/test_mlx_full_finetune_native.py -q --no-cov
SOUP_1992_EVIDENCE_DIR=/tmp/issue1992-evidence PYTHONPATH=src pytest tests/test_mlx_full_finetune_native.py -q --no-cov
ruff check src/ tests/
PYTHONPATH=src python scripts/gen_reference_docs.py --check-structure
PYTHONPATH=src python scripts/generate_recipe_snapshot.py
```

Repeat the feature command in an isolated environment with `mlx==0.31.2`,
`mlx-lm==0.31.3` and `transformers==5.16.1` to exercise the declared floor.
