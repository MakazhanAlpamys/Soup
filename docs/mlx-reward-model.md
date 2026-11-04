# Native MLX reward-model components (#2003)

`backend: mlx`, `task: reward_model` trains a scalar reward head and ordinary
LoRA on a dense built-in Llama base. Both `soup train` and `soup sweep` use the
native trainer. This is a scoring artifact, not a generation adapter or a
PPO/GRPO reward-model integration. The dependency floor is unchanged (#2471).

Minimal configuration (raw JSONL rows with `prompt`, `chosen`, `rejected` strings):

```yaml
base: /local/dense-llama
backend: mlx
task: reward_model
output: ./reward-output
data:
  train: ./pairs.jsonl
  format: dpo
  max_length: 512
  val_split: 0.1
training:
  quantization: none
  rewind_log: false
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 2
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    use_rslora: false
    target_modules: [q_proj, v_proj]
```

Run `soup train --config soup.yaml --yes`; score the saved bundle with the API
below. Tests in `tests/test_mlx_reward_model.py` use offline tiny models and
are included in the existing MLX smoke CI job. Pretrained measurements are
separate PR evidence, not downloads in CI.

## Scoring API

```python
from soup_cli.utils.mlx_reward import load_reward_scorer, score_completions

scorer = load_reward_scorer("./reward-output")
scores = score_completions(scorer, [("Question", "An answer"), ("", "Another answer")], batch_size=2)
```

Reuse the returned handle. Each `(prompt, completion)` item produces one finite
float in input order. All inputs are validated before the first forward. Empty
input returns `[]`, but invalid `batch_size` is still an error. A prompt may be
empty; completions must contain meaningful non-EOS text. Scoring refuses an
input longer than the saved maximum; it never silently truncates or drops one.

The module imports only stdlib at import time. Loading lazily requires the MLX
extra and a public fast HF tokenizer. It uses local files only, not a network
fallback. `base="/relocated/base"` can relocate an identical base: config and all
weight bytes must match. A relocated base need not contain tokenizer files;
scoring always uses the bundled tokenizer. Each load hashes the base once, an
O(base bytes) read; each scoring call reuses the loaded model without rehashing.

## Training and numerical contract

`MLXRewardModelTrainerWrapper` subclasses `MLXSFTTrainerWrapper`, reusing
`_require_mlx` and `_apply_lora`, target resolution/scaling, the optimizer planner
and builder, global gradient clipping, seed resolution and loss summaries.
It does not call SFT `setup`, `_check_unsupported`, or the SFT/DPO training loops.
SFT's assistant-mask warnings and advice about 4-bit LM models are inappropriate
for this scalar loss. Existing shared schema checks still run; the final
`config.mlx_reward.validate_reward_scope` check refuses unapplied non-default
training, LoRA and data settings before loading. The same native optimizer and
scheduler support as SFT is used. Dropout is applied, and rsLoRA uses SFT's
`alpha / sqrt(rank)` scaling; both survive save/reload.

Extra restrictions reflect the dense scalar bundle contract: `quantization`
must be `none`; a loaded quantized base is also refused. Ordinary LoRA targets
must be a unique subset of the seven Llama attention/MLP projections, rank
1–4096 and positive alpha, with dropout in `[0, 1)`. DoRA/alternative adapter
initialization, full fine-tuning, streaming and tokenized data are outside
this slice. Raw pairs use `format: auto` or `dpo`; chat/token-mask tunables do
not define a scalar loss and non-default values are refused. `eval_steps` is
refused because validation is per epoch. `training.eval_gate.enabled` and
`eval.auto_eval` are refused because their LM evaluation paths cannot score
these bundles. Set `rewind_log: false` explicitly: SFT's default-on recorder
requires per-row token losses, which this pairwise scalar loop does not emit.
Project defaults are not changed. Other task-only knobs retain their neutral
schema defaults; changing them is refused rather than silently ignored.

The qualified configuration is a dense built-in Llama with ordinary Q/V LoRA,
dropout 0 and AdamW. The loaded architecture and modules are checked, including
absence of quantized linear/embedding modules. The base is frozen before LoRA
and the head are attached. The head is bias-free, fp32, shape `[1, hidden_size]`,
normal initialization with standard deviation 0.02 and a dedicated MLX key
from the resolved training seed (unset = 42). It pools the **inner Llama final
norm output at terminal EOS**, casts that vector to fp32, and applies the head.
It never computes vocabulary logits or a class softmax.

The loss is `mean(logaddexp(0, -(chosen_score - rejected_score)))`; dataset margin
columns and reward centering are not used. Every retained pair appears once
per epoch. A private shuffle RNG keeps tail microbatches and tail accumulation
windows. The number of optimizer updates is
`epochs * ceil(ceil(retained_pairs / batch_size) / accumulation_steps)`.
`batch_size: auto` resolves to 1. Microbatch means are multiplied by actual pair
count, summed, divided by window pair count, then clipped once globally before
one optimizer update. Scheduler, save and log steps count completed updates.
Validation is pair-weighted at epoch boundaries (including the final boundary),
restoring model mode and MLX RNG. Saving also preserves the training RNG.
Boolean gradient checkpointing recomputes the inner trunk without modifying
MLX-LM's global decoder class.

### Text, length and filtering

Join `prompt + " " + completion` for a nonempty prompt, otherwise completion.
Tokenize the whole string once with `add_special_tokens=True` and offsets;
append a missing terminal EOS. Training truncates on the right, reserving one
EOS slot. A surviving answer token must overlap non-whitespace completion text;
a BPE token starting at the preceding separator still counts. BOS/EOS tokens
with zero offsets cannot count. Right/left padding is normalized using explicit
masks, never `token_id != pad_id`.

Malformed/non-string/blank/EOS-only answers stop before any updates. Both sides
are checked before deciding whether to skip the pair. Pairs with a lost answer
or identical retained sequences are skipped and counted separately per split,
with the first original normalized row index/side and remedy reported. No
retained train rows, or a supplied nonempty validation split filtered to zero,
stops training. Row and scoring item indices are zero-based; diagnostics never
include sample text.

Pinned example: `HuggingFaceTB/SmolLM2-135M-Instruct` at
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, EOS = PAD = 2, no automatic BOS:

- `Color: red blue green` yields `[13646,42,2382,4461,2654,2]`; replacing green
  with yellow gives 5724 instead of 2654. At M=5 both truncate to
  `[13646,42,2382,4461,2]`, so the pair is skipped as collapsed.
- At M=3, prompt `Color:` and answer `red` lose the answer and are skipped.
- Padded `Color: red`: tokens `[13646,42,2382,2,2]`, mask `[1,1,1,1,0]`, pool
  index 3. M=3/M=5 illustrate helpers only; the public schema minimum stays 64.

Compared with the pinned TRL 0.29.1 / Transformers path: TRL filters overlength
pairs instead of truncating; EOS=PAD pooling selects the last non-pad-id token
(index 2 above), not the native terminal EOS; TRL disables dropout by default,
whereas the reused native LoRA helper consumes configured dropout; short Transformers microbatches are batch-weighted
rather than pair-weighted. **No cross-backend score equality or calibration is
claimed.**

## Saved generations and read boundaries

```text
output/reward_model.json
output/bundles/<32 lowercase UUID4 hex characters>/manifest.json
output/bundles/<id>/reward.safetensors
output/bundles/<id>/tokenizer/...
```

Each save interval and final save publishes a complete scoring generation,
including the completed update count. These are not optimizer/RNG checkpoints.
Only one training writer per output root is supported. The writer validates an
existing generation first; corruption is refused, never silently overwritten.
Legacy model/adapter files in the root are refused without deletion.

Publication stages under `bundles/.staging-<id>/` on the same filesystem, checks
serialized tensors and saved tokenizer, renames the immutable payload, then
atomically replaces the root pointer **last**. A pre-publication failure leaves
the earlier pointer readable. Published generations are retained; only this
save's unpublished staging/orphan is cleaned on failure. Disk use therefore
grows with the number of saves. `soup_run.json` is auxiliary: the bundle's own
base identity is authoritative. This is not a promise of filesystem durability
against power loss or unchanged auxiliary CLI logs after a crash.

Read-time checks include:

- Exact version/type and pointer fields. `bundle` must match
  `bundles/[0-9a-f]{32}`; absolute paths, backslashes and traversal are refused.
- `realpath`/`commonpath` containment, linked-directory/reparse checks and
  checked non-following descriptors; bundle hardlinks are refused too.
- Pointer, manifest and safetensors header bounded to 4 MiB; duplicate JSON
  keys, nonfinite numbers and malformed tensor byte ranges are refused.
- Exact actual member inventory, including every file below `tokenizer/`,
  SHA-256/size checks, local fast tokenizer reload with remote code disabled,
  and saved EOS/vocabulary/length/pooling contract checks. Extra tokenizer
  files, linked directories or changed assets fail before tokenizer loading.
- The full `model*.safetensors` glob used by MLX-LM, config and optional shard
  index are hashed. Missing/extra indexed shards, duplicate tensor keys,
  unsupported weight formats and quantization sidecars are refused. HF cache
  symlinks are allowed only at this separate base boundary. `model_file` and
  model/tokenizer `auto_map` are refused before either model loader, including
  MLX-LM 0.31.3's unguarded `model_file` route.
- LoRA metadata has bounded rank (4096), layers (65536), target count (1024)
  and key length (256). Expected tensor names/shapes/dtypes come from the
  checked base, attached LoRA and head, not from manifest declarations alone.
  The payload is hashed, its header checked, and the **same descriptor**
  rewound for `mx.load(..., format="safetensors")`; finite values are checked
  before applying every exact expected tensor. No pickle or custom code.

The trusted local base tree is not protected against a hostile process
concurrently replacing the whole tree. Scoring does not fall back to an older
published generation when the selected one is corrupt.

## Consumers and CLI boundaries

One light helper, `refuse_reward_artifact`, recognizes a root by `lexists` on
`reward_model.json` before parsing, including a directory, broken link or
truncated file with that name. A payload is recognized by `reward.safetensors`
or a bounded no-follow manifest declaring `artifact_type: mlx_reward_model`.
The helper also checks local bases recorded in LM adapter metadata before a
consumer can start output or model work; resolved-base seams check again.

The same change guards chat; infer before backend dispatch; serve and named
adapters; merge; export including ONNX; push; model-loading eval routes; draft;
shrink; live evaluation; native MLX loading; and adapter info, compare, diff,
merge, arithmetic, blame, scan, check and audit. Recursive adapter listing
prunes reward roots/payloads and their subtrees. Config-only branch/checkout and
file-only signing/safety/report commands retain their semantics.

The training CLI rejects `--push-as`, `--gate`, `--diagnose-gate` and
`--capture-activations`, with resume refusals taking precedence, and replaces
all four chat/push/merge/export hints with the Python scorer API. Public
subprocess tests cover these refusals and successful offline train/scoring;
the real sweep execution path is also tested without a mocked dispatch.

## Final diagnostic catalog

These are `ValueError` messages unless stated otherwise. `{member}`, `{key}`
and other untrusted values are escaped and bounded. Sample text is never shown.
The two row-filter messages below skip/count; all other invalid row diagnostics
stop before training. This catalog replaces the split v1/v2 proposal lists.

```text
MLX reward: batch_size must be a positive integer, not bool. Set batch_size >= 1.
MLX reward: items must be a sequence of (prompt, completion) pairs. Supply a list or tuple of pairs.
MLX reward item {i}: expected a (prompt, completion) pair. Supply exactly two strings.
MLX reward item {i} {field}: expected a string. Convert this field to text.
MLX reward item {i} completion: blank text. Supply a nonblank completion.
MLX reward item {i} completion: no non-EOS content tokens. Supply a meaningful completion.
MLX reward item {i}: {N} tokens including terminal EOS exceed saved max_length={M}. Shorten the input or use a bundle trained with a larger data.max_length.
MLX reward item {i}: non-finite score. Check the bundle and model numerics.
reward_model {split} row {i}: expected a prompt/chosen/rejected mapping. Supply a preference row.
reward_model {split} row {i} {side}: required field is missing. Supply this answer.
reward_model {split} row {i} {field}: expected a string. Supply string prompt/chosen/rejected fields.
reward_model {split} row {i} {side}: blank answer. Supply a nonblank answer.
reward_model {split} row {i} {side}: no non-EOS content tokens before truncation. Supply a meaningful answer.
reward_model {split} row {i} {side}: completion lost at max_length={M}. Raise data.max_length or shorten the prompt.
reward_model {split} row {i} chosen/rejected: identical after tokenization/truncation. Supply distinguishable answers; if truncated, raise data.max_length or shorten the prompt.
reward_model {split}: no usable preference rows remain. Correct the reported rows or length setting.
MLX reward: fast tokenizer with offsets and EOS required.
MLX reward: dense Llama backbone required.
MLX reward: scoring-only bundle; use load_reward_scorer/score_completions.
MLX reward: output contains a model or adapter file {member}. Choose a new output directory.
MLX reward: base member {member} is unavailable. Provide the identical local/cached base with base=.
MLX reward: base member {member}: expected SHA-256 {expected}, found {found}. Provide the matching base with base=.
MLX reward: bundle member {member}: {reason}. Restore an intact bundle.
MLX reward: tensor {key}: expected shape={expected_shape}, dtype={expected_dtype}; found shape={found_shape}, dtype={found_dtype}. Restore an intact bundle.
MLX reward: tensor {key}: {missing|unexpected|non-finite values}. Restore an intact bundle.
MLX reward: {member} format_version: expected 1, found {value}. Use a compatible intact bundle.
MLX reward_model does not support {flag}. Remove {flag}; use load_reward_scorer/score_completions for scoring.
--resume is not supported for task 'reward_model' on backend 'mlx': scoring bundles do not save optimizer/RNG state. Start a fresh run in a new output directory.
```

`--hf-resume` substitutes its flag name into the same resume message.
Dependency failures are `ImportError` with exactly:

```text
MLX backend requires the 'mlx' and 'mlx-lm' packages. Install with: pip install "soup-cli[mlx]"
```

Numerical failures during train/validation are `FloatingPointError`:

```text
reward_model train: non-finite loss. Check model numerics.
reward_model val: non-finite score. Check model numerics.
```

The internal planner requires positive integer dimensions and reports
`reward_model: planner dimensions must be positive integers.` Calling `train`
before setup reports `MLX backend: setup(dataset) must be called before train()`
as `RuntimeError`.

The final task-specific schema diagnostics are:

```text
MLX reward_model does not support {field}={value}. Set {field} to {allowed}.
MLX reward_model does not support training.eval_gate.enabled. Set training.eval_gate.enabled: false; use the reward trainer validation metrics.
MLX reward_model does not support eval.auto_eval. Set eval.auto_eval: false; held-out preference evaluation runs in the reward trainer.
```

The generic refusal names the literal field and escaped value. `{allowed}` is
`none for a dense base` for quantization; `false; per-row token-loss recording
is SFT-only` for rewind logging; `false` for reward quantization; `null; supply
raw preference rows` for `data.tokenized_path`; `auto or dpo (raw
prompt/chosen/rejected strings)` for format; `false; native reward epochs
shuffle retained pairs` for streaming; `true or false` for checkpointing;
`an integer in [1, 4096]` for rank; `a finite positive number` for alpha;
`a finite number in [0, 1)` for dropout; `auto or a nonempty unique subset of
mlp.down_proj, mlp.gate_proj, mlp.up_proj, self_attn.k_proj, self_attn.o_proj,
self_attn.q_proj, self_attn.v_proj` for targets; `text (raw preference pairs)`
for modality. For other unapplied changed settings it is the schema default's
Python representation. Existing earlier schema and shared optimizer diagnostics
retain precedence; they are not rewritten by this feature.
