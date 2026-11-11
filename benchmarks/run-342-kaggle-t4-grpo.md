# Kaggle T4 GRPO watchdog and capacity record (#342)

Four FP16 GRPO arms ran on single-device Kaggle Tesla T4 sessions on 2026-11-07
(UTC): two short smoke runs, one capacity attempt, and a longer capture run with
one watchdog trip. A separate fixed-batch replay used two T4s with the model
sharded across them. The inputs were synthetic. The first three arms used direct
`TRL.GRPOTrainer` calls with Soup's watchdog attached explicitly. The capture arm
used Soup's GRPO wrapper. These results provide no evidence for or against the
original H100 BF16 failure in [#342].

## Recorded environment and settings

**Measured (Kaggle reports and install log).** All four arms used `cuda:0`, a Tesla T4
(SM 7.5, 15,636,037,632 bytes reported by PyTorch, about 14.56 GiB visible), on Linux
with Python 3.13.16, PyTorch `2.11.0+cu130` (CUDA build 13.0), Transformers `5.20.2`,
TRL `0.29.1`, PEFT `0.21.2`, Accelerate `1.15.0`, bitsandbytes `0.50.2`, Datasets
`4.8.5`, and Soup `0.77.2`. The retained reports did not capture the driver version
or a Hugging Face model revision.

| Kaggle script version | Model | Soup commit checked out | Base load | `per_device_train_batch_size` | `num_generations` | `max_completion_length` | `max_steps` |
|---|---|---|---|---:|---:|---:|---:|
| 5 | `Qwen/Qwen2.5-0.5B-Instruct` | [`9ca9819747a624da4cbbff03cc8e1a7d2a81100d`][soup-v5] | FP16, no quantization config | 2 | 2 | 64 | 24 |
| 6 | `Qwen/Qwen2.5-7B-Instruct` | [`bcd9898b62e62ac6d74c9d985b40b3bf19155b5d`][soup-v7b] | NF4 double quantization, FP16 compute | 8 | 8 | 512 | 16 requested |
| 9 | `Qwen/Qwen2.5-7B-Instruct` | [`bcd9898b62e62ac6d74c9d985b40b3bf19155b5d`][soup-v7b] | NF4 double quantization, FP16 compute | 2 | 2 | 512 | 16 |

**Source reading (saved launch settings, versions 5, 6, and 9 above).** The common explicit `GRPOConfig` settings
were `gradient_accumulation_steps=1`, `learning_rate=2e-4`, `beta=0.02`,
`logging_steps=1`, `save_strategy="no"`, `report_to="none"`, `fp16=True`, and
`bf16=False`. The 7B scripts explicitly enabled gradient checkpointing and ran
`prepare_model_for_kbit_training(..., use_gradient_checkpointing=True)` on the
quantized base before applying LoRA. The 0.5B script left the checkpointing argument
at the TRL default. Version 9 set the allocator default to `expandable_segments:True`
when `PYTORCH_CUDA_ALLOC_CONF` was unset. Version 6 did not set that default.

All arms used causal-LM LoRA `r=32`, `alpha=64` on `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, and `down_proj`. The tokenizer padded on the left
and used EOS as pad when it had no pad token. The dataset was 32 prompts of the form
`Answer only with the integer result of {n} + {n}.` for `n` in `range(32)`.
The reward function returned `float(index % 2)` in completion order. It did not
score answer correctness. Each script checked out the commit shown above, installed
its `[train]` extra, removed the preinstalled `torchao==0.10.0`, and imported
`soup_cli` from the checkout. Both direct 7B scripts used `device_map={"": 0}`
and `low_cpu_mem_usage=True` in `AutoModelForCausalLM.from_pretrained`.

### Fourth arm: longer capture run

**Measured (final capture report timestamps).** The retained report records
`started_at_ns=1794049208993816302` and `finished_at_ns=1794073097407026880`.
Converted to UTC, these are 2026-11-07 11:00:08.993816302 and
2026-11-07 17:38:17.407026880. Both are on the date in the header. These bracket
the report's capture run, including its final assertion, rather than individual
optimizer steps.

**Source reading (saved launch settings).** The runner checked out
[`811b94c8bc1ac8164cf3e93cc2a1d0fef7c6e588`][soup-capture], installed its
`[train]` extra and removed `torchao`. It set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only if unset. It called
`GRPOTrainerWrapper(config, device="cuda", report_to="none")` and supplied
400 synthetic prompts of the same integer-addition form, for `n` from 0 through
399, with rewards alternating between 0 and 1 in completion order. It installed
the opt-in capture harness with a 2 GiB total cap and left SDPA at its native
selection. The explicit training settings were:

```yaml
base: Qwen/Qwen2.5-7B-Instruct
task: grpo
data:
  max_length: 512
  val_split: 0.0
training:
  epochs: 1
  lr: 0.0002
  batch_size: 2
  gradient_accumulation_steps: 1
  quantization: 4bit
  gradient_checkpointing: true
  grpo_fp16: true
  grpo_beta: 0.02
  num_generations: 2
  reward_fn: accuracy  # the runner overrides the selector with its synthetic reward
  seed: 1
  data_seed: 1
  logging_steps: 1
  save_steps: 201
  lora:
    r: 32
    alpha: 64
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

## Outcomes

**Measured (Kaggle T4).** The bounds below are from the per-step logs. The counts
and end-of-run parameter checks are from the retained reports.

| Arm | Outcome | Watchdog and adapter check |
|---|---|---|
| 0.5B, 2 generations | 24 optimizer steps, all logged losses and gradient norms finite | One callback, `nan_skip_count=0`, `nan_skip_fraction=0.0`, no non-finite trainable parameters after training, `trainer_state.json` present |
| 7B NF4, 8 generations, 512-token cap | **Capacity row.** CUDA OOM before backward in reference-policy log-probability calculation, at Accelerate's `tensor.float()` output conversion. Requested 2.33 GiB with 166.81 MiB free. | No backward or watchdog verdict from this arm |
| 7B NF4, 2 generations, 512-token cap | 16 optimizer steps, all logged losses and gradient norms finite (logged norm range `0.3241` to `1.059`) | One callback, `nan_skip_count=0`, `nan_skip_fraction=0.0`, no non-finite trainable parameters after training, `trainer_state.json` present |
| 7B NF4, 2 generations, 512-token cap, longer capture | Trainer reached `global_step=400`; Kaggle marked the notebook `ERROR` when a post-training assertion expected 200 | One watchdog trip at step 62, `nan_skip_count=1`, `nan_skip_fraction=0.0025`; no non-finite trainable parameters after training |

**Measured (Kaggle T4) and source reading.** Version 9 cast 392 trainable BF16 LoRA
tensors to FP32 before training by calling the [existing Soup dtype helper] directly.
The [Soup GRPO wrapper] already calls that helper before `train()` for the FP16 path.
This is part of the direct harness setup, not a separate Soup fix. The eight-generation
row applies to the attempted configuration and memory state, not all possible T4 setups.

**Measured (Kaggle T4, retained trainer state).** The longer run recorded:

| Trainer step | Loss | Gradient norm | KL |
|---|---:|---:|---:|
| 61 | 0.13806078 | 20.775705 | 6.903039 |
| 62 | 0.14272296 | `inf` | 7.136148 |

The watchdog named
`base_model.model.model.layers.23.mlp.gate_proj.lora_B.default.weight`.
**Source reading (pinned Soup checkout).** The T4 commit includes [#2060],
which added watchdog parameter attribution.

**Source reading (saved launcher).** The number 200 was an expectation in a
final assertion, not a trainer step cap. The launcher supplied 400 prompts for
one epoch and assumed that two generations would halve the trainer step count.
The assertion ran after training returned.

**Measured (separate synthetic Kaggle T4 control, same Torch build).** The
control used the captured linear shape and operand maxima `max|X|=4.1875` and
`max|dY|=16912`. A dense same-sign FP16 autocast case produced `+inf` in the
weight gradient stored as FP32. The FP32 control returned `7,081,900`.
Dividing the synthetic incoming gradient by 128 kept the FP16 autocast case
finite, with a maximum absolute weight gradient of `55,328`. A cancellation
case also stayed finite with the same operand maxima. The actual operand
alignment is unknown, so this control does not identify the cause of step 62.

**Arithmetic (not a measurement).** If the incident thread's illustrative independent
per-step hazard of `0.005` were applied, the probability of seeing no events would be
`0.995^24 ≈ 0.887` for the 24-step run and `0.995^16 ≈ 0.923` for the 16-step run.
No T4 event rate is estimated here.

## Fixed-batch replay of step 62

**Measured (separate Kaggle replay, two Tesla T4s).** One process reconstructed
`Qwen/Qwen2.5-7B-Instruct` at Hub revision
`a09a35458c702b33eeacc393d103063234e8bc28`, using Soup commit
`811b94c8bc1ac8164cf3e93cc2a1d0fef7c6e588`, Torch `2.11.0+cu130`, CUDA build
13.0, cuDNN runtime `92700`, Transformers `5.20.2`, TRL `0.29.1`, PEFT `0.21.2`,
Accelerate `1.15.0`, bitsandbytes `0.50.2`, and Datasets `4.8.5`. The model was
sharded across both T4s. It used NF4 with FP16 compute, rank-32 LoRA with alpha 64
and dropout 0.05 on all seven projections, DAPO loss with beta 0.02, and gradient
checkpointing.

Each arm restored the 392 pre-step trainable tensors, Python and Torch CPU RNG
states, and both CUDA RNG states. It reused the captured prompt/completion IDs,
masks, advantages, and reference log-probabilities. Gradients were inspected
after scaled backward and after unscale, before clipping. No optimizer update
was performed.

| Loss scale | Unscaled forward loss | Gradient result before clipping |
|---:|---:|---|
| 65536 | 0.1427229642868042 | Only the layer-23 `gate_proj.lora_B.default.weight` named above was non-finite, with five positive infinities, one negative infinity, and no NaNs, both before and after unscale |
| 512 | 0.1427229642868042 | All 392 trainable gradients were finite before and after unscale |
| 1 | 0.1427229642868042 | All 392 trainable gradients were finite before and after unscale |

**Source reading and inference (saved scaler states and launcher).** The
historical scale of 65536 was inferred from later saved scaler states and the
fresh-start launcher; it was not measured in the step-62 capsule. The original
base-model revision and device map were not recorded. Matching the saved forward
loss establishes the reported replay comparison; the first overflowing operation
and the historical H100 cause remain open.

The fingerprints below include the replay report and, for cross-reference, the
separate pinned CPU control report cited with it. The CPU control is separate
from this T4 record; the workflow note and CPU probe are reviewed in [#2642].
Both reports and the capsule stay private.

## Report fingerprints

SHA-256 of the original retained report bytes.
Version 5's small report is reproduced below. Version 9's report is about 31 KiB
and lists 392 tensor names, so the fingerprint and the measured summary above stand
in for it. The capacity report contains the full traceback and is referenced by
fingerprint. The saved launch scripts, per-step and raw Kaggle logs, model
artifacts, and training capsules stay private. The longer capture run's final
report and terminal trainer state are also referenced by fingerprint.

| Report | SHA-256 |
|---|---|
| Version 5, completed 0.5B smoke | `2917047f558f6fd6328bade93dc9d1d2d1a83daf02c0466f0ace855c7de0a532` |
| Version 6, 7B capacity attempt | `388f5b2ca4a2a983704108a5da76f073c52f551ffc28de585e0cb0538d8451f8` |
| Version 9, completed 7B smoke | `9e9698c5b9699373ecf09ceb536e7acc9cd45ab32a756f8e7b58902d83ac76c2` |
| Longer capture, final report | `3e15cfb72de306e5c91bfe11fe77835dc84aebd48cf1db66b549063043f748c8` |
| Longer capture, terminal trainer state | `4feb4af39b934dae5362ae9275d9e217246ab69c0bbf861a5a230682816bdc62` |
| Fixed-batch two-T4 replay report | `c5f97b103687129eecb260b9a511e881879b05f97b1ec74f6653137c8adbbfbf` |
| Separate pinned CPU control report (cross-reference only) | `bdeea86a2726e8772883cc6502aaf3dac957c0f123fd953210ffa210338e35cc` |

<details>
<summary>Version 5 report (measured on Kaggle)</summary>

```json
{
  "caveat": "No finding from this run establishes the original H100 root cause.",
  "completed": true,
  "environment": {
    "cuda": {
      "available": true,
      "capability": [
        7,
        5
      ],
      "index": 0,
      "name": "Tesla T4",
      "total_memory": 15636037632
    },
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "platform": "Linux-6.18.55+-x86_64-with-glibc2.39",
    "purpose": "reduced synthetic T4 watchdog smoke, not an H100 reproduction",
    "python": "3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]",
    "soup_commit": "9ca9819747a624da4cbbff03cc8e1a7d2a81100d",
    "torch": "2.11.0+cu130",
    "torch_cuda": "13.0"
  },
  "finished_at_ns": 1794038338738693651,
  "global_step": 24,
  "nonfinite_trainable_parameter_names_after_train": [],
  "started_at_ns": 1794038115201282726,
  "trainer_state_exists": true,
  "watchdog_callback_attached": true,
  "watchdog_callback_count": 1,
  "watchdog_event": false,
  "watchdog_observation_rows": [
    {
      "nan_skip_count": 0,
      "nan_skip_fraction": 0.0
    }
  ],
  "watchdog_trip_rows": []
}
```

</details>

[#342]: https://github.com/MakazhanAlpamys/Soup/issues/342
[#2642]: https://github.com/MakazhanAlpamys/Soup/pull/2642
[#2060]: https://github.com/MakazhanAlpamys/Soup/pull/2060
[soup-v5]: https://github.com/MakazhanAlpamys/Soup/commit/9ca9819747a624da4cbbff03cc8e1a7d2a81100d
[soup-v7b]: https://github.com/MakazhanAlpamys/Soup/commit/bcd9898b62e62ac6d74c9d985b40b3bf19155b5d
[soup-capture]: https://github.com/MakazhanAlpamys/Soup/commit/811b94c8bc1ac8164cf3e93cc2a1d0fef7c6e588
[existing Soup dtype helper]: https://github.com/MakazhanAlpamys/Soup/blob/bcd9898b62e62ac6d74c9d985b40b3bf19155b5d/src/soup_cli/utils/mixed_precision.py#L85
[Soup GRPO wrapper]: https://github.com/MakazhanAlpamys/Soup/blob/bcd9898b62e62ac6d74c9d985b40b3bf19155b5d/src/soup_cli/trainer/grpo.py
