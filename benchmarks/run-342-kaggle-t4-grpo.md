# Kaggle T4 GRPO smoke and capacity record (#342)

Two short FP16 watchdog smoke runs and one capacity attempt on a single training device
on Kaggle. The inputs were synthetic, and the runs used a direct `TRL.GRPOTrainer`
with Soup's watchdog attached explicitly. The record is scoped to these three arms.
These results neither reproduce nor rule out the original BF16 failure in [#342].

## Recorded environment and settings

**Measured (Kaggle report and install log).** All three arms used `cuda:0`, a Tesla T4
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

**Source reading (saved launch settings).** The common explicit `GRPOConfig` settings
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
its `[train]` extra, and imported `soup_cli` from the checkout. The 7B scripts used
`device_map={"": 0}`.

## Outcomes

**Measured (Kaggle T4).** The bounds below are from the per-step logs. The counts
and end-of-run parameter checks are from the retained reports.

| Arm | Outcome | Watchdog and adapter check |
|---|---|---|
| 0.5B, 2 generations | 24 optimizer steps, all logged losses and gradient norms finite | One callback, `nan_skip_count=0`, `nan_skip_fraction=0.0`, no non-finite trainable parameters after training, `trainer_state.json` present |
| 7B NF4, 8 generations, 512-token cap | **Capacity row.** CUDA OOM before backward in reference-policy log-probability calculation, at Accelerate's `tensor.float()` output conversion. Requested 2.33 GiB with 166.81 MiB free. | No backward or watchdog verdict from this arm |
| 7B NF4, 2 generations, 512-token cap | 16 optimizer steps, all logged losses and gradient norms finite (logged norm range `0.3241` to `1.059`) | One callback, `nan_skip_count=0`, `nan_skip_fraction=0.0`, no non-finite trainable parameters after training, `trainer_state.json` present |

**Measured (Kaggle T4) and source reading.** Version 9 cast 392 trainable BF16 LoRA
tensors to FP32 before training by calling the [existing Soup dtype helper] directly.
The [Soup GRPO wrapper] already calls that helper before `train()` for the FP16 path.
This is part of the direct harness setup, not a separate Soup fix. The eight-generation
row applies to the attempted configuration and memory state, not all possible T4 setups.

**Arithmetic (not a measurement).** If the incident thread's illustrative independent
per-step hazard of `0.005` were applied, the probability of seeing no events would be
`0.995^24 ≈ 0.887` for the 24-step run and `0.995^16 ≈ 0.923` for the 16-step run.
No T4 event rate is estimated here.

## Report fingerprints

SHA-256 of the original retained `issue342-smoke-report.json` bytes for each version.
Version 5's small report is reproduced below. Version 9's report is about 31 KiB
and lists 392 tensor names, so the fingerprint and the measured summary above stand
in for it. The capacity report contains the full traceback and is referenced by
fingerprint. Raw Kaggle logs, model artifacts, and training capsules stay private.

| Report | SHA-256 |
|---|---|
| Version 5, completed 0.5B smoke | `2917047f558f6fd6328bade93dc9d1d2d1a83daf02c0466f0ace855c7de0a532` |
| Version 6, 7B capacity attempt | `388f5b2ca4a2a983704108a5da76f073c52f551ffc28de585e0cb0538d8451f8` |
| Version 9, completed 7B smoke | `9e9698c5b9699373ecf09ceb536e7acc9cd45ab32a756f8e7b58902d83ac76c2` |

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
[soup-v5]: https://github.com/MakazhanAlpamys/Soup/commit/9ca9819747a624da4cbbff03cc8e1a7d2a81100d
[soup-v7b]: https://github.com/MakazhanAlpamys/Soup/commit/bcd9898b62e62ac6d74c9d985b40b3bf19155b5d
[existing Soup dtype helper]: https://github.com/MakazhanAlpamys/Soup/blob/bcd9898b62e62ac6d74c9d985b40b3bf19155b5d/src/soup_cli/utils/mixed_precision.py#L85
[Soup GRPO wrapper]: https://github.com/MakazhanAlpamys/Soup/blob/bcd9898b62e62ac6d74c9d985b40b3bf19155b5d/src/soup_cli/trainer/grpo.py
