# GRPO zero-spread reward monitor: functional validation

Source: Soup main `2db8e0c5` plus the candidate monitor (`8f875a9a`).
Python 3.12, PyTorch 2.14.1+cu130, Transformers 5.19.0, TRL 0.29.1.
Physical H200 GPU 0 was selected by UUID; other processes remained running.

A seeded one-layer random Llama (hidden 32, vocabulary 8), local WordLevel
tokenizer and four synthetic prompts ran six actual GRPO optimizer steps per
case, with two completions per group. No checkpoint or dataset was downloaded.
These are functional controls, not a model-quality or throughput benchmark.

| Reward function | TRL zero-spread fraction | Log measurements | Warnings |
|---|---:|---:|---:|
| Constant zero | 1.0 | 6 | 1, after the fifth distinct measurement |
| Alternating 0 and 1 in each group | 0.0 | 6 | 0 |

The live display retained the corresponding 100% / 0% reading. Training reached
all six steps in both cases. Combined PyTorch peak allocated memory was
67,302,912 bytes (about 64.2 MiB); this is not whole-device/context memory.
[Every step's original TRL log](evidence/reward-collapse-2674/result.json) is
retained, including loss, gradient norm, rewards and the fraction read by Soup.

The initial callback/display regressions produced 8 failures and 11 passing
controls on unchanged main. The 20 feature cases and existing callback/display
tests then passed (36 total). They cover threshold boundaries, invalid readings,
repeated log events, evaluation-only logs, healthy streak reset, world-rank-zero
reporting, new-run reset, unchanged log inputs and no training-stop request.

Reproduce with the checkout's training dependencies, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=1 \
  python benchmarks/harness/grpo_reward_monitor_smoke.py
```

The fixed >50% / five-measurement rule is a proposed warning heuristic, not a
validated quality cutoff. It uses TRL's existing metric and changes no reward,
gradient, trainer configuration, or stopping policy. Scope confirmation and the
area holder's first refusal remain with the maintainer; no ready PR is implied.
