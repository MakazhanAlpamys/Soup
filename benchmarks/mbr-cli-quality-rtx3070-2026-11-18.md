# Moonlight route-preserving MBR CLI quality screen on RTX 3070

Measured by @Artix6034 on 2026-11-18.

**Question.** Does the pending `training.l2l_route_preserving_mbr` mode change held-out answer-token negative log likelihood (NLL) after a short Moonlight fine-tune through the Soup CLI?

## Setup and commands

One RTX 3070 with 8 GiB VRAM, a Linux container with an approximately 42.5 GiB memory limit, and PyTorch 2.11.0+cu130. The model was Moonlight-16B-A3B with per-expert NF4 weights and BF16 computation. The CLI quality record does not specify the checkpoint revision, physical host RAM, kernel, driver or other package versions. The integration report also describes a separate RTX 3060 Ti Linux smoke and a Windows CPU audit.

The preparation and evaluator scripts live at the evidence commit. The original CLI launch arguments, run configuration and exact evaluator snapshot were kept with the run archives.

```text
benchmarks/harness/p3_prepare_fresh_metamath_eval.py
benchmarks/harness/p3_cli_fresh_quality_eval.py
```

## Method and results

The control used L2L with two-read experts. The candidate also enabled MBR. Both arms used rank-8 attention LoRA and the same short MetaMathQA training examples for two epochs. Each resumed from a full checkpoint with restored AdamW, scheduler and RNG state. A common streamed BF16 evaluator scored fresh held-out question groups from the same dataset. The analysis paired results by question and estimated uncertainty by resampling pairs.

| Check | Evidence retained |
|---|---|
| Paired training and resume | The campaign report describes completed arms and matching input and label hashes in the second epoch. The step and checkpoint receipts are unavailable here. |
| Fresh held-out NLL | The report describes a paired screen. Its per-question scores and paired interval are unavailable here. |
| Integration | The report describes targeted checks and a Linux one-step smoke. Its logs and configurations are unavailable here. |

**Verdict — no verdict.** The experiment documents a paired CLI quality screen. The retained evidence cannot substantiate its quality estimates.

## Limitations and superseded runs

The screen used one seed, short training and length-filtered examples from the same corpus, with no predeclared non-inferiority margin. The refreshed split narrows exact question overlap while leaving possible semantic overlap. The resumed run extends the earlier checkpoint screen. The fresh split supplements the earlier held-out set, and the earlier manual generation screen used different adapters. Later window and hook fixes have no matching quality rerun. The post-fix Linux smoke superseded the earlier integration smoke for that fix, and a clean post-fix full Windows suite remained unverified.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/p3-moonlight-40step-fresh-quality-20261118.md
