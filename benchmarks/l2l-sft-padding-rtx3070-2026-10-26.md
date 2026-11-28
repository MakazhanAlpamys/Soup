# Gate record - padding sensitivity of L2L SFT forward and backward

**Verdict (2026-10-26) - Mixed.** Padded BF16 windows exceeded the gradient-accumulation tolerance on the RTX 3070. Equal-length controls matched bit for bit on both tested cards.

Measured by @M1kado0.

## 1. Setup

- RTX 3070 8 GB, driver 595.71.05 on the initial NVMe host and 580.142 on the repeat host. The repeat's cgroup limit was 52.9 GB. RTX 3090 24 GB, driver 580.126.09, with 134.9 GB visible to the original container and no verified cgroup ceiling in that arm. The equal-length controls had cgroup limits of 52.9 GB and 64.8 GB on the two respective cards.
- Ubuntu 24.04 CUDA 13.0 development container, torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0.
- `mistralai/Mistral-7B-v0.1` with NF4 weights and BF16 compute. The stand used real weights and seven synthetic token rows. `training.l2l` is pending.

## 2. Method and results

Each arm ran forward and backward without an optimizer step. Two identical gradient-accumulation (GA) runs checked repeatability. The tolerance was 1.5 times the error measured between GA with MATH and EFFICIENT_ATTENTION SDPA, applied per window. The stand tested windows of 1, 2 and 4 microbatches, including incomplete final windows. The equal-length control used 230-token rows.

| Run | Window 1 | Window 2 | Window 4 |
|---|---|---|---|
| RTX 3070 mixed lengths, seed 5 | Bit-equal | Gradient error 0.0397 versus 0.0347 limit at step 2. Loss error 3.91e-4 versus 2.78e-4 at step 0 | Loss error 3.88e-4 versus 2.64e-4 at step 0 |
| RTX 3070 mixed lengths, seed 6 | Pass | Gradient error 0.0415 versus 0.0409 at step 2 | Pass |
| RTX 3070 mixed lengths, seed 7 | Pass | Gradient error 0.0426 versus 0.0251 at step 2, with loss failures | Gradient error 0.0330 versus 0.0235 at step 1, with a loss failure |
| RTX 3090 mixed lengths, seed 5 | Bit-equal | Within tolerance, worst loss error 5.5e-4 versus 6.7e-4 | Within tolerance |
| Equal-length controls on both cards | Bit-equal | Bit-equal | Bit-equal |

Full-window layer loads were 62 for L2L at all three window sizes, versus 62, 122 and 244 for GA. The first counter measured an NF4 restore delta of 0 after previous arms had populated cached views. The corrected whole-run count was 224 in all three RTX 3070 repeats and in the equal-length controls. The initial RTX 3090 run retains its original counter limitation.

## 3. Commands

The harness lives at the evidence commit. Each archived run report prints its complete command. The two commands below reproduce the original successful launches. The repeat and equal-length control used the same driver with the recorded `--data-seed` values and `--equal-length 230` where applicable.

```bash
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --out /root/runs/fwdbwd-8gb-bf16-run2
/root/soup/.venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 24gb --out /root/runs/fwdbwd-24gb-bf16
```

## 4. Limits and earlier runs

A device mismatch invalidated the first launch on the 8 GB host. The successful launch used a harness that placed the GA rows on CUDA. The corrected restore counter supersedes interpretation of the initial zero delta. The padding control localizes sensitivity to the tested execution on equal versus mixed lengths. It leaves the responsible CUDA operation unresolved. The study recorded BF16 results only and did not measure training throughput or model quality.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results
