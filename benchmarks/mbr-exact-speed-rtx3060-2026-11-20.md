# Exact Moonlight MBR speed and short-run training on RTX 3060

Measured by @M1kado0 on 2026-11-19 and 2026-11-20.

**Question.** Can grouped routed expert execution match native per-microbatch L2L bits while retaining the step-time benefit of grouped MBR?

## Setup and commands

Two Linux hosts with an RTX 3060 each, one AMD Ryzen Threadripper PRO 3945WX and one Intel Xeon E5-2697A v4. The evidence records driver 615.71, PyTorch 2.14.1+cu130, CUDA 13.0 for PyTorch and a 12 GiB expert-cache allowance. The experimental kernel work used Triton 3.8.0 and nvmath-python 1.0.1 on the Threadripper host. Physical RAM, exact kernel releases and the full stack for both final runs are unreported. Moonlight-16B-A3B used BF16 compute and per-expert NF4 weights.

The launchers, data preparation and reporting harnesses live at the evidence commit. Run the launchers in a fresh benchmark directory because they reset their output folders.

```bash
bash benchmarks/results/a0-exact-levers-20261119/abc-run.sh
python benchmarks/harness/p72_prepare_train_data.py --model <source> --output qual/train_data.pt
bash benchmarks/results/a0-exact-levers-20261119/train-run.sh "C A B"
```

## Method

The arms were native per-microbatch L2L, native grouped MBR and an exact MBR prototype. All arms used packed transfer and the warm host cache. Both grouped arms also used expert prefetch. The prototype added a cuBLASLt expert GEMM, ordered backward, fast row gather and asynchronous expert leases. The Soup fields `training.l2l_route_preserving_mbr`, `training.l2l_mbr_group_size` and `training.l2l_mbr_ordered_backward` are pending. The cuBLASLt path is a harness prototype.

The fixed 8 × 64-token timing window used groups of eight and five balanced six-run blocks on each host, with every arm appearing once in each half of a block in reverse order. Each fresh process contributed the median of three timed steps after one warm-up. The separate deterministic run used two epochs over 160 short MetaMathQA examples, 40 AdamW steps and 128 held-out question groups. It recorded losses and adapter hashes after each update.

## Results

| Fixed-window timing | Threadripper | Xeon |
|---|---:|---:|
| Native L2L (s) | 21.01 | 37.22 |
| Native grouped MBR (s) | 5.53 | 10.23 |
| Exact MBR (s) | 5.22 | 9.30 |
| Exact MBR versus native L2L | −75.2% | −75.0% |
| Exact MBR versus native grouped MBR | −5.6% | −9.1% |

There were five valid blocks and zero void blocks per host. Exact MBR and native L2L had identical gradient digests in the timing runs. In deterministic training their losses and adapter hashes matched at all 40 steps on both hosts. Their held-out scores also matched on every recorded group.

| Deterministic training | Native L2L | Native grouped MBR | Exact MBR |
|---|---:|---:|---:|
| Threadripper elapsed (s) | 1076.7 | 274.8 | 281.0 |
| Xeon elapsed (s) | 1802.2 | 498.0 | 482.1 |
| Logical expert fetches | 848,600 | 125,496 | 125,434 |
| Mean NLL per held-out group | 0.2506914 | 0.2507335 | 0.2506914 |
| Token-weighted NLL | 0.2596613 | 0.2598785 | 0.2596613 |

**Verdict — positive.** The prototype preserved the recorded native L2L hashes and held-out scores, and matched or improved on grouped MBR step time in the fixed-window timing runs.

## Limitations and superseded runs

The timing protocol used ordinary execution. The padded training inputs needed deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` to obtain repeatable exactness. The deterministic training times are presented separately. The custom reduction follows the PyTorch 2.14 reduction order. Generalization across models, GPU types, library versions, seeds and longer fine-tunes remains untested. The balanced three-arm timing runs superseded the started two-arm timing series for the final prototype. Earlier variants without asynchronous leases served as development comparisons.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/a0-exact-levers-20261119
