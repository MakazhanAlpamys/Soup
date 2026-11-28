# Moonlight MBR phase profile and expert I/O on RTX 3060

Measured by @M1kado0 on 2026-11-19.

**Question.** Where does a Moonlight MBR step spend its time, and how much can packed transfer, expert prefetch and a warm host cache save?

## Setup and commands

One RTX 3060 with 12,288 MiB VRAM, an AMD EPYC 7402 and 270,106,738,688 bytes of host RAM. The host receipt records Linux 5.15.0-190-generic, driver 610.57.04, CUDA runtime 13.0.96, cuBLAS 13.1.1.3, bitsandbytes 0.50.2 and Accelerate 1.15.0. It does not list a Python or PyTorch version. Moonlight-16B-A3B used BF16 compute and a per-expert NF4 cache. The window was 8 × 64 tokens, with a group of eight for the pending `training.l2l_route_preserving_mbr` mode.

The harnesses live at the evidence commit.

```bash
python benchmarks/harness/p72_prepare_moonlight_cache.py --root /workspace/p3-baseline
python benchmarks/harness/p72_f1_blocks.py --root /workspace/p3-baseline --output results/f1
PYTHONPATH=$PWD/src:$PWD python benchmarks/harness/p72_mbr_phase_profile.py --storage nf4
```

## Method

For the phase profile, the synchronous arm used direct expert reads and instrumented the L2L step without the optimizer. The profiler audited coverage and checked the gradient digest against an uninstrumented run. The timing gate compared synchronous transfer with packed transfer plus one-expert prefetch and a 12 GiB warm host cache. In each of five control–candidate–candidate–control blocks, a fresh process ran every arm, with three timed steps after warm-up. The prespecified rule required identical gradient digests, a significant two-sided paired t-test at 5% and at least a 10% reduction.

## Results

| Profiled phase | Seconds | Share of 18.68 s step |
|---|---:|---:|
| Direct expert reads | 8.57 | 45.9% |
| Host-to-GPU copy and associated work | 2.39 | 12.8% |
| Backward autograd excluding expert compute | 1.91 | 10.2% |
| Attention, router and shared experts | 2.32 | 12.4% |
| Expert GEMM, forward and backward | 1.44 | 7.7% |
| NF4 dequantization | 0.64 | 3.4% |
| Layer weight loading | 0.46 | 2.5% |
| Expert dispatch | 0.68 | 3.7% |
| Other | 0.28 | 1.5% |

| Timing comparison | Change in step time | Paired t |
|---|---:|---:|
| Packed transfer, prefetch and warm host cache versus synchronous transfer | −63.1% | −124 |

All 20 gated processes passed the required suspend and power checks and produced the same gradient digest. The instrumented and baseline gradient digests also matched.

**Verdict — positive.** Direct expert reads dominated the synchronous profile, and the combined I/O configuration passed the speed and gradient gates on this host.

## Limitations and superseded runs

The warmed cache held the full Moonlight expert set. The result is bounded to that working set, host, model and window. The container could not attribute foreign reads to a block device. The individual transfer and cache series lacked the full validity and gradient checks of the combined gate. The corrected profiler superseded an earlier profile that dropped autograd worker-thread frames.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/p72-diagnostics
