# Route-preserving MBR group-size sweep on RTX 3060 Ti

Measured by @Artix6034 on 2026-11-17.

**Question.** How does the number of microbatches in a routed expert group affect step time, logical expert reads and numerical agreement with layer-to-layer training (L2L)?

## Setup and commands

One RTX 3060 Ti (8 GiB class, 8,276,869,120 bytes visible) in a Linux container, Python 3.12.15, PyTorch 2.11.0+cu130 and CUDA 13.0. The preflight records a cgroup limit of 91,658,125,312 bytes, zero cgroup swap and an overlay mount. Physical RAM, the kernel version, the NVIDIA driver and the remaining package versions were unreported. The model was `moonshotai/Moonlight-16B-A3B` at `2a9a177f4841649f849ba941e41d2ee73c042314`, with BF16 compute and an NF4 per-expert cache.

The sweep driver and its bridge live at the evidence commit. The recorded training source anchor was `5f02e9c57d4eae26fb697a7e5f63686d2d1b2c96`. The per-arm JSON binds the historical bridge and step implementation by hash. Later versions of those files may differ.

```bash
bash benchmarks/harness/run_p3_mbr_group_size_sweep.sh
# Driver invokes benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
```

## Method

Each group size used the same fixed 8 × 64-token window and identical initial LoRA weights. Both arms used two-read experts and grouped routes, with one AdamW update per arm. After an untimed diagnostic pass, the driver ran control, candidate, candidate, control and took the median of the two samples per arm. Attention and route selection ran per microbatch. The pending Soup fields are `training.l2l_route_preserving_mbr` and `training.l2l_mbr_group_size`.

## Results

| Group size | Dispatches per routed layer | L2L median (s) | MBR median (s) | Speedup | Logical fetches, L2L → MBR | Route-slot mismatch | Gradient relative L2 | MBR minus L2L loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 98.60 | 91.09 | 1.08× | 19,814 → 19,814 | 0.000% | 0.000% | 0.000000 |
| 2 | 4 | 100.80 | 52.92 | 1.90× | 19,814 → 11,146 | 2.666% | 6.095% | +0.002878 |
| 4 | 2 | 100.22 | 31.40 | 3.19× | 19,814 → 6,014 | 3.012% | 10.465% | +0.005937 |
| 8 | 1 | 98.27 | 18.37 | 5.35× | 19,814 → 3,144 | 2.985% | 9.948% | +0.004878 |

The per-layer audit found that forward and input-gradient backward fetched the expected unique experts per group and recompute fetched zero. Route IDs and gate weights matched each arm's original forward on recompute. The size-one arm matched the L2L loss, routes, raw gradients and adapter update exactly.

**Verdict — mixed.** Larger groups reduced logical reads and step time on this window and changed the loss, routes and raw gradients.

## Limitations and superseded runs

This is one window and two timed observations per arm. The size-one timing difference has no independent speed verdict. Model quality and physical SSD traffic were outside the measurement. Temporary memory ballast consumed RAM inside the existing cgroup, so the run failed the hard 32 GiB capacity criterion. The four diagnostic passes preceded timing and are excluded from the medians. The reported sweep supersedes no earlier group-size sweep.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/results/mbr_group_size_sweep_20261117
