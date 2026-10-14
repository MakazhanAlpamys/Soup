# E3: LoRA versus LoRA-FA under RAM layer streaming

**Exploratory evidence; formal acceptance pending.** On the tested Qwen2.5-0.5B
recipe, LoRA-FA reduced optimizer storage but did not improve speed or materially
reduce peak allocated memory. Keep ordinary LoRA as the default for this recipe.
This record does not establish a general LoRA-FA recommendation across models.

## Measured results

RTX 3090 24 GiB, batch 1, rank 16, alpha 32, `q_proj`/`v_proj`, seed 42.
Each arm trains on 32,768 input tokens. Five fresh-process ABBA blocks per length;
60 accepted arms. Step time is the mean of ten per-arm medians after excluding
two warmup steps; it excludes data fetch, evaluation and saving.

| Length | LoRA step, s | FA step, s | FA time increase | Peak allocated GiB, LoRA / FA |
|---|---:|---:|---:|---:|
| 512 | 0.226576 | 0.252051 | 11.24% | 1.278357 / 1.273230 |
| 2048 | 0.283414 | 0.307178 | 8.39% | 3.956509 / 3.951382 |
| 4096 | 0.456134 | 0.484007 | 6.11% | 7.520589 / 7.515462 |

FA saves **5.25 MiB of peak PyTorch allocated memory** at each length. Optimizer
state is 8.250366 MiB for LoRA and 3 MiB for FA. Allocated memory is not total
device VRAM; reserved memory and whole-device samples are retained in the record.
The latter include setup/evaluation and may miss transient peaks.

| Training length | Held-out loss at fixed window 512, LoRA / FA |
|---|---:|
| 512 | 2.400699 / 2.467944 |
| 2048 | 2.543355 / 2.582384 |
| 4096 | 2.623601 / 2.639126 |

Quality is a short-training observation on WikiText, not a convergence or
multi-seed result. Compare methods within length: lengths use 64/16/8 optimizer
steps respectively. Larger length is not an otherwise identical quality test.

## Prior report and repeat rule

The [existing operating-point report](gate-725-lorafa-operating-point.md) concerns
a tiny LLaMA microbenchmark and an analytic adapter activation ratio. Its 50%
Adam-state reduction applies to square projections, not total VRAM. Actual state
bytes for these Qwen targets decrease by about 63.64%; whole-memory savings do not.

The pinned F1 module's `blocks_to_add` returns zero for every length after five
accepted blocks. Pooled sigma in seconds is 0.000200608, 0.003419831 and
0.016776379. The formula's N is 2 in each case, below the five-block pilot minimum.
This planning rule targets a 10% effect; it is not a significance test for the
smaller observed effects.

## Validity and deviations requiring review

All 60 accepted arms have `suspend`, `power` and `foreign_reads` = `ok` using an
empirically validated overlay resolver. Controls gave exact 64/192/64 MiB disk
deltas; idle and own-read controls passed, and 1.25 GiB foreign reads were rejected.
Only the device resolver is injected through F1's interface; thresholds and other
sensors are unchanged. Boot/mount evidence is rechecked. This observes the guest's
virtual disk, not all physical-host activity. Four processes were uninspectable.

The original unmapped run remains `unknown`. The whole `512/block04` was excluded
after 4,688,121,856 bytes of foreign reads; `block04-replacement01` replaces it.
Its first good arm was not reused. Both rejected records remain in raw evidence.

**The rule was not Git-committed before the first arms.** The separate protocol
was recorded before blocks 02–05, after block 01 had been observed. This fails the
formal precommit requirement of the research house rules/F1 METHOD. Do not call
the series preregistered. Acceptance of this deviation, or a prospectively
committed confirmation series, is a reviewer decision.

## Evidence and reproduction

All artifacts are in [results/e3-vast-20261015](results/e3-vast-20261015/):

- [Full report](results/e3-vast-20261015/report.txt): hardware, versions, revisions,
  quality/memory tables, exclusions and command history.
- [Raw results](results/e3-vast-20261015/raw-results.zip): original JSON, logs,
  commands, package versions, source manifest, declarations and rejected arms.
- [Analysis](results/e3-vast-20261015/analysis.json) and
  [protocol](results/e3-vast-20261015/protocol.json).
- [Reproduction package](results/e3-vast-20261015/reproduction.zip): original
  `vast_runner.zip` plus the exact overlay/series scripts and archive analyzer.
- [SHA256 hashes](results/e3-vast-20261015/SHA256SUMS.json).

The reproduction archive deliberately preserves historical research sources,
including pinned Soup 8387417 with a local save-guard fix and F1 614229d. It is
not a production integration and does not demonstrate current `main` behavior.
No model weights, credentials or cloud lifecycle commands are included.

For CPU-only analysis, extract `reproduction.zip`, then extract its inner
`vast_runner.zip` into the same parent directory. The supplemental scripts remain
alongside the original `run.py` and `arm_validity.py`. Run:

```bash
python vast_runner/analyze_e3_archive.py raw-results.zip --output analysis.json
```

For GPU reproduction, follow `report.txt` and the archived runner README.
Recalibrate the overlay mapping on the new instance; never reuse another boot's
certificate. The README's old offer recommendations/prices are historical, not
current rental advice. Setup is the tested Linux/CUDA path; local CPU analysis
does not execute GPU training. Declare/commit any new final-study protocol before
running it. No additional GPU runs are required by this series' repeat calculation.
