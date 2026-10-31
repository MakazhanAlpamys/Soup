# Protocol provenance for #2162

This is a guide to the original local freezes, not a new experimental protocol.
No further model runs are authorized or launched by these files.

The public claim proposed a fixed-K, two-cell experiment. The later experiments
changed the design after earlier results were known. Their individual rules were
frozen locally before their own execution; they were not posted publicly before
running. Original timestamps and resource logs establish the recorded order, not
independent external timestamp attestation.

| Stage | Authoritative rule and roster | Change from preceding stage |
|---|---|---|
| Native census / initial exactness | [exactness freeze](results/issue2162-rtx3090-v1/exactness-v1/freeze.json) | Existing Mistral-7B NF4 model; native K measured separately at 2048 and 4096. Ten native/repeat/QKVO/profile processes, two updates each. |
| Fixed native K | [freeze](results/issue2162-rtx3090-v1/timing-v1/freeze.json), [rule transcription](results/issue2162-rtx3090-v1/timing-v1/RULE.md) | Five alternating AB/BA pairs per cell, primary 2048; K11/QKVO9 and K3/QKVO8. Two-cell correction. |
| One whole layer reallocated | [freeze](results/issue2162-rtx3090-v1/reallocation-v1/freeze.json), [original protocol](results/issue2162-rtx3090-v1/reallocation-v1/RULE.md) | Seq2048 only, native K11 / reduced K10 / combined K10+QKVO19. Six balanced three-arm blocks; three contrasts. |
| Projection-scoped dispatch | [freeze](results/issue2162-rtx3090-v1/scoped-v2-resume1/freeze.json), [original protocol](results/issue2162-rtx3090-v1/scoped-v2-resume1/RULE.md) | Same K10/cache layout, compare narrowed dispatch to native and previous implementation. Adds a positive-increment requirement and conditional six-pair confirmation. |

Every process times 50 updates after 10 warmups. Its first two warmups reproduce
the retained native reference bit-for-bit. The two later stages additionally have
normal/repeat/profile gates for every arm before timing.

The timing statistic is the median across whole pairs/blocks of
`100 * (median(comparator steps) - median(candidate steps)) / median(comparator steps)`.
The initial experiment uses 10,000 NumPy `default_rng(42)` resamples of five whole
pairs, with linear quantiles at .025/.975 and .0125/.9875 (two cells). Later stage A
experiments resample six whole blocks, using .025/.975 and .05/6 / (1-.05/6)
(three contrasts). All originally specified primary rules failed. No control or
secondary result is promoted into their place.

The last experiment also required positive ordinary and corrected lower bounds
for scoped versus legacy dispatch. Its stage B would have used six new alternating
native/scoped pairs, 10,000 whole-pair resamples with seed 73, median >=2% and a
positive 95% lower bound. Stage A failed, so stage B was not run.

All original rules enforce exactness, exact cache accounting and allocated and
reserved peaks within 8 GiB. The publication additionally evaluates the #2161
memory comparison requested by the maintainer: allocated increase <= saved bytes
+ 64 MiB. That additional calculation is identified as retrospective and changes
none of the frozen verdicts.

## Evidence representation

The [manifest](results/issue2162-rtx3090-v1/manifest.json) hashes every published
input and preserves the original file hashes. Each experiment contains:

- `freeze.json`, the executed `ledger.json`, and the original independent audit.
- `timings.jsonl`, one original receipt per process with all 50 raw durations and
  losses, exact byte peaks and measured-window timestamps.
- `resources.jsonl`, all original host/GPU samples tagged by process. Column order
  in each GPU row is index, UUID, memory used MiB, utilization %, temperature C,
  power limit W, SM clock MHz. Initial untimed gate telemetry used its earlier
  format and is preserved without relabelling it as timing telemetry.
- `processes.jsonl`, complete exactness receipt metadata, with repeated tensor
  updates referencing `reference-updates.json`. Replacing that reference and
  restoring the recorded JSON/CRLF encoding reproduces each original file SHA256.
  The public audit checks all 84 receipts, not a subset of tensors.
- `source-hashes.json` and `source-blobs.json`, binding every frozen Python source
  to its original Git commit/blob and byte encoding. Original source archive
  hashes remain in the freezes; the redundant ZIPs are omitted.
- `logs.jsonl`, including the native auto message, process outputs and CPU tests.

The [failed preparation](results/issue2162-rtx3090-v1/scoped-v1-rejected-preflight/state.json)
preceded all model execution. The subsequent
[admission recovery](results/issue2162-rtx3090-v1/scoped-v2-resume1/recovery-amendment.json)
preserves the refusal at 8% residual utilization and six completed gates reused
without model repetition. The original admission archive remains preserved
locally; its file hashes, previous ledger and failed process log are public.

The original audits predate publication. The added
`benchmarks/harness/issue2162_audit.py` independently recomputes the numerical
results from the compact record and checks raw receipt reconstruction, exactness,
ordering, freezes, cap binding, memory and telemetry. Git verification is optional
for shallow CI checkouts and explicit in its output; the recorded closure ran it
over all 2,927 bindings.

The harness files are research code. To reproduce model execution, check out the
relevant frozen commit listed in `source-blobs.json`, provide the existing model
and shard material named in `model-material.json`, and use that commit's census,
exactness and timing helpers with the recorded roster. Do not use later production
sources and claim a replay of these measurements. The read-only audit needs no
GPU or model files.

## Limits and further work

The 24 GiB RTX 3090 was capped at 8 GiB; it is not a physical 8 GiB-card result.
Inputs are synthetic. The three experiments were adaptive and their original
corrections are per experiment, not one family across every explored variant.
None supports adding a production option.

Any future investigation should start from a justified cell with a clearly larger
predicted benefit, publish its hypothesis, cell, arms, rule and pair count before
execution, and include the previous 1.421695%, 1.860896%, 1.997942% and 2.034504%
results in its correction family, as requested in the owner response. Repeating
this unchanged cell until its number crosses 2% is not the proposed next step.
