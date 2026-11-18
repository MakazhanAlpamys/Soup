# Later #2162 measurements on a 5090 Laptop and a second 3090

Refs #2162. This publishes the two campaigns that were previously reported in
the issue but were absent from the merged #2644 record. They are **separate
measurements**, not a confirmation of the earlier hand-chosen B layout or an
implementation proposal. The accepted #2162 scope remains opt-in QKVO outputs
only; the gains below involve gate/up outputs, which are outside that scope.
The 5090 run did not follow the original published thermal protocol because of
cooling pauses, profile changes and an interruption. The second-3090 protocol
was posted after the run; the requested three-hour review period was not observed.

## Verdicts under each locally frozen rule

Positive percentages mean less time per synchronized update than the named
comparator. A process median covers 50 measured updates after 10 warmups; each
six-block/pair result is the median of paired percentage reductions. The
ordinary and family-adjusted intervals come from 10,000 whole-block/pair
bootstrap resamples. The family size differs by campaign and includes earlier
contrasts. The small samples do not establish a portable population gain.

| Card and contrast | Median | Ordinary 95% | Family-adjusted interval | Rule outcome |
|---|---:|---|---|---|
| RTX 5090 Laptop, mixed/native | 8.166451% | [-5.595217%, 27.472056%] | family 16 [-15.799741%, 37.033011%] | **Stage A failed**: both lower bounds are negative; no confirmation |
| RTX 5090 Laptop, mixed/QKVO | 10.886924% | [-12.338090%, 28.846839%] | family 16 [-13.108285%, 37.298958%] | Incremental lower bounds negative |
| RTX 5090 Laptop, QKVO/native | -1.401845% | [-12.058342%, 13.431082%] | family 16 [-16.562319%, 18.545262%] | Control only |
| Second RTX 3090, mixed/native | 7.407784% | [6.999374%, 7.533636%] | family 24 [6.703648%, 7.582978%] | **Stage A passed** its local numerical rule |
| Second RTX 3090, mixed/QKVO | 10.717112% | [10.136038%, 10.847786%] | family 24 [10.134580%, 10.882873%] | Incremental lower bounds positive |
| Second RTX 3090, QKVO/native | -3.781183% | [-3.847102%, -3.287311%] | family 24 [-3.855151%, -2.842695%] | QKVO control slower than native |
| Second RTX 3090, fresh confirmation mixed/native | 6.968553% | [6.520774%, 7.398327%] | family 24 [6.512996%, 7.419135%] | **Confirmation passed** its local numerical rule |

Both cells used `uniform-base-flops-v1` with no timing retuning of the rule:
native auto K=11; the rule chose K=0 with QKVO and gate/up outputs in layers
0..30. The FLOP score does not price attention, dequantization, dispatch or
transfers. It does not predict latency on another card. Mistral-7B NF4/bf16,
batch 1, sequence 2048, SDPA default, and the exactness/memory checks are
recorded in each [freeze](results/issue2162-later-5090-v1/freeze.json) and
[protocol](results/issue2162-later-5090-v1/protocol.md); the second card has its
own [freeze](results/issue2162-later-3090-v1/freeze.json) and
[protocol](results/issue2162-later-3090-v1/protocol.md).

The 5090 completed six gates and 18 timings: 46 exact replays against two
reference updates, 180 warmups and 900 measured updates. Its allocated/reserved
mixed peak was 7,184,283,136 / 7,434,403,840 bytes. The second 3090 completed
six gates, 18 Stage A timings and 12 fresh AB/BA confirmation timings: 70 exact
replays, 300 warmups and 1,500 measured updates. Its same mixed peak was
7,184,283,136 / 7,434,403,840 bytes. Both were 24 GiB cards capped at 8 GiB,
not physical 8 GiB cards. The two 3090 studies are on separate physical cards
but share an architecture and host; their absolute times are not compared.

## Operating history and limits

The 5090 reached 90 °C in sampled processes and recorded software thermal
slowdown. Its six block reductions span -15.799741% to 37.033011%, so the
point estimate is not evidence of a stable gain. After a failed thermal gate,
the operator changed the laptop to Standard/MSHybrid and authorized native
thermal management. During the Performance run, there were cooling pauses,
a temporary Turbo selection during admission with no worker, and the removal
of a software 75 °C admission ceiling. A duplicated child-process ceiling was
then corrected; the refused attempt stopped before importing torch. Later a
Playwright Chrome process interrupted one timing with an unknown partial-update
count. Completed timings were retained and the incomplete process restarted;
no completed model was repeated. The [5090 recovery audit](results/issue2162-later-5090-v1/desktop-resume-recovery-audit.json)
and [full report](results/issue2162-later-5090-v1/results.md) preserve the
attempts, temperatures, flags, counter scope, and mixed admission regimes.
The amendments and pauses mean this is not a faithful, uninterrupted execution
of the earlier published thermal conditions.
Three later native-only cooling/stability screens on the 5090 (qualification
versions 2, 3 and 4) each completed but did not qualify. Their rows are not
published in this record or its raw archive; they remain in the author's local
`5090-cooling-qualification-v{2,3,4}-evidence.zip` archives.

The second 3090 was a fresh same-rule campaign with six gates, 30 timings and
family 24. Its sampled peak was 53 °C, with no sampled thermal-slowdown flag.
Existing inference services stayed resident, and the guard refused Windows/WSL
CI workers. WDDM and aggregated WSL process visibility do not prove physical
GPU exclusivity. The [second-card report](results/issue2162-later-3090-v1/results.md)
records every process median and the operational constraints. This is one
synthetic training cell. The protocol was frozen locally before execution but
posted only after the run, and the owner's requested three-hour public review
window was not observed. It is another experiment on the record, not a
pre-registered confirmation of an earlier round.
The 27 October attempt on this second 3090 and its continuations were interrupted
by CI activity. Their rows are not published here or in this archive; the
original and two continuation failure ZIPs remain preserved locally as
`3090-rule-v1-failure.zip`, `3090-rule-v1-resume1-failure.zip` and
`3090-rule-v1-resume2-failure.zip`.

Neither campaign changes the QKVO-only verdict in
[the earlier record](gate-2162-auto-qkvo.md), establishes a result for a
physical 8 GiB card, or accepts gate/up retention for production. There is no
change here to `src/`, configuration, defaults, schema or dependencies.

## Evidence and CPU reproduction

Git holds LF-normalized copies of the freezes, protocols, original reports,
decision files, closure summaries, and independent audits in the two result
directories. The 14 snapshots total 136,057 bytes. Five copies replace local
paths or the target GPU identifier with placeholders; apart from these fields
and newline normalization, they match the originals. The original bytes remain
in the canonical campaign archives. The optional raw archive is held by the
author outside this repository. It is not part of Git, may change or disappear,
and nothing in CI reads it. It is not linked while its raw rows contain
identifying values. The archive is 24,809,923 bytes, SHA-256
`9043e2f675b46620fb38accc0cf85ef11c3b0a3acc7f181b22cd824501ea117d`.
The [manifest](results/issue2162-later-cells-v1/manifest.json) binds every Git
snapshot and the four canonical nested ZIPs. The archive's `inventory.json`
binds every member. The two final campaign archives and four 5090 failure
archives remain byte-identical. Git does not silently fetch the archive.

Without the archive, the verifier checks Git snapshot integrity, recomputes the
six-block medians from the published block medians and decision files, and names
the raw checks that remain unavailable. With the archive supplied, it checks SHA-256,
CRC and extraction containment, reconstructs each temporary campaign tree,
runs the original CPU auditors, and requires their outputs to match the
published audits except for completion timestamps; both reports must match
after LF newline normalization (and matched byte for byte on the original
Windows host). This optional mode executes Python scripts from the archive,
which are not in this diff: `rtx5090/evidence.zip!audit-5090-desktop-resume.py`
and `rtx3090/scripts/audit.py`. Inspect them before running it. It never
executes a GPU model:

```sh
python benchmarks/harness/issue2162_later_cells.py
python benchmarks/harness/issue2162_later_cells.py --bundle issue2162-later-cells-v1.zip
```

The optional full replay returned `verified: true`: 5090 Stage A false with 18 timings,
second 3090 Stage A and confirmation true with 30 timings. That verifies the
preserved record, not the unobserved effects of WDDM contention or whether the
locally frozen protocol met the owner's public review process.
