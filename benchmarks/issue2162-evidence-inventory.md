# Issue #2162 evidence dependencies

This is a measurement record. The QKVO experiments remain below their fixed
primary rules. B's hand-chosen QKVO plus gate/up layout passed its frozen
two-stage rule on one RTX 3090 24 GiB capped at 8 GiB. Gate/up retention is
outside #2162's QKVO-only scope; this record proposes no production feature.
Later 5090 and second-3090 campaigns are outside this record.

## In Git and external bulk

Git retains timing receipts with every duration/loss, exactness metadata and
shared reference tensors, freezes, protocols, ledgers, census, historical audits,
failed-attempt provenance, manifests and MLP `numerical-inputs.json`. The latter
holds all 30 original B timing receipts without aggregation, the six gate memory
peaks and eight historical contrasts. Its original-file hashes refer to the
`mlp-budget-b-v1/files.json` map. With bulk supplied, the same selection is rebuilt
from the restored originals and must equal this numerical input exactly.

The four bulk kinds (13 files, 9,282,142 uncompressed bytes) are outside Git:

| Evidence directory | External paths |
|---|---|
| `issue2162-rtx3090-v1` | `census-v1/resources.jsonl`, `exactness-v1/resources.jsonl`; `resources.jsonl`, `source-blobs.json`, `source-hashes.json` in each of `timing-v1`, `reallocation-v1`, `scoped-v2-resume1` |
| `issue2162-mlp-v1` | `objects.json`, `source-blobs.json` |

[Download the bulk archive](https://github.com/Shutaru/Soup-2162-evidence/releases/download/issue2162-bulk-v1/issue2162-bulk-af962369.zip).
Its SHA-256 is
`52a9a550fc620684698bbb4ea83fc5b4ad894057a0464f9c93f2d8af1eda04b5`.
Both manifests record the archive identity and every bulk-file SHA-256 under
`bulk`, while preserving those hashes in their original published-file maps.
The ZIP preserves the canonical bytes from head `af962369`; all 13 CRCs and
file digests were checked before removing these paths from the Git tree.

Extract the verified ZIP to `bulk/`, so that it contains both `issue2162-*`
directories. No tool downloads or extracts an archive implicitly. `--bulk` takes
that extracted parent directory. Each supplied file is checked by the manifest's
LF-normalized SHA-256 before a full audit starts; a corrupt present file fails
rather than falling back to a numbers-only result. Partial directories name the
remaining missing files and produce a numbers-only result.

## What the two modes check

Without `--bulk`, both audit CLIs complete the checks supported by Git inputs,
return `mode: "numbers-only"`, `numbers_verified: true`, `verified: false`, and
name every missing path and unavailable check. Exit zero means those numerical
checks completed, not that the full gate verdict passed. Numerical corruption
raises an error. Supplying `--verify-git-sources` cannot certify absent bindings.

- QKVO: recompute all eight statistical contrasts, 56 timing medians and memory
  deltas; reconstruct/hash the 84 retained exactness receipts from shared tensors;
  check frozen rosters and failed-attempt provenance. Timing-window timestamps
  check recorded timing order, but do not replace process admission timestamps.
- B: recompute all three primary contrasts, six confirmation pairs, family-12
  intervals, 30 memory comparisons and eight historical contrasts from Git's
  numerical rows. `primary_passes` and full stage eligibility are unavailable
  (`null`); `numerical_rule_passes` reports only the numerical rule.

The full operational verdict cannot be recomputed without the resource rows:
the auditors inspect individual GPU/host/CI samples and sampling gaps. A's raw
operator/mechanism and planning-screen verification, tensor replay checks and
both complete MLP audit outputs require the originals in `objects.json`. The
historical A audit remains in Git, but numbers-only mode does not relabel it as
independently recomputed. Source inventory checks require `source-hashes.json`
and `source-blobs.json`; Git-blob verification additionally requires
`--verify-git-sources CHECKOUT` and reachable frozen objects/RTX source commits.

With complete bulk evidence, the QKVO full audit body runs against a temporary
complete tree. The MLP reader restores all 281 original files byte for byte,
checks the new numerical selection against those originals and runs both
independent auditors. Their full outputs must equal the historical closure
outputs. Git verification checks 2,927 QKVO and 3,654 MLP bindings (2,010 distinct
blobs), retaining the original source identities and newline conversions. Three
small original auditor/test source versions are retained as `frozen/git-*.txt`
snapshots, named and hash-bound in the MLP manifest. This keeps their blob IDs
reachable while excluding the old bulk-containing record commits from this
branch history. The three RTX source commits remain ancestors; the MLP base
commit is a provenance label and its verifier reads the recorded blob objects.

## Reproduce without CUDA

```sh
# Git inputs alone: numbers and an explicit missing-check list.
python benchmarks/harness/issue2162_audit.py \
  --evidence benchmarks/results/issue2162-rtx3090-v1 --output numbers-qkvo.json
python benchmarks/harness/issue2162_mlp_record.py \
  --evidence benchmarks/results/issue2162-mlp-v1 --output numbers-mlp.json

# Complete evidence and Git sources, after verifying/extracting the archive.
python benchmarks/harness/issue2162_audit.py \
  --evidence benchmarks/results/issue2162-rtx3090-v1 \
  --bulk bulk --verify-git-sources . --output audit-2162.json
python benchmarks/harness/issue2162_report.py \
  --evidence benchmarks/results/issue2162-rtx3090-v1 \
  --audit audit-2162.json --output report-2162.md
python benchmarks/harness/issue2162_mlp_record.py \
  --evidence benchmarks/results/issue2162-mlp-v1 \
  --bulk bulk --verify-git-sources . --output audit-2162-mlp.json
python benchmarks/harness/issue2162_mlp_report.py \
  --evidence benchmarks/results/issue2162-mlp-v1 \
  --bulk bulk --output report-2162-mlp.md
```

Both generators render a shorter numbers-only report, including missing checks,
when given a numbers-only audit. Full mode regenerates the checked-in gate records.
Default tests need no archive or network. Set `SOUP2162_BULK` to an extracted
archive to run the two full-evidence integration tests, including Git checks and
report reproduction. Original model/source ZIPs and all attempts remain preserved;
preparing this revision executes no model.
