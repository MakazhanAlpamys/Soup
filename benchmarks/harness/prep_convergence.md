# Convergence data reconstruction

`prep_convergence.py` publishes only the data-preparation slice of #379,
corresponding to STEP 11 and STEP 19 of `gate-h100-validation.md`.
It prepares five disjoint 3,000-row training subsets and one 300-row held-out
sample of `dair-ai/emotion` (`split` configuration). Both resident and streamed
arms must consume the same `train_sN.jsonl` and `task_eval.jsonl` files.

This is a **reconstruction**, not recovered historical data. The record did not
publish the original seed, row indices or prompt template. This script supplies
an explicit new selection and template; its hashes cannot establish the original
dataset bytes or reproduce the recorded accuracy. No training, CUDA measurement,
adapter-init seed alignment or throughput result is claimed.

## Historical protocol and reconstruction choices

The manifest separates two required lists. `from_record` contains:

- The `dair-ai/emotion` dataset and exact-match metric scoring.
- Five disjoint 3,000-row subsets of the 16,000-row train split.
- One fixed 300-row test sample, identical for every run.
- The file names `train_s0.jsonl` and `task_eval.jsonl`.

These come from [the H100 record](../gate-h100-validation.md), STEP 11
(lines 2270–2274, 2284–2286, 2356–2358) and STEP 19 (3281–3282).
`new_choices` contains:

- The `split` config and the caller-asserted revision.
- The SHA-256 selection rule and the explicit seed.
- Exact-text deduplication, keeping the first label.
- Excluding all test text from train.
- The prompt template, label strings and row formats.
- The file names `train_s1.jsonl` through `train_s4.jsonl`.

`--verify` requires both lists with these contents, as well as
`historical_bytes_recovered: false`. Manifests from the earlier PR revision,
which lack these lists, must be regenerated from their pinned exports and seed.

## Prepare and verify

Export `text` and integer `label` columns from the train and test splits at an
immutable dataset revision. For example, in an environment with `datasets`:

```python
import hashlib
import json
from pathlib import Path
from datasets import load_dataset

revision = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
dataset = load_dataset("dair-ai/emotion", "split", revision=revision)
for split in ("train", "test"):
    data = b"".join((json.dumps({"text": row["text"], "label": row["label"]},
                              ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                    for row in dataset[split])
    Path(f"emotion-{split}.jsonl").write_bytes(data)
    print(split, hashlib.sha256(data).hexdigest())
```

Pass the resulting hashes, an explicit integer selection seed and a **new** output
directory. The harness itself uses no ML dependencies and makes no network call:

```bash
python benchmarks/harness/prep_convergence.py \
  --train emotion-train.jsonl --test emotion-test.jsonl \
  --train-sha256 <train-sha256> --test-sha256 <test-sha256> \
  --revision cab853a1dbdf4c42c2b3ef2173804746df8825fe \
  --seed 42 --output convergence-data
python benchmarks/harness/prep_convergence.py --verify convergence-data
python benchmarks/harness/prep_convergence.py --self-test
```

Exit 0 means the preparation/verification checks passed, not that a quality gate
passed. No arguments exits 2 (missing required inputs). Bad source bytes, labels,
insufficient unique rows, an existing output directory or failed verification
exit nonzero. `--self-test` uses synthetic full-size sources on CPU, including
repeatability and a negative tamper control; it is not real-dataset evidence.
Preparation or verification failures exit 1. Validation errors explain the
reason, such as `source hash mismatch`, `output already exists` or
`output hash mismatch`.

Labels follow the dataset order: sadness, joy, love, anger, fear, surprise.
Exact duplicate text is retained once; all test text is excluded from training.
Remaining source indices are ranked by SHA-256 of canonical JSON containing
`[seed, split, source_index, {text, label}]`. The first 15,000 training indices
are divided into five consecutive 3,000-row subsets; the first 300 test indices
form the held-out set. This avoids dependence on Python's global RNG or version.
The script fails if deduplication leaves too few rows.

Training JSONL uses `messages` with user/assistant roles; held-out JSONL uses
`prompt`, `expected` and `scoring: exact` for `soup ship --task-mode metric`.
The manifest records the source revision (caller-asserted), actual source hashes,
seed, selection rule, prompt template, label order, source indices and output
hashes/counts. `--verify` checks bytes, counts, roles, labels and non-overlap.
Preserve the manifest's SHA-256 independently when transferring artifacts:
self-consistency cannot authenticate a manifest changed together with its files,
nor prove that a local export actually came from the claimed upstream revision.

## Reference export and preparation

The real-data check uses dataset revision
`cab853a1dbdf4c42c2b3ef2173804746df8825fe`, the export serialization above,
and selection seed **42**. SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Train export (16,000 rows) | `2caa7ca5999c29915a7e1f4dc0ced104cd622f1e6688904f66e8dd76644da2e8` |
| Test export (2,000 rows) | `7672fb677dfc299705da4dd77e25b168649391d69593f51a4f552c5e56262cff` |
| Current `manifest.json` | `aec02fbaa8e9ad86df8a5d401db59b956e1451cd09a098604a76ee42ba016e79` |

The earlier manifest hash was
`bfd4bccc74c8cee1d0c3df6c37d3db8080edd2f3319422d669a280e42d8c9900`.
Adding the two provenance lists changes the manifest hash; all six output JSONL
files remain byte-identical with the same exports and seed. The run excludes
42 duplicate/test-overlap train rows and selects 15,000 training rows plus 300
held-out tasks. These hashes identify this reconstruction, not the historical
benchmark inputs or training results. The dataset exports are not committed.
