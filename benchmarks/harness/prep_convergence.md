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
