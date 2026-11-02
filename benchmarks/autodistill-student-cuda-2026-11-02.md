# AutoDistill captured-token objective: CUDA qualification, 2026-11-02

This is a numerical qualification record for the proposed internal student-loss
slice of [#580](https://github.com/MakazhanAlpamys/Soup/issues/580), by @kok-o.
Source: `794fecabf77e77c97071b4f71b63d48fe42936ba`.

## Result and scope

| Command selection | Passed | Failed | Skipped | Pytest duration |
|---|---:|---:|---:|---:|
| Five CUDA-marked cases | 5 | 0 | 0 | 10.96 s |
| Complete student-loss test file, including those five | 52 | 0 | 0 | 6.70 s |

The second row includes the first row's cases; these are not 57 independent tests.
Durations include imports/startup and are not a throughput comparison. No peak-memory,
pretrained-model quality or complete AutoDistill workflow gate was run.

Three CUDA cases compare sparse coarse-tail KL and analytic gradients against a
separately expressed grouped-probability oracle at temperatures 0.5, 1 and 2.5.
Two CUDA cases compare actual tiny Llama/AdamW updates with an independent dense
teacher objective, for full-parameter and PEFT LoRA training. They verify finite
gradients, matching updates, changed student logits and unchanged frozen parameters.

The tiny fixture has vocabulary 8, hidden size 16, intermediate size 24, one Llama
layer, two attention heads, one KV head and dropout disabled; seed 19. Three recorded
contexts have lengths 2, 3 and 2. The update uses temperature 1.7, CE/KD weights
0.3/0.7, AdamW learning rate 0.01 and weight decay 0.03. LoRA uses rank/alpha 2 and
q/v targets. Selected probabilities are not renormalized and no implicit T-squared
multiplier is used. Gradient/update tolerances are stated in the pinned test source.

The GPU cases explicitly use float64 student logits/backbones. The test file's
half/bfloat16/float32 edge cases execute on CPU. This record does not qualify reduced
precision GPU training, MLX/Apple Silicon, cache admission, teacher-process isolation,
consumption/checkpoint state, exact resume or the adaptive loop.

## Host and environment

- Vast.ai, one NVIDIA GeForce RTX 3090, 24 GiB; compute capability 8.6.
- Driver 615.71.09; Torch 2.14.1+cu132; CUDA runtime 13.2.
- Curated image `vastai/pytorch:cuda-13.2.1-auto`; Linux 6.8.0-147-generic,
  glibc 2.39, Python 3.12.15; interpreter `/venv/main/bin/python`.
- Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, Pydantic 2.14.1, pytest 8.4.2.
- Torch/torchvision were constrained to the image's installed versions during setup.
  No teacher or pretrained model weights were downloaded.
- The JSON's float32/TF32 fields describe the launcher process. Pytest runs in fresh
  processes; those flags were not separately injected into its workers. No claim
  about their runtime settings is inferred from the launcher fields.

## Commands and raw evidence

Environment: `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`TOKENIZERS_PARALLELISM=false`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

```bash
python -m pytest --noconftest -o addopts= \
  -o 'markers=gpu: requires a CUDA device' -q --tb=short \
  --junitxml=cuda.xml -m gpu tests/test_autodistill_student_loss.py
python -m pytest --noconftest -o addopts= \
  -o 'markers=gpu: requires a CUDA device' -q --tb=short \
  --junitxml=all.xml tests/test_autodistill_student_loss.py
```

These standalone tests need no repository-wide autouse fixtures. `--noconftest`
avoids those unrelated fixtures and their CUDA-skip policy; the five selected cases
must actually construct CUDA tensors and all five must execute, with zero skips.
This is not a GPU run of the whole repository suite.

The byte-original `cuda-evidence.json`, `cuda.log`, `cuda.xml`, `all.log` and
`all.xml` are preserved inside `autodistill-student-cuda-2026-11-02/raw-evidence.zip`.
An independently readable JSON snapshot (with a final newline),
`source-provenance.json` and `cpu-suite-summary.json` are alongside it. The zip
preserves original end-of-file bytes through text hooks and Windows checkouts.
The provenance records both Git blob and transferred-file hashes. Windows `git archive`
converted LF to CRLF; transferred hashes match the remote files exactly, and the only
difference from the Git blobs is that line-ending conversion. Raw Git and transfer
hashes are deliberately not called identical.

## Failed setup and budget

The first RTX 5080 attempt reached SSH, then failed before installing dependencies:
the non-login shell had no bare `python` command. Its container was destroyed.
The successful attempt used the documented `/venv/main/bin/python` environment and
verified a real CUDA tensor operation before installing the small test dependencies.

Both task-created instances were destroyed after report collection, and a separate
API read confirmed neither remained. The pre-existing instance was left untouched.
The two rental-rate estimates are $0.015694 and $0.016544, **$0.032238 total** against
the authorized $2 limit. Rates include the selected storage allocation and both
offers had zero inbound/outbound traffic price. This is elapsed-time times advertised
rate, conservatively including loading time, not a claim about a final settled invoice;
account-wide credit changes also include the pre-existing rental.

No default top-k, loss weights, compression or production-quality claim follows from
this qualification. The maintainer still needs to agree the next #580 scope.

## Local full-suite control

On the same source commit, Windows / Python 3.12.15 / Torch 2.14.1+cpu completed
**48,716 passed, 6 failed, 507 skipped**, with **88.0277% coverage** (77% required).
The failure set matches six targeted checks on clean main `74081886` exactly:
one executable-search assertion, three long executable-path cases and two child-pytest
coverage-loading cases. Their exact names and the command are in `cpu-suite-summary.json`.
This is not a green full-suite result. The measurement record and its links were added
afterward; runtime source did not change after the full run.
