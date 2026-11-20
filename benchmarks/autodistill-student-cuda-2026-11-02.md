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

The CUDA run predates the review updates to the tests and the objective docstring.
It establishes neither default top-k or loss weights nor compression or model quality.
The maintainer still needs to agree the next #580 scope.

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

The reports are committed as text with a final newline. Hashes below use LF
line endings, as stored in Git. Convert CRLF to LF before hashing a Windows
checkout that uses CRLF. The source provenance records both Git blob and
transferred-file hashes. Windows `git archive` converted LF to CRLF, and the
transferred hashes match the remote source files.

| File | SHA-256 (LF) |
|---|---|
| [`all.log`](results/autodistill-student-cuda-2026-11-02/all.log) | `e113e9d8b52ff6f544591e798da448182e80314987ee76edcedf6532bb2d5770` |
| [`all.xml`](results/autodistill-student-cuda-2026-11-02/all.xml) | `05e7bfec5d21490a1cf47795c951159a7280de0cec5669fd1e50f8146bb11c0e` |
| [`cpu-suite-summary.json`](results/autodistill-student-cuda-2026-11-02/cpu-suite-summary.json) | `47a4ed6d973361848774c0cc4777edd79bc28207ecab62cb51d8fec8ad937435` |
| [`cuda-evidence.json`](results/autodistill-student-cuda-2026-11-02/cuda-evidence.json) | `2df4148e464e35ebd3e8315c120ed30e27b0fd1a2368a157e500fcc9876101cf` |
| [`cuda.log`](results/autodistill-student-cuda-2026-11-02/cuda.log) | `9a530402132f5e3753d240885f3ab914f4ff0086e3b895472e98e644a0796f37` |
| [`cuda.xml`](results/autodistill-student-cuda-2026-11-02/cuda.xml) | `2cb9aca34a57eb0cef25551abd23c922c1245dd214cce279899921980c41bfd3` |
| [`source-provenance.json`](results/autodistill-student-cuda-2026-11-02/source-provenance.json) | `ee0d222c18d4f2e1cf22f7619dac9d57cbc0dc5e3903d255e2649bdcdb0e2566` |

## Local full-suite control

On the same source commit, Windows / Python 3.12.15 / Torch 2.14.1+cpu completed
**48,716 passed, 6 failed, 507 skipped**, with **88.0277% coverage** (77% required).
The failure set matches six targeted checks on clean main `74081886` exactly:
one executable-search assertion, three long executable-path cases and two child-pytest
coverage-loading cases. Their exact names and the command are in `cpu-suite-summary.json`.
This is not a green full-suite result. The measurement record and its links were added
afterward; runtime source did not change after the full run.
