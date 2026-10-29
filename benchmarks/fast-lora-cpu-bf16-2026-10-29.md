# CPU bf16 Fast-LoRA MLP: Ryzen 7 7840HS, 2026-10-29

On this host, #2187's seed-4 failure reproduces with torch 2.8.0+cpu in the
native AVX512, `default` and `avx2` modes. The failing gradient is
`mlp.up_proj.lora_B.default.weight`, shape **48 x 4**. The other eleven seeds
pass. Overriding the instruction path does not remove the failure here.

With torch 2.14.1+cpu all twelve seeds pass in all three modes, but the seeded
bf16 input changes between torch versions. This observation does not establish
that the kernel's numerical behavior improved on an identical input.

Refs [#2187](https://github.com/MakazhanAlpamys/Soup/issues/2187). No kernel or
criterion change is proposed by this measurement record.

## Source and environment

- Soup source: `bba95299d305025e20890b113642d2f0695cc59f`.
- Kernel Git blob: `ff40fc7032205101b29945e920be8db613a821dd`.
- Test Git blob: `d2a7e8a93abf8fc8497e2f7c99cb197705ec3c40`.
- CPU: AMD Ryzen 7 7840HS with Radeon 780M Graphics, 8 cores / 16 logical CPUs.
- Windows; Python 3.12.15; PEFT 0.21.2; transformers 5.19.0.
- Separate torch 2.8.0+cpu and 2.14.1+cpu environments; one intra-op thread
  (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). No model or dataset downloads.
- The existing `aten_half_matmuls` fixture runs unchanged: MKLDNN is disabled
  during the test and the SDPA math backend is selected. Each capability setting
  is applied before torch imports, in a fresh process.

## Results

The errors below are maximum absolute deviations from the test's float64 oracle
for seed 4's **up-projection LoRA B gradient**. The unchanged criterion is
`fast_error <= 2 * peft_error + 1e-8`; the separate output bound is 1.05x.

| torch | Requested capability | Reported capability | Fast error | PEFT error | Ratio | Twelve-seed sweep |
|---|---|---|---:|---:|---:|---|
| 2.8.0+cpu | unset | AVX512 | 6.5987193124e-4 | 3.0533368472e-4 | 2.161150 | seed 4 fails; 11 pass |
| 2.8.0+cpu | default | NO AVX | 6.5987095005e-4 | 3.0533957401e-4 | 2.161105 | seed 4 fails; 11 pass |
| 2.8.0+cpu | avx2 | AVX2 | 6.5987193124e-4 | 3.0533368472e-4 | 2.161150 | seed 4 fails; 11 pass |
| 2.14.1+cpu | unset | AVX512 | 3.7086410697e-4 | 3.7391884501e-4 | 0.991830 | 12 pass |
| 2.14.1+cpu | default | DEFAULT | 3.7085353167e-4 | 3.7390826972e-4 | 0.991830 | 12 pass |
| 2.14.1+cpu | avx2 | AVX2 | 3.7086410697e-4 | 3.7391884501e-4 | 0.991830 | 12 pass |

The initial three instrumented 2.14.1 sweeps and their instrumented repeats agree
on the verdict. Raw per-seed, per-tensor errors and fixture fingerprints are retained
in [the JSON record](records/fast-lora-cpu-bf16-20261029.json).
The published recorder reproduces all 72 rows and fixture hashes exactly.
Uninstrumented seed-4 controls also fail on 2.8.0 in all three modes and pass
on 2.14.1 in the native mode; the failed controls retain pytest exit status 1.

The issue's original torch 2.8.0+cu128 report gives approximately 6.5988e-4 fast
and 2.0565e-4 PEFT (3.2x). This host's CPU build reproduces the seed and bound
violation, but not that exact reference error or ratio. The original report
has not identified its CPU model or which B gradient failed, so neither is
inferred from these measurements.

## Fixture comparison and limits

All twelve `x0` byte hashes differ between torch versions. The model parameters
are byte-identical across versions when comparing the same capability mode.
Within a version, native and AVX2 agree; the default path changes the three
float32 LoRA B initialization hashes. These are separately seeded experiments,
not a fixed-tensor comparison across builds or instruction sets.

The failure persists under all three requested instruction paths on torch 2.8.
This narrows the evidence on this CPU; it does not decide whether to change the
kernel or the 2x criterion. A controlled replay of identical input tensors and
confirmation on the original host/build remain useful next steps. No CUDA,
performance, memory-saving or training-quality result is claimed.
The overrides here are the torch/ATen capability settings requested in the issue;
independent BLAS-library dispatch was not constrained or inspected.

## Reproduce

Use a fresh process for each capability. From the checkout root, the original
unmodified test command for one seed is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -o addopts='' -m 'not gpu' \
  'tests/test_issue837_fast_lora_mlp.py::test_bf16_mlp_stays_within_twice_pefts_error_against_float64[4]' -q
```

The sweep with observational error/fingerprint capture is:

```bash
PYTHONPATH=benchmarks/harness:src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SOUP_BF16_CAPTURE=run.json python -m pytest -p record_fast_lora_cpu_bf16 \
  -o addopts='' -m 'not gpu' tests/test_issue837_fast_lora_mlp.py -k twice_pefts -q
```

Repeat with `ATEN_CPU_CAPABILITY=default` and `ATEN_CPU_CAPABILITY=avx2` set
before starting Python; leave it unset for the native run. On Windows, set the
variables through `$env:NAME='value'` and separate `PYTHONPATH` entries with `;`.
The recorder observes existing test locals after return/failure and performs no
tensor writes. Run without coverage or a debugger trace. A failed original
assertion still exits 1 and remains a failed measurement; the recorder does not
relax the bound or turn it into a pass.
