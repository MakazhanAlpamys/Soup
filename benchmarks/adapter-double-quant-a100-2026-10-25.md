# Measurement record - double-quantization calibration for OLMoE LoRA

Measured by @Lameda12. Calibration began 2026-10-25 UTC and continued into
2026-10-26 UTC.

## 1. Question and setup

Does NF4 double quantization of a frozen base change what LoRA learns on
`allenai/OLMoE-1B-7B-0924` (revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`)?
The planned comparison used NF4 fake quantization with BF16 write-back, double
quantization off versus on, with PEFT's expert parameter wrapper. Unquantized
BF16 served as a scale reference.

Calibration used one NVIDIA A100 PCIe 40 GB (40,960 MiB, compute capability 8.0)
on a hosted Linux instance, driver 595.99.02, driver CUDA 13.2 and PyTorch CUDA
13.0. The recorded stack was Python 3.12.15, PyTorch 2.14.1, Transformers
5.19.0, PEFT 0.21.2 and bitsandbytes 0.50.2. The sampled calibration run had
650,415,583,232 available RAM bytes before execution. Total RAM, host CPU
and OS release were unavailable in the run stamps.

## 2. Pre-registered decision and calibration method

The main rule fixed the metric as token-weighted response-target negative
log-likelihood (NLL) on 2,048 held-out decision rows, with FP64 sums and
identical data order within each of seeds 42, 7 and 123. The primary effect
was NLL with double quantization on minus NLL with it off. The rule called for
a 90% percentile interval from 2,000 paired row-bootstrap resamples shared
across arms and seeds. After validity checks, an interval wholly inside
±0.005 nat would pass, wholly below −0.005 nat would pass as a benefit, and
wholly above +0.005 nat would fail. A failed validity check or an interval
crossing a boundary would give no verdict. Suite scores were exploratory.
Filled decision rule

Calibration used separate training and smoke rows. It ran short training
arms of 16 updates with LoRA rank 16, alpha 32, microbatch 1 and
accumulation 16 to check memory and update time. It compared sampled expert operations
with reference arithmetic, measured wrapper-control tolerances and suite
noise floors, and repeated one arm to check NLL and adapter repeatability.
The main harness added per-stage upload of attempt records, the journal and
adapter files. It resumes from the journal, skips completed stages and retains
void counts across restarts. The full decision series remained pending.
Main harness and plan

## 3. Commands

The calibration harness lives at execution commit
`5ab786249abc732cf5ce911d3d8eb24def702a7d`. With the verified data root,
split manifests and run directory prepared, the runbook uses these commands.
The fallback command documents the attempted stage. The fixed reporter lives
at the linked runbook revision.
Calibration runbook

```bash
H=benchmarks/harness/e4_part2_calibration.py
python $H split --data-root $DATA --out $SPLIT --dataset-revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
C="--out $RUN --data-root $DATA --split-dir $SPLIT --manifest $MAN --device cuda --gpu-index 0 --instance-id $IID"
python $H drive --stage F --f-prefix F $C
python $H run --stage T $C --run-id T
python $H run --stage T $C --fallback-gateup --run-id T_fallback_gateup
python $H run --stage W $C
for i in 1 2 3 4 5; do python $H run --stage N --repeat $i $C; done
python $H run --stage P $C
python $H report --out $RUN
```

## 4. Calibration results

| Base arm | Peak reserved bytes | Median seconds per update |
|---|---:|---:|
| NF4, double quantization off | 17,846,763,520 | 8.030015620053746 |
| NF4, double quantization on | 17,846,763,520 | 7.98066275799647 |
| Unquantized BF16 | 17,846,763,520 | 8.152295885025524 |

| Calibration check | Recorded result |
|---|---|
| Card selection | A100 40 GB selected against a 36,000,000,000-byte reserve limit |
| Expert-operation thresholds | 31 cells covered. Gate and up projection fallback thresholds remained open |
| Wrapper-control tolerances | Output 0.2197437286376953, input gradient 0.017418980598449707, LoRA A gradient 0.6594619750976562, LoRA B gradient 0.08243513107299805, update 0.0004000067710876465 |
| Suite noise | Floor 0.0 on all 8 suites. Per-suite flag threshold was one item. The safety mini-suite scored 0.0 on every run |
| Repeatability on smoke rows | NLL 1.626105685835482 and 1.6260342313352905, difference -7.145450019163668e-05. Adapter files differed |
| Projected main-series time | 5.722281638913684 hours for training updates alone |

**Verdict — no verdict. No verdict row passed through a model, and no double-quantization verdict exists.**

## 5. Limitations and superseded attempts

- The final-card calibration superseded the earlier A100 calibration from the
  retired instance. On the final instance, two missing-input-verification setup
  voids preceded the measured restart and produced no measurements.
- The restart journal records three voids and zero failed runs. The interrupted
  suite-noise attempt was rerun successfully. The gate/up fallback exceeded
  its distinct-row pool and then had an unclosed attempt, leaving four required
  output cells uncalibrated. No recorded attempt used the revised pool-based
  targets. The original summary omitted both fallback voids. The reporter fix
  includes every run and unclosed attempt in the journal.
- The 28 SHA-256 manifest entries cover 16 available files and 12 adapter files
  held separately. The held-out split manifests and box-side verification
  output are absent from the calibration packet. The projected schedule omits
  loading, evaluation, controls and retries. The safety mini-suite was already
  at its floor, leaving refusal behavior unverified.

Evidence: benchmarks/evidence/adapter-double-quant-a100-2026-10-25/ (source commit fc75a5156984)
