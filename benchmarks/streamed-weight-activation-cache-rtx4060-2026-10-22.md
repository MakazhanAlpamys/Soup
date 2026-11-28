# Resident weights and retained activations in streamed 8B training

Measured by @kok-o on 2026-10-22.

## 1. Question and setup

Which use of spare GPU memory passes the practical speed gate for streamed 8B training:
persistent decoder weights, retained activations, or both?

The Linux CUDA measurements used one physical desktop RTX 4060 (8188 MiB), driver
580.95.05 and a pre-staging usable RAM lower bound of 127228428288 bytes. The public
environment leaves the Linux release unspecified. Python 3.12.3, Torch 2.8.0+cu128,
Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1 and Accelerate 1.14.0 ran over Soup
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`. The allocator cap was 7 GiB with
a 0.75 GiB workspace reserve.

The model was `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`,
with dense BF16 weights and rank/alpha 4 PEFT LoRA on seven projections.
Context was 1024, microbatch one and accumulation two. The fixed synthetic inputs
and initializer were shared across arms.

## 2. Method

Ordinary streaming was the control. The planner selected layers 16–19 for weights
alone and layer 17 for retained activations alone or the combined arm under the same
cap. Reference-first allclose (`rtol=.01`, `atol=1e-5`), exact inputs/initial adapters,
finite tensors, three-decimal loss agreement and causal/padding checks preceded timing.
Each candidate had four fresh AB/BA/BA/AB pairs, six warmup and eight timed updates.
The gate required at least 2% median paired reduction, at least 75% positive pairs and
a positive 95% whole-pair bootstrap lower bound (20,000 resamples, seed 792).

## 3. Results

| Candidate | Median step s, control / candidate | Paired reduction | 95% bootstrap | Positive pairs | Gate |
|---|---|---:|---|---:|---|
| Resident weights | 9.973 / 9.440 | 5.347% | [5.025%, 5.685%] | 4/4 | Pass |
| Retained activations | 9.950 / 10.007 | -0.431% | [-0.794%, -0.305%] | 0/4 | Fail |
| Both | 9.967 / 9.892 | 0.788% | [0.072%, 1.047%] | 4/4 | Fail |

| Arm | Allocated GiB | Reserved GiB | CPU high-water RSS GiB |
|---|---:|---:|---:|
| Control | 4.142 | 4.709 | 18.436 |
| Resident weights | 5.579 | 6.162 | 18.428 |
| Retained activations | 4.705 | 5.150 | 18.429 |
| Both | 4.705 | 5.143 | 18.435 |

Allocator figures are maxima over measured updates and RSS is the process high-water
mark. All eligible primary and semantic captures matched bitwise. Every timing pair
matched all 14 losses and all 504 final adapter tensors. Each arm changed its adapters.

**Verdict — mixed.** Resident weights passed the timing gate. Retention alone was slower and the combined selection fell below the practical threshold.

## 4. Commands

The frozen setup, qualification worker, timing worker and analyzer live at the evidence
commit. The setup script supplies the environment and runs the timing worker after
authenticated qualification on the exact GPU. The replay requires the retained tensor
archives at the recorded local roots.

```bash
python benchmarks/stream8-cache-20261022/prepare.py --output output/cache-repro
/root/stream8-cache-resume/venv/bin/python /root/stream8-cache-resume/pin_entry.py /root/stream8-cache-resume/timing_worker.py
python benchmarks/stream8-cache-20261022/resume/analyze.py --root output/stream8-cache-resume-20261022-f --reference output/stream8-cache-resume-20261022-d --output output/cache-independent-replay.json
```

## 5. Limits and run history

The initial transport preflight and later setup failures produced no qualified timing.
One timing cohort was void after incomplete archive transfer. The accepted twelve pairs
were rerun in full on the same physical GPU used for the retained qualification. The
full tensor archives remain local, with receipts and compact per-run files at the
evidence commit. Different resident layer counts prevent isolating an additive interaction.
The step gate excludes setup, export and saving. The measured overlay predates pending
`training.stream_resident_layers`. Native activation retention has the
`training.stream_recompute` field.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/stream8-cache-20261022
