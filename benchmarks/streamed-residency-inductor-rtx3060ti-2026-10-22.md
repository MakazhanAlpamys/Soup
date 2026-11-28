# Resident weights and Inductor on a physical 8 GiB GPU

Measured by @kok-o on 2026-10-22.

## 1. Question and setup

Can persistent decoder weights accelerate dense BF16 streaming on a physical 8 GiB GPU?
Do resident-only Inductor variants preserve numerical parity, and what model size
completed native updates under the same planner?

The qualifying Linux host had an RTX 3060 Ti (8192 MiB), driver 595.91.07 (reports CUDA
13.2), and usable host RAM of 62,411,112,448 bytes at the 8B preflight and 64,212,733,952
bytes at the final 14B preflight. The retained environment leaves the Linux release
unspecified. The CUDA allocator cap was 7 GiB with a 0.75 GiB workspace reserve.
Python 3.12.3, Torch 2.8.0+cu128, Triton 3.4.0, Transformers 5.17.0, PEFT 0.20.0,
TRL 0.29.1 and Accelerate 1.14.0 ran over Soup
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`.

The primary model was `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`.
The native capacity repeat used `Qwen/Qwen3-14B` at
`40c069824f4251a91eefaf281ebe4c544efd3e18`. The frozen dense BF16 recipe used
ordinary PEFT LoRA rank/alpha 4 on all seven projections, dropout 0, seed 792,
nonzero initial adapters from generator 123, 32 synthetic rows, context 1024,
microbatch one and accumulation two.

## 2. Method

Compare ordinary streaming with eager private GPU slots for decoder layers 16–19,
then compile only those decoder inner forwards. Both Inductor candidates used fullgraph
with CUDA graphs and error suppression disabled. The resident scheduler, weight ownership
and checkpointing stayed eager. The original reference-first gate tested exact inputs
and initial adapters, finite tensors, three-decimal loss agreement and allclose at
`rtol=.01`, `atol=1e-5`, followed by causal/padding checks for eligible arms.
Only qualified arms ran four fresh AB/BA/BA/AB pairs, six warmup and eight measured updates.
The speed gate required at least 2% median paired reduction, at least 75% positive pairs and a
positive 95% whole-pair bootstrap lower bound (20,000 resamples, seed 792).

## 3. Results

| Pair order | Ordinary median step, s | Resident median step, s | Reduction |
|---|---:|---:|---:|
| AB | 5.970576 | 5.837229 | 2.233% |
| BA | 5.989684 | 5.842737 | 2.453% |
| BA | 5.991754 | 5.850807 | 2.352% |
| AB | 5.995708 | 5.850630 | 2.420% |

The paired median was 2.386%, with 4/4 positive pairs and 95% interval [2.233%, 2.453%].
Resident weights raised whole-run allocated/reserved peaks from 4.142/4.709 GiB to
5.579/6.162 GiB. The 2037 training and 1015 semantic tensors were bitwise equal, as
were final adapters and all 14 losses within each timing pair.

| Candidate | Failed final adapter tensors | Failed gradient tensors | Failed logits tensors | Three-decimal loss gate |
|---|---:|---:|---:|---|
| Default Inductor | 504 | 694 | 4 | Fail |
| Precise Inductor | 504 | 656 | 4 | Fail |

Both compiled variants executed real graphs and were excluded from timing.
Qwen3-14B completed two native optimizer updates, with 2261 finite saved tensors,
560 changed adapter tensors and 1120 nonzero gradient tensors. Its allocated/reserved
peaks were 5.132/5.674 GiB and CPU high-water RSS was 33.054 GiB. The planner selected
zero resident decoder layers after reserving workspace. The 32B run was screened out
at 63,534,301,184 usable host bytes against 74,114,181,120 required. The 24B fallback
stopped at staging admission.

**Verdict — mixed.** Eager residency passed for 8B, both compiler variants failed accuracy, and 14B completed a short native-only run.

## 4. Commands

The executed setup, worker, model-preparation script and benchmark live at the evidence
commit. The archived setup invokes the worker below. It generates the per-arm
`entry.py` invocations, including the selected Inductor modes and native capacity repeat.

```bash
/root/liger/venv/bin/python /root/liger/pin_entry.py /root/liger/worker.py
```

## 5. Limits and run history

The October 21 and 22 retention preflights stopped before model execution. In the
standalone RTX 3050 preflight, the transfer was hash-verified at 0.378901 MiB/s
against a 2 MiB/s minimum. It produced no RTX 3050 training measurement. Subsequent
incomplete capture exports are historical and excluded from the final timing. The
final repeat recovered the exact original 14B scientific trace by SHA-256. Its earlier
archive remains incomplete. The full canonical tensor archives are retained locally.
Four pairs characterize this host and fixed recipe. The 14B run establishes short
native fit under the measured setup. The measured overlay predates pending
`training.stream_resident_layers`.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/stream8-compile-20261022-resume
