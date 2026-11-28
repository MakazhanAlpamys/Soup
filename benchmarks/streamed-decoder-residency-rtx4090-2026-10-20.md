# Persistent decoder weights in streamed 7B training

Measured by @kok-o on 2026-10-20.

## 1. Question and setup

Does keeping selected frozen decoder weights on the GPU shorten optimizer steps while
preserving the training and padding behavior of ordinary layer streaming?

One Linux CUDA host with an RTX 4090 (24 GB, 24564 MiB reported), driver 590.48.01
and driver-reported CUDA 13.1. Every arm used a 10 GiB CUDA allocator cap and a
13.0000 GiB pinned host weight store. Total host RAM and the Linux release are
unspecified in the retained environment. The stack was Python 3.12.3, Torch
2.8.0+cu128, Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1 and Accelerate 1.14.0.

The model was `openlm-research/open_llama_7b_v2` at
`e5961def23172a2384543940e773ab676033c963`, with dense BF16 weights and ordinary LoRA
rank/alpha 4 on all seven projections, dropout 0, nonzero initial adapters and seed 792.
The fixed synthetic fixture has 32 rows, sequence 1024, microbatch 1 or 2 and
accumulation 2. The tested Soup baseline was
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`.

## 2. Method

The control used two rotating slots. Candidates gave layers 12–19 or 8–23 private
GPU slots and streamed the remaining layers through the same pool. The pinned host
store stayed intact. Before timing, the reference-first comparator required exact initial
tensors and inputs, finite tensors, loss agreement to three decimals, and tensor agreement
at `rtol=1e-2`, `atol=1e-5`. Causal-prefix and masked-padding checks ran separately.

Each candidate and microbatch used four fresh-process pairs in AB/BA/BA/AB order,
with six warmup and eight timed optimizer updates per process. Final adapters and losses
were checked after all 14 updates. The practical gate required a median paired time
reduction of at least 2%, at least 75% positive pairs and a positive 95% whole-pair bootstrap
lower bound (20,000 resamples, seed 792).

## 3. Results

Step times are medians of process medians. Memory is the median per-process measured-window peak.

| Resident layers | Microbatch | Step s, control / candidate | Paired reduction | 95% bootstrap | Allocated GiB, control / candidate | Reserved GiB, control / candidate |
|---|---:|---|---:|---|---|---|
| 8 | 1 | 1.974721 / 1.662359 | 15.826% | [15.795%,15.840%] | 1.8448 / 4.8606 | 2.1230 / 5.1387 |
| 8 | 2 | 2.197845 / 2.064987 | 6.038% | [5.887%,6.058%] | 2.4739 / 5.4896 | 3.0078 / 6.0234 |
| 16 | 1 | 1.974758 / 1.357570 | 31.246% | [31.191%,31.332%] | 1.8448 / 7.8763 | 2.1230 / 8.1543 |
| 16 | 2 | 2.197580 / 1.942621 | 11.596% | [11.570%,11.624%] | 2.4739 / 8.5053 | 3.0078 / 9.0391 |

The baseline repeat and each candidate's training capture matched all 1813 tensors bitwise.
Both semantic comparisons matched all 903 tensors. Logical decoder GiB/update fell from
45.2362 to 33.1732 with eight resident layers and to 21.1102 with sixteen, at both microbatch sizes.

**Verdict — positive.** All four cases passed the practical gate and kept bitwise capture parity.

## 4. Commands

The frozen harness and configurations live at the evidence commit. These are the archived
batch-one capture commands. Use `b2.yaml` for microbatch two, layers 8–23 for the
sixteen-layer arm, `--semantics` for semantic capture, and `--warmup 6 --steps 8`
in place of `--capture` for timing.

```bash
python /root/liger/prepare_7b.py
python /root/liger/pin_entry.py /root/liger/entry.py --config /root/liger/b1.yaml --output /root/liger/results/check-baseline --capture
python /root/liger/pin_entry.py /root/liger/entry.py --config /root/liger/b1.yaml --indices 12,13,14,15,16,17,18,19 --output /root/liger/results/check-resident8 --capture
```

## 5. Limits and run history

The cap excludes CUDA driver context. The fixed-microbatch test leaves open how the same
memory would perform if spent on larger microbatches. Four pairs and one synthetic workload
limit generalization. The tensor archives remain in local retention. A network interruption
paused export and the same evidence was recovered. No measured arm was superseded.
The measured overlay predates the pending `training.stream_resident_layers` setting.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/evidence/streaming-residency-20261020
