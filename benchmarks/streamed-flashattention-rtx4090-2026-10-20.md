# FlashAttention with persistent streamed decoder weights

Measured by @kok-o on 2026-10-20.

## 1. Question and setup

At a fixed resident-layer count, does external FlashAttention improve step time over Torch
SDPA while meeting the original training and padding checks?

The accepted Linux hosts each used a 24564 MiB RTX 4090, driver 580.119.02 (reports CUDA
13.0), Python 3.12.3, Torch 2.8.0+cu128, Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1 and
Accelerate 1.14.0. The slower-link host used an AMD EPYC 9474F with reported host-RAM
field 110655 and advertised PCIe bandwidth 6.3 GB/s. The faster-link host used an AMD EPYC
9654 with reported host-RAM field 110654 and qualified pinned H2D at 26.79 GB/s. The
hardware JSON leaves the RAM field's unit unspecified. Both used a 10 GiB CUDA allocator cap
and kept a 13 GiB pinned host store. The public environment leaves the Linux release unspecified.

The fixed model was `openlm-research/open_llama_7b_v2` at
`e5961def23172a2384543940e773ab676033c963`, dense BF16 with rank/alpha 4 PEFT LoRA
on seven projections, dropout 0, nonzero initial adapter B and seed 792. The workload
was the frozen 32-row synthetic fixture, context 1024, microbatch 1 or 2 and accumulation 2.
External FA 2.7.4 used a custom build with Torch arithmetic macros and fast math disabled.
The measured Soup baseline was `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`.

## 2. Method

The hybrid used native SDPA for explicit masks and the external kernel for unmasked
attention. Each pair held residency fixed at eight layers (12–19) or sixteen (8–23).
The original reference-first capture required exact inputs and initial tensors, finite
values, three-decimal loss agreement, and allclose at `rtol=1e-2`, `atol=1e-5`.
The masked-padding and causal gates ran before timing. Qualified cases ran in four fresh
AB/BA/BA/AB pairs, six warmup and eight measured steps per process. The speed gate
required at least 2% median paired reduction, at least 75% positive pairs and a positive
95% whole-pair bootstrap lower bound (20,000 resamples, seed 792).

## 3. Results

| Link cohort | Resident layers | Microbatch | Step s, SDPA / hybrid | Paired reduction | 95% bootstrap |
|---|---:|---:|---|---:|---|
| Slower | 8 | 1 | 5.867194 / 5.879058 | -0.784% | [-1.543%, -0.152%] |
| Slower | 8 | 2 | 6.182374 / 6.158324 | 0.355% | [-0.281%, 0.849%] |
| Slower | 16 | 1 | 4.096665 / 4.114799 | -0.433% | [-0.511%, -0.306%] |
| Slower | 16 | 2 | 4.671641 / 4.617346 | -0.038% | [-0.439%, 2.576%] |
| Faster | 16 | 1 | 1.338522 / 1.384998 | -3.464% | [-3.501%, -3.416%] |
| Faster | 16 | 2 | 1.958509 / 1.959084 | -0.053% | [-0.369%, 0.003%] |

Allocated/reserved peaks were identical within each comparison. The faster-link
sixteen-layer arms used 7.8763/8.1543 GiB at microbatch one and 8.5053/9.0391 GiB at
microbatch two. The primary hybrid captures matched all 1813 tensors and the hybrid
semantic captures matched all 904 tensors bitwise. Pure external attention failed the
cross-backend padded-input check on 445 gradients, one logits tensor and the
three-decimal loss comparison at each tested residency count. It was excluded from timing.
All 24 valid timing pairs had bitwise-equal final adapters and losses after 14 updates.
Each hybrid timing process recorded 1,792 external dispatches.

**Verdict — negative.** No hybrid configuration passed the practical timing gate, and the pure external route failed numerical qualification for timing.

## 4. Commands

The frozen harness and custom build live at the evidence commit. These are the saved
capture command forms. The archived protocol gives the batch-two config, other
layer indices, pure-external arm, and timing order.

```bash
python /root/liger/prepare_7b.py
python /root/liger/pin_entry.py /root/liger/entry.py --attention sdpa --indices 12,13,14,15,16,17,18,19 --config /root/liger/b1.yaml --output /root/liger/results/check-native --capture
python /root/liger/pin_entry.py /root/liger/entry.py --attention hybrid --indices 12,13,14,15,16,17,18,19 --config /root/liger/b1.yaml --output /root/liger/results/check-hybrid --capture
python /root/liger/pin_entry.py /root/liger/h2d_probe.py
```

## 5. Limits and run history

A faster-link cohort with other GPU compute processes had its speed and memory
comparisons voided. A subsequent cohort qualified numerically and stopped during
archive extraction before timing. The final faster-link cohort supplies the valid
repeat. The slow-link observations stand on their own. A native 16-token semantic
profile observed SDPA flash kernels, while the 1024-token profile remained blocked
by a caller precondition and then a transfer preflight. The timing results establish
no full-shape kernel attribution, stock-wheel behavior or task quality. Persistent
weights came from the overlay that predates pending `training.stream_resident_layers`.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
