# Sustained sequential read from one laptop NVMe drive

## Question

What read rate does one laptop NVMe sustain over 3600 s with a sequential,
unbuffered workload larger than system RAM? How does the rate hold up after
warm-up as the drive heats?

## Environment and harness

The 2026-10-15 run used a Windows 11 laptop (build 26200 in the smartctl
banner), a Micron MTFDKCD512QFM-1BD1AABLA 512 GB NVMe SSD (firmware
1002V3LN, `C:` on physical disk 0), and 16,857,817,088 bytes of physical RAM.
The recorded
software was Python 3.11.15, psutil 7.2.2 and smartctl 7.5. The CPU model is
unspecified. The workload used raw synthetic bytes. A model, tensor dtype,
GPU, VRAM, CUDA and torch were inapplicable.

The `runner/sustained.py` harness in the reproduction archive reads a
32 GiB fixture (34,359,738,368 bytes) of OS-generated random bytes using
`open_direct` with Windows unbuffered I/O. It reads sequentially in aligned
4 MiB chunks and wraps at end of file. The harness checked the fixture's
certificate, size and modification time at run time. For the recorded
`seq4m` run, `REPO` denotes the checkout used to validate the protocol
commit. The offline command uses the extracted reproduction tools and
raw-results archive.

```text
python runner/sustained.py run --pattern seq4m --repo REPO --commit c8e7dd35c3ed793bdd61c47ae9af3e897751136f
python tools/analyze_c4.py evidence regenerated
```

## Method

The selected run completed 3600 s. The first 600 s were warm-up for the analysis.
The planner heuristic is 0.9 times the 10th percentile of the 50 complete,
non-overlapping 60 s post-warm-up means. Reader counters were sampled about
once per second and SMART temperatures about every five seconds. The protocol
required mains power, an unobstructed laptop, a starting drive temperature
at or below 40 °C and an experimental stop at 75 °C. It also required checking
for suspend and other processes' disk reads. The selected run passed these
checks, with mains power in all 1793 samples. No other process reached
1,000,000,000 logical read bytes and every process was inspectable.

## Results

Throughput uses decimal GB/s. The table preserves the precision of the
machine-readable analysis.

| Metric | Observed value |
|---|---:|
| Duration (s) | 3600.0 |
| Reader bytes | 5,375,101,239,296 |
| Whole-run throughput (GB/s) | 1.4930836775822223 |
| First 600 s (GB/s) | 1.4936244818193058 |
| After 600 s (GB/s) | 1.4929755167348053 |
| Last 600 s (GB/s) | 1.4927929890133333 |
| Lowest complete post-warm-up 60 s mean (GB/s) | 1.4900473277098665 |
| P10 of post-warm-up 60 s means (GB/s) | 1.4916292514065068 |
| Provisional planner heuristic (GB/s) | 1.3424663262658563 |
| Drive temperature at start / maximum / end (°C) | 29 / 65 / 65 |
| Reader samples / thermal samples | 3600 / 713 |
| Physical read bytes above the reader count | 100,305,920 |
| Aggregate physical write bytes | 156,771,328 |

**Verdict: POSITIVE.** One valid hour-long run measured 1.4929755167348053 GB/s after warm-up, specific to this host, fixture and read pattern.

## Limitations

Five other attempts were excluded: a temperature stop under an earlier cutoff,
a stale sensor, a full run over the foreign-read threshold, an interruption and
an incomplete run without a final validity summary. The attempts exceeded the
declared single replacement. The rate comes from one passing exploratory
candidate, so repeatability and formal protocol acceptance remain open.
Windows reports process I/O as logical reads across devices and disk counters
as physical aggregates. The sources cannot attribute individual dips or
isolate all other physical I/O. The source did not log firmware thermal-transition
counters, so the cause of any short dips remains unknown. Other drives, read
patterns and training workloads were outside this run. The planner number is
a host-specific heuristic.

Measured by @Artix6034

Evidence: https://github.com/SoupTeam/soup-rnd/tree/b3c86a29dbb7efe40d1646595961f250347ce683/benchmarks/results/c4-local
