# Linux NVMe read depth on an RTX 3090 host

Measured by @kok-o on 2026-10-18.

**Verdict (negative).** Both bounded-write diagnostic comparisons fell short of the
benefit rules for two pending reads per drive. The strict quiet-disk gate has no verdict.

## 1. Question and setup

Does allowing two pending layer reads on one physical NVMe, with four range workers
and the same staging budget, improve direct-read throughput and shorten a streamed
training step enough to justify changing the default? The training comparison held
`training.stream_read_ahead` at 3 and varied only the per-drive pending-read limit
through a benchmark adapter.

- The host had an AMD Ryzen 5 7600X, an RTX 3090 (24,576 MiB), 32,704,180,224
  bytes of physical RAM and one WD_BLACK SN850X 2000GB (`nvme0n1`). The container root was overlay
  and the memory cgroup limit was 31,395,414,016 bytes. A 1 GiB direct read and the
  physical counters identified the underlying drive.
- The software stack included Linux 6.8.0-138-generic x86_64, glibc 2.39,
  NVIDIA driver 595.99.02, Python 3.12.12, PyTorch 2.14.1+cu130 (CUDA 13.0),
  Transformers 5.20.1, PEFT 0.21.2,
  bitsandbytes 0.50.2, TRL 0.29.1, Accelerate 1.15.0, Datasets 5.1.0,
  Safetensors 0.8.0, NumPy 2.5.4, pytest 9.1.1 and Ruff 0.17.1. Every timed step
  process recorded 1860 MHz at both SM clock sampling points.
- The read probe used 41,104,179,200 bytes of allocated random data. Both arms read
  the same 24 ordered 400 MiB spans with four aligned ranges per span. Both allocated
  and touched two pinned 400 MiB arenas. Depth 1 used one arena and depth 2 could use
  both. The queue allowed four unfinished range tasks per pending span and at
  most four executing reads. The timer ran from submission through final
  completion. Each arm had an untimed one-span warm-up, full-span byte checks
  before timing and retained-slot byte checks after timing.
- The step probe used a synthetic 68.99B-parameter Llama shape with 80 layers, hidden
  size 8192 and vocabulary 32768. The BF16 checkpoint used seed 1234 and was
  converted to an NF4 disk cache with 36,388,412,723 bytes of files and
  36,388,511,744 bytes allocated. The training path used bfloat16, batch 1,
  sequence 256, two device buffers, pinned host staging, LoRA rank 8 on
  q/k/v/o, adapter seed 3, input seed 17 and
  PagedAdamW8bit at 1e-4. Each process built a fresh model and optimizer, ran
  one warm-up, then timed three forward, backward and optimizer steps with
  event instrumentation disabled.

## 2. Method and decision rules

Both input stores exceeded physical RAM. The probes checked `O_DIRECT` on
the descriptors and required physical reads within 80–120% of submitted bytes,
at least 4 GB of available host memory at arm boundaries and no foreign GPU client.
Both comparisons used fresh processes in alternating depth-1/depth-2 pairs
(1–2–2–1 blocks) with suspend, power and foreign-read checks. Five blocks
formed the pilot. The adaptive repeat calculation called for two blocks in
each comparison, so both completed with the five-block minimum and zero
additional blocks. An invalid arm voided its entire sequence.

The strict protocol required pre-arm reads and writes at or below 1 MiB/s
and timed writes at or below 1 MiB/s. After idle monitoring showed periodic
background writes, a separately frozen diagnostic retained the pre-arm read
limit and capped timed writes at both 0.2% of submitted bytes and 8 MiB/s.
The diagnostic could measure an effect under those conditions. Production
acceptance still required an independent pass of the strict protocol.

The read-rate rule required every depth-2/depth-1 ratio to reach 1.10 for a
pass and classified ratios all within 0.95–1.05 as no benefit. The step rule
required every depth-1/depth-2 speedup to reach 1.05 and the median speedup to
reach 1.10. The rule applied to per-arm medians and means, with equal host
staging, finite losses and passing byte and DMA checks. The allowed increase
in peak allocated VRAM was 2% or 64 MiB, whichever was larger.

## 3. Results

Each comparison completed 20 valid arms in ten pairs. Step times are per-arm
medians and means of three steps, for 60 timed steps in total.

| Pair | Depth 1 GB/s | Depth 2 GB/s | Rate ratio | Depth 1 median s | Depth 2 median s | Depth 1 mean s | Depth 2 mean s | Median-step speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.26145 | 7.31786 | 1.00777 | 9.49899 | 9.32520 | 9.54329 | 9.36355 | 1.01864 |
| 2 | 7.26463 | 7.33392 | 1.00954 | 9.57860 | 9.32497 | 9.60798 | 9.36804 | 1.02720 |
| 3 | 7.25687 | 7.32685 | 1.00964 | 9.53314 | 9.32051 | 9.57503 | 9.35661 | 1.02281 |
| 4 | 7.26767 | 7.32165 | 1.00743 | 9.56334 | 9.32584 | 9.59806 | 9.36778 | 1.02547 |
| 5 | 7.27327 | 7.33309 | 1.00822 | 9.58177 | 9.31587 | 9.60479 | 9.35575 | 1.02854 |
| 6 | 7.27535 | 7.31474 | 1.00541 | 9.56598 | 9.33406 | 9.60149 | 9.37316 | 1.02485 |
| 7 | 7.26348 | 7.32725 | 1.00878 | 9.55575 | 9.32614 | 9.57424 | 9.36913 | 1.02462 |
| 8 | 7.27319 | 7.32369 | 1.00694 | 9.55651 | 9.32325 | 9.59045 | 9.36679 | 1.02502 |
| 9 | 7.26150 | 7.32347 | 1.00853 | 9.58810 | 9.33657 | 9.60336 | 9.37995 | 1.02694 |
| 10 | 7.27156 | 7.32429 | 1.00725 | 9.57433 | 9.32652 | 9.60744 | 9.36021 | 1.02657 |

- **Reads.** Median paired improvement 0.800% (0.541–0.964%). Physical reads
  were 1.000000–1.000008 times submitted bytes. Writes were 0–99,328 bytes per arm.
- **Steps.** Median paired speedup 1.025243 (+2.524%), range 1.018636–1.028543.
  The mean-step rule also failed. Both depths peaked at 3.844141 GB of
  allocated VRAM and used 2,398,052,352 bytes of host staging. Losses were
  finite (11.43365002–12.03265762). Physical reads were 0.965935–0.994574
  times submitted bytes. Writes were 40,960–8,465,408 bytes per step counter
  window, with a maximum average of 0.833148 MiB/s.

## 4. Commands

The harnesses and command files live at the evidence commit. The accepted
read arms ran at `e10b7d9aeab0a9ffd86ee4df36a817e0464c0de2`. The accepted
step arms ran at `a3e3fcddad55cbf1e117e2a2191bd680411a9e92`.
The historical output paths identify separate sequences. For a new run, use
fresh output directories. The evidence preserves the admission sensor script
and the final preflight script in the Linux host's command files.

```bash
cd /workspace/c2
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
/venv/main/bin/python benchmarks/harness/synth_checkpoint.py --shape llama-70b --vocab 32768 --seed 1234 --out /workspace/c2-weights
/venv/main/bin/python benchmarks/harness/c2_linux_campaign.py --prepare-model --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --out /workspace/c2-cache-build.json
/venv/main/bin/python benchmarks/harness/c2_linux_depth_probe.py --fixture /workspace/c2-fixture.bin --prepare --out /workspace/c2-fixture-prepare.json
# Admission child, run under the checks in sensor-diagnostic-command.txt.
/venv/main/bin/python benchmarks/harness/c2_linux_depth_probe.py --fixture /workspace/c2-fixture.bin --admit-only --diagnostic-writes --out /workspace/c2-admission-e2.json
# Read sequence at the first measurement revision.
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind primitive --fixture /workspace/c2-fixture.bin --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-primitive-e1
# First step attempt at the same revision.
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind step --fixture /workspace/c2-fixture.bin --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-step-e1
# Second step attempt at 193e433c2c0e851ee3f517f1fdda3f56696bad38.
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind step --fixture /workspace/c2-fixture.bin --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-step-e2
# Final retry at the step measurement revision. The preflight script checks
# cache identity against the retained second step sequence.
/venv/main/bin/python /dev/shm/prequiet_e3.py
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind step --fixture /workspace/c2-fixture.bin --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-step-e3
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /venv/main/bin/python -m pytest tests/test_c2_*.py -q --no-cov --tb=short > /workspace/c2-final-tests-e3.log 2>&1
```

The Windows byte and queue check used the preparation harness and the CPU
interpreter selected as `$py`.

```powershell
$env:PYTHONPATH = "$PWD\src"
& $py benchmarks/harness/c2_depth_probe.py --cpu-check --out cpu-new-sequence.json
```

## 5. Restart history and limitations

- The CPU-only preparation used Windows build 26300, Python 3.10.0 and
  PyTorch 2.13.0+cpu. Its environment also recorded Transformers 5.17.0,
  PEFT 0.20.0, TRL 0.29.1, Safetensors 0.8.0, NumPy 1.26.4, pytest 9.1.1
  and Ruff 0.16.4. The synthetic read fixture passed 38/38 full-span byte
  checks, with at most 4 or 8 pending ranges at the two depths. It yielded
  no Windows timing verdict.
- Earlier Linux host attempts ended at RAID admission, idle-write checks,
  container startup or package installation. The resumed campaign ran on a
  newly admitted host. Its first strict preflight failed the idle-read check.
  The two void read sequences on the earlier host contributed no pairs.
- The first diagnostic step sequence stopped before timing because its own
  prefetch triggered the idle check. The second moved the idle check before
  model construction, then stopped at child 12 after 1,725,698,560 bytes of
  unattributed reads exceeded the 1 GB limit. The final retry required twelve
  consecutive five-second read-idle windows and matching cache metadata.
  Both earlier step sequences were excluded in full. The restart history
  contains no training-checkpoint resume measurement.
- Focused Linux validation passed 73 tests, including delayed CUDA copies at
  both depths. Expanded validation had 277 passed, 2 failed and 1 skipped.
  The failures concerned tiny pinned-arena alignment and foreign-read attribution
  for an unreaped child. The earlier full-suite attempts did not complete cleanly.
- One physical drive under bounded writes cannot establish an isolated SSD
  ceiling or transfer the result to Windows or striped drives. The read probe
  lacked the exact outer counter bracket duration. Cache identity checks used
  size and mtime. Synthetic losses only check finiteness here. The evidence
  archive retains the raw journals, arm data and discarded runs.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/4aee1bf556141d91b5ab3de6eef0430bf12d5073/benchmarks
