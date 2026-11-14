# Deviations from the frozen protocol

`PROTOCOL.md` is frozen and has not been edited (its committed text has the sha256
`bcb6ad4275aba456c1527ded0bb853f7ba1110a974620beaa171cc06249ab164`). Everything below is where the
harness or the run differs from what that text says, why, and what it does to the numbers. D-1 to D-3
were found on the first rented run of part 1, before any value entered a gate; D-4 concerns how parts 2
and 3 were run. None moves a threshold.

## D-1 The streaming arms pin their store explicitly (`stream_pin: true`)

**Protocol (section 2, S1):** `stream_pin` automatic. **Run:** every streaming arm except S4 sets
`training.stream_pin: true` in its `soup.yaml`; S4 (the leave-one-out arm for pinning) still sets
`false`. SB inherits the `true` unless the SB rule selected S4's change.

**Why.** Determinism. With the key set, the pin is a stated property of every run, and a failed page-lock
makes Soup refuse instead of falling back to pageable memory, so no arm can turn pageable silently. It
changes nothing on a box where Soup pins automatically, and both rented hosts are such boxes: the
automatic decision at the Soup commit that ran (`pinned_limit_bytes=None` in `stream_setup.py`, so
`decide_pinning` returns `pinned=True`) attempts to pin and falls back, loudly, only if the allocation
fails, and nothing under `src/` reads `RLIMIT_MEMLOCK`. The container's `ulimit -l` was 8,192 KB on both
hosts, and Soup's banner said `(pinned)` with `16.11 GB page-locked` on both: on the first host with the key
unset (S0, S1, S2 and S3 all say `(pinned)`), on the second with it set. The warning in `setup.sh` about
`ulimit -l` below 32 GB ("may not be pinned") is the harness's own text, not Soup's behaviour, and its
premise did not hold on either host.

**The first rental's slowdown was the host, not pinning.** The first rental of part 1 (aborted during S4)
ran the same image, the same Soup commit and the same libraries on a different host, and was about 2x slower
for every arm, DeepSpeed included:

| | first host | second host (the numbers of the record) |
|---|---|---|
| CPU | EPYC 7402P, 48 CPUs, 157 GB RAM | EPYC 9684X, 384 CPUs, 1,511 GB RAM |
| driver, PCIe link at idle | 570.207, gen 4 x16 | 615.71.09, gen 5 x16 |
| S1, repeat 1 | 27.29 tok/s, 333 s wall | 61.57 tok/s (median of 3: 61.99), 156 s wall |
| S2 / S3, repeat 1 | 27.39 / 27.32 tok/s | 61.14 / 62.52 tok/s |
| untimed warm-up S1 | 27.18 tok/s (0.45x of the second host) | 59.88 tok/s |
| untimed warm-up D1 (DeepSpeed, no Soup) | 19.80 tok/s (0.50x of the second host) | 39.56 tok/s |

The first host's values come from the aborted run (one repeat each) and enter no gate, no ratio and no SB
derivation. Which property of the host moved them (CPU, host memory, PCIe generation, driver) was not
isolated. For scale, STEP 3 measured 63.52 tok/s for this arm on a box whose `ulimit -l` was 63 GB.

**Effect on the record.** S1 is "current main with the pin requested explicitly", which is the same run as
the automatic choice. The run json of every Soup arm records whether the store was pinned or pageable
(parsed from Soup's own banner). `setup.sh` keeps recording `ulimit -l` in `hardware.txt`.
S0's yaml drops the key (it did not exist at `2c9a078`; the json lists it).

**Time estimate checked.** The estimate for a streaming run was 3.2 min (2.2 train + 1.0 start-up). Measured
on the second host: 155 s of whole-process wall time for S1 (132 s of `train_runtime`).

## D-2 Arm S0 is not measured

**Protocol (section 2):** S0 = Soup at `2c9a078`, same yaml as S1 where the keys exist, in a separate venv.
**Run:** S0 is not in the plan (removed from `SHAPE_A_ARMS`; `job_part1.sh` sets `SKIP_S0=1`, so the S0 venv is
not built either).

**Why.** On the rented box the S0 venv's `soup train` fails before training starts:

```
RuntimeError: non-layer weights left unmaterialised after tying: ['model.embed_tokens.weight', 'lm_head.weight']
```

**Effect on the record.** G1's secondary value, S0 / D1 ("the old code on this box"), is not reported;
`gates.json` shows it as not run. The S0 to S1 difference, the sum of the code-level changes since STEP 3
that have no switch (protocol section 2), is not measured, and the record says so. G1 itself (S1 / D1) and
the ablation (S2..S7 against S1) do not depend on S0. The arm's code stays in `soup_arm.py` and a plan that
holds an S0 slot still refuses to start without the S0 executable.

## D-3 The Hugging Face Trainer owns the scheduler in the DeepSpeed arms

**Protocol (section 1, Optimizer):** constant schedule with no warm-up in every arm; "in DeepSpeed arms where
DeepSpeed must own the optimizer and scheduler (torch strict-zip issue seen in STEP 3), its AdamW / WarmupLR
with the same values". **Run:** in D1..D4 and the two prefetch variants DeepSpeed owns the optimizer (the json
has an `optimizer` block: AdamW, lr 2e-4, weight decay 0) and the Trainer owns the scheduler
(`lr_scheduler_type="constant"`, `warmup_steps=0`); the json has no `scheduler` block.

**Why.** With the torch of the rented box the strict-zip failure that made the earlier record hand the
scheduler to DeepSpeed does not occur, and a DeepSpeed `WarmupLR` with `warmup_num_steps: 0` is refused by the
Trainer's own config check (`warmup_num_steps=0 vs hf warmup_steps=1`, seen on the first rented run). Giving
the scheduler to the Trainer keeps the one thing the protocol insists on: the same constant 2e-4 in every arm.

**Effect on the record.** None on throughput. The first optimizer step of a D arm now runs at lr 2e-4, as in
every other arm (under `WarmupLR` it would have run at lr 0, which would have shifted the first losses
slightly).

## D-4 Parts 2 and 3 ran in one job on one instance

**Protocol / README:** the series runs as three rentals, one instance per part. **Run:** parts 2 and 3
ran back to back in one job, `job_part23.sh` (one set-up, one pull), on instance 59572480: part 2 from
13:11:56Z to 13:36:56Z (20 of 20 runs ok), part 3 from 13:36:56Z to 14:14:27Z (15 of 15 runs ok). Both parts
read part 1's run jsons from the one shipped payload and derived SB and D-best from them as before.

**Why.** One set-up and one pull instead of two. The instance is the same machine as part 1's (the hardware
record differs only in host name, time, free memory and disk fill).

**The first part-2 attempt.** An earlier rental, instance 59565501 (11:52:35Z to 12:20:08Z), ran part 2 to
the end: its job finished with exit code 0 at 12:18:21Z and all 20 runs are ok. Its results were not pulled
then: the stopped instance needed a restart for the pull, requested at 12:32:53Z, and the files were
pulled only at 14:21:10Z, after a second restart request at 14:19:48Z. Part 2 was therefore run again inside
instance 59572480, which finished before the first attempt's files arrived. The rerun is the record's value for
part 2; the first attempt is reported next to it as a replicate and enters no gate. Both give the same
verdicts.

**Effect on the record.** (1) Parts 2 and 3 share one machine and one session, so the README's limit "every
part is a different rented machine" does not hold between them; part 1 was the same machine on another
rental. (2) Part 2's first timed run (SB at Shape B, repeat 1) was the first run on a cold instance, as the
README says of the first timed runs of parts 2 and 3; it was slower than its second repeat in both attempts
(section 11 of the record). (3) Part 3 started on an instance that had just run part 2.
