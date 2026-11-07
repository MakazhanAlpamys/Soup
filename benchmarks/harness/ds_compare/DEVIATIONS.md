# Deviations from the frozen protocol

`PROTOCOL.md` is frozen and has not been edited (its committed text has the sha256
`bcb6ad4275aba456c1527ded0bb853f7ba1110a974620beaa171cc06249ab164`). Everything below is where the
harness or the run differs from what that text says, why, and what it does to the numbers. All three
were found on the first rented run of part 1, before any value entered a gate; none moves a threshold.

## D-1 The streaming arms pin their store explicitly (`stream_pin: true`)

**Protocol (section 2, S1):** `stream_pin` automatic. **Run:** every streaming arm except S4 sets
`training.stream_pin: true` in its `soup.yaml`; S4 (the leave-one-out arm for pinning) still sets
`false`. SB inherits the `true` unless the SB rule selected S4's change.

**Why.** On the rented container `RLIMIT_MEMLOCK` is 8 MB, soft and hard, and the job cannot raise it. Soup's
automatic pin decision (`decide_pinning` in `src/soup_cli/utils/layer_stream.py`) refused to pin, so the first
rented run executed every streaming arm from pageable memory:

| arm (rep 1, unpinned, informational) | tok/s | wall | peak VRAM (nvidia-smi) |
|---|---|---|---|
| S1 | 27.29 | 333 s | 2,922 MiB |
| S2 | 27.39 | 333 s | 3,304 MiB |
| S3 | 27.32 | 333 s | 3,066 MiB |

These values come from the aborted run, one repeat each, with the store unpinned. They enter no gate, no
ratio and no SB derivation. For scale: the earlier record (STEP 3) measured 63.52 tok/s for this arm on a box
whose `ulimit -l` was 63 GB, and its store was pinned.

CUDA pinning works on the rented box regardless of that limit: a measured allocation pinned 2 GiB in 1.97 s.
soup-cli at `2c9a078` (the S0 revision) also pinned on this same rented box; its banner said "(pinned)". So
the automatic refusal is newer behaviour of the current `main`, not a property of the box that every
revision shares. Forcing the pin gives the streaming arms what the earlier record's box gave them, and what
the DeepSpeed arms already have (`pin_memory: true` on every offload in their json).

**Effect on the record.** S1 is "current main with the pin forced on", not "current main with its automatic
choice". On a box with a large memlock limit the two are the same run. With `stream_pin: true` Soup refuses on
a CUDA target rather than falling back to pageable memory, so a forced run cannot turn pageable silently; the
run json of every Soup arm also records whether the store was pinned or pageable (parsed from Soup's own
banner). `setup.sh` keeps recording `ulimit -l` in `hardware.txt`.
S0's yaml drops the key (it did not exist at `2c9a078`; the json lists it).

**Time estimate checked.** The estimate for a streaming run is 3.2 min (2.2 train + 1.0 start-up). Expected
wall time pinned: STEP 3's 129 s train time, plus the 33 s of start-up and teardown seen in the unpinned run
(333 s wall less 300 s train, 8,192 tokens at 27.29 tok/s), plus about 16 s to pin the 16 GB store at the
measured rate: about 180 s. The estimate (192 s) stands; it is replaced by measured wall times as runs finish.

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
