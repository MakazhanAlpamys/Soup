# Issue difficulty

To help you pick an issue that fits your time and experience, every triaged issue carries one
label from `difficulty:1` to `difficulty:10`.

It estimates how much work and risk the fix carries. It is **not** a statement about how
important the bug is.

A difficulty label also means a maintainer has looked at the issue and confirmed it. An issue
with no difficulty label is not triaged yet. Duplicates, trackers, questions and issues closed
as not planned carry none.

Issues closed before this system existed were rated afterwards, from the pull request that
closed them, so those older labels are approximate. A label is an estimate: when a label and
the table below disagree, the table is the definition and the label gets fixed.

## What the numbers mean

| Level | What it looks like | Example |
|---|---|---|
| 1 | Wording, a typo or a one-line fix. No new logic. | #1611, #1642 |
| 2 | One non-test file, under about 15 changed lines; the issue spells the fix out; one test. | #1641 |
| 3 | One non-test file, up to about 50 changed lines; tests run on CPU without torch. | #1598 |
| 4 | Two to four source files in one subsystem, following a pattern that already exists; tests run on CPU without torch. | #1596 |
| 5 | Five or more source files, or a trainer, or the config schema; tests need torch. | #1345 |
| 6 | Several tasks, trainers or backends at once, or non-trivial numerics; each affected one needs its own regression test. | #805 |
| 7 | A new mechanism or a deep change to a core path (trainer internals, layer streaming, the data pipeline); proving it takes a measurement. | #1425 |
| 8 | The cause is not known yet or is hard to reproduce, or the result must be bit-exact. | #342 |
| 9 | A new subsystem, or support for several model families at once. | #265 |
| 10 | Research-grade: hand-written kernels or backward passes with a numerical gate. Rare; most hard work stops at 8 or 9. | #792 |

## Special hardware

If the fix cannot be verified without special hardware or a paid service (a large GPU, several
GPUs, a paid cloud), add 3 levels to what the work itself would get, up to 10. A one-module
change that can only be checked on an H100 is level 6, not level 9. Say so in your claim, as
[CONTRIBUTING.md](../CONTRIBUTING.md) asks.

Disagree with a number? Say so in the issue thread and give the reason; we will look again.

## Opening an issue

The issue form has an optional "Suggested difficulty" field. Your guess helps, and a maintainer
sets the final label. Please search first, and keep to one defect per issue.
