# Supplemental 10 questions

The user approved these ten questions and their target sections in chat on November 23, 2026. Their wording and target records are unchanged from the draft.

The ranker was frozen at `f40be434fb6b4dbd1b1688cf12c633de5f1f6ef5` before the questions were drafted. Authorship and eligibility are preserved in the fixture and `supplemental10-lock.json`: the implementation agent generated this set, so it does not establish outside-tuning independence.

The approved fixture and its hash were frozen in commit `f1f507ad` before the first scoring run. The run returned exit code 1: Hit@5 stayed at 7/10 and MRR@10 decreased from 0.4750 to 0.4533. Full outputs are in `supplemental10-results.json`, with the execution record in `supplemental10-execution.json`. The candidate, wording and labels were not changed after scoring. This set is supplemental evidence and does not replace the confirmed fresh-set requirement.

| ID | Question | Approved target sections |
| --- | --- | --- |
| R01 | Can I do local adapter training on my M2 laptop, or do I need to rent a GPU? | `backends-and-ops.md` → Apple Silicon (MLX Backend) |
| R02 | The machine rebooted halfway through a run. How do I continue training from the saved state? | `backends-and-ops.md` → Resume Training |
| R03 | Can Soup try different batch sizes and pick the largest one my card can handle? | `performance-and-quantization.md` → Live CUDA Batch-Size Probe; `performance-and-quantization.md` → OOM-probe auto batch size |
| R04 | I have chosen and rejected answers but cannot fit a second model. Which preference methods avoid a reference model? | `training.md` → ORPO Training (No Reference Model); `training.md` → SimPO Training (Simple Preference) |
| R05 | My training dataset repeats the same instruction with small wording changes. Can I filter those rows by embedding similarity? | `data.md` → Semantic dedup (`soup data dedup --semantic`) |
| R06 | I need a checkpoint right now, but I want the running job to keep training afterwards. | `backends-and-ops.md` → Checkpoint-Now Trigger File |
| R07 | Does the merge command check whether the licenses of my adapters are compatible? | `adapters-and-governance.md` → License-Conflict Matrix at Merge |
| R08 | Can I stop a run automatically if its evaluation falls below the model I started with? | `evaluation.md` → Eval-Gated Training |
| R09 | The server is already running. Can I switch to another LoRA adapter without restarting it? | `serving-and-export.md` → Dynamic LoRA Hot-Swap |
| R10 | What does soup data dedup --semantic --threshold 0.85 change compared with default deduplication? | `data.md` → Semantic dedup (`soup data dedup --semantic`); `data.md` → It is not a paraphrase detector — and why the default is 0.8 |


Fixture: `tests/fixtures/docs_search/agent_draft_review.json`. SHA-256: `4318f4eb6a1f36d62e766b7a45cb6643b63b00efe2ab24a8dfbb07789fc50aa4`.
