# Questions for user review

Authorship: AI-drafted by the Codex implementation agent who developed and tuned the ranker. These questions were not supplied by someone outside the tuning work. User review is pending.

Candidate frozen before drafting: `f40be434fb6b4dbd1b1688cf12c633de5f1f6ef5`. The ten questions and provisional target sections below have not been scored. Source inspection confirmed the sections exist; no rankings were run.

This is a draft for review, not independent fresh-set evidence. The set contains five reworded development intents, four additional intents and one technical control. Multiple plausible target sections are retained where the question permits them.

Please review whether each question is natural and whether the listed sections answer it. Keep edits and any changed target labels explicit. Approval of the text will not change the recorded AI authorship or make this an outside-tuning set.

| ID | Question | Provisional target sections |
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

Unscored fixture: `tests/fixtures/docs_search/agent_draft_review.json`. SHA-256: `e954c6d4a69fe26cfe458a39a0390ccd80f9684b52929618d56de5cf13f1a2c2`.

No Hit@5 gain or other result is claimed. The confirmed fresh-query requirement remains pending unless an eligible new set is provided or the maintainer explicitly changes the evaluation plan.
