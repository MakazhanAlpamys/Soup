# Friend-supplied 30: target mapping before scoring

User attestation: these 30 questions and intended targets were supplied by a friend outside the project/tuning work. Individual attribution not supplied; attestation not independently verified.

Not specified; no claim of absence of assistance.

All 30 received question texts and target descriptions are preserved in `tests/fixtures/docs_search/friend30.json`. No question is selected based on a ranking result. The ranker remains frozen at `f40be434`. The fixture uses the same 17-guide, 702-section snapshot as the candidate and development comparisons. The original-28 pinned-docs v1 comparison remains separate.

Coverage is not a representative random sample: adapters-and-governance: 23/30 questions, autodistill: 4/30 questions, backends-and-ops: 3/30 questions. There are no normalized literal repeats against the original 28, maintainer development ten, supplemental ten, or previously inspected 8+6 questions. Shared intents are allowed and are not claimed to be novel.

The supplied subtopic descriptions map to existing section ownership; code comments and prose labels inside a section do not create new chunks. FR13 retains both supplied targets, scanner and license-conflict gate. No alternate target was inferred from a ranking.

| ID | Supplied target description | Frozen document and owning section |
| --- | --- | --- |
| FR01 | Data Flywheel → canary quality verdict / --canary-max-wait | `adapters-and-governance.md` → Adapters, Registry & Governance > Data Flywheel (`soup loop`) (ID `3`) |
| FR02 | Bill of Materials → Attaching energy + CO₂ | `adapters-and-governance.md` → Adapters, Registry & Governance > Bill of Materials (`soup bom emit`) > Attaching energy + CO₂ (`--energy`) (ID `12`) |
| FR03 | Adapter Management → merge strategies / rank check | `adapters-and-governance.md` → Adapters, Registry & Governance > Adapter Management (git for LoRA) (ID `9`) |
| FR04 | Adapter Management → How --weights is applied | `adapters-and-governance.md` → Adapters, Registry & Governance > Adapter Management (git for LoRA) (ID `9`) |
| FR05 | Adapter Sign + Verify | `adapters-and-governance.md` → Adapters, Registry & Governance > Adapter Sign + Verify (`soup adapters sign` / `verify`) (ID `20`) |
| FR06 | Run-vs-Config Audit → MLX peak_lr / unknown is never ok | `adapters-and-governance.md` → Adapters, Registry & Governance > Run-vs-Config Audit (`soup adapters audit`) (ID `18`) |
| FR07 | Knowledge Editing → soup edit set | `adapters-and-governance.md` → Adapters, Registry & Governance > Knowledge Editing (`soup edit set`, ROME / MEMIT / AlphaEdit) (ID `5`) |
| FR08 | GRACE Codebook — Lifelong Knowledge Edits | `adapters-and-governance.md` → Adapters, Registry & Governance > GRACE Codebook — Lifelong Knowledge Edits (ID `7`) |
| FR09 | Activation Steering → soup serve --steer | `adapters-and-governance.md` → Adapters, Registry & Governance > Activation Steering (`soup steer`) (ID `6`) |
| FR10 | Soup Cans → Credentials / redaction | `adapters-and-governance.md` → Adapters, Registry & Governance > Soup Cans (Shareable Recipes) (ID `10`) |
| FR11 | Model Registry & Lineage → Retries | `adapters-and-governance.md` → Adapters, Registry & Governance > Model Registry & Lineage (ID `8`) |
| FR12 | Adapter Lifecycle → soup adapters bisect | `adapters-and-governance.md` → Adapters, Registry & Governance > Adapter Lifecycle (`soup adapters {merge,pr,bisect}`, `soup lock`) (ID `2`) |
| FR13 | Adapter Backdoor Scanner / License-Conflict Matrix backdoor gate | `adapters-and-governance.md` → Adapters, Registry & Governance > Adapter Backdoor Scanner (`soup adapters scan`) (ID `19`); `adapters-and-governance.md` → Adapters, Registry & Governance > License-Conflict Matrix at Merge (ID `24`) |
| FR14 | EU AI Act Annex XI/XII Auto-Doc | `adapters-and-governance.md` → Adapters, Registry & Governance > EU AI Act Annex XI/XII Auto-Doc (`soup train --annex-xi`) (ID `14`) |
| FR15 | Audit Log → SOUP_AUDIT_LOG_PATH / --no-audit-log | `adapters-and-governance.md` → Adapters, Registry & Governance > Audit Log (`soup audit-log`) (ID `16`) |
| FR16 | Airgap Bundle | `adapters-and-governance.md` → Adapters, Registry & Governance > Airgap Bundle (`soup airgap-bundle`) (ID `25`) |
| FR17 | Data Flywheel → --min-new-pairs / data cursor | `adapters-and-governance.md` → Adapters, Registry & Governance > Data Flywheel (`soup loop`) (ID `3`) |
| FR18 | Data Flywheel → Learn from feedback | `adapters-and-governance.md` → Adapters, Registry & Governance > Data Flywheel (`soup loop`) (ID `3`) |
| FR19 | Data Flywheel → refused deploy backoff / pending_deploy | `adapters-and-governance.md` → Adapters, Registry & Governance > Data Flywheel (`soup loop`) (ID `3`) |
| FR20 | AutoDistill → Transactional publication, resume, corruption | `autodistill.md` → AutoDistill artifact contract (Milestone A) > 8. Transactional publication, resume, and corruption (ID `44`) |
| FR21 | AutoDistill → Missing-probability policy | `autodistill.md` → AutoDistill artifact contract (Milestone A) > 5. Missing-probability policy (ID `36`) |
| FR22 | AutoDistill → Plan-only estimate | `autodistill.md` → AutoDistill artifact contract (Milestone A) > 9. Plan-only estimate (ID `45`) |
| FR23 | AutoDistill → Consumption and replay | `autodistill.md` → AutoDistill artifact contract (Milestone A) > 7. State machines > 7.3 Consumption and replay (ID `42`) |
| FR24 | Backends → Apple Silicon (MLX Backend) | `backends-and-ops.md` → Backends, Platform & Ops > Apple Silicon (MLX Backend) (ID `53`) |
| FR25 | Adapter Lifecycle → soup lock / drift exits 3 | `adapters-and-governance.md` → Adapters, Registry & Governance > Adapter Lifecycle (`soup adapters {merge,pr,bisect}`, `soup lock`) (ID `2`) |
| FR26 | Backends → Terraform-Style Plan & Apply | `backends-and-ops.md` → Backends, Platform & Ops > Terraform-Style Plan & Apply (`soup plan` / `soup apply`) (ID `116`) |
| FR27 | Namespace Pinning (Anti-AI-Jacking) | `adapters-and-governance.md` → Adapters, Registry & Governance > Namespace Pinning (Anti-AI-Jacking) (ID `23`) |
| FR28 | Backends → Autopilot (Zero-Config) | `backends-and-ops.md` → Backends, Platform & Ops > Autopilot (Zero-Config) (ID `52`) |
| FR29 | Pre-Flight Adapter Health Audit → inactivity reasons | `adapters-and-governance.md` → Adapters, Registry & Governance > Pre-Flight Adapter Health Audit (`soup adapters check`) (ID `22`) |
| FR30 | License-Conflict Matrix at Merge | `adapters-and-governance.md` → Adapters, Registry & Governance > License-Conflict Matrix at Merge (ID `24`) |

FR25 keeps the supplied Adapter Lifecycle target. That pinned section contains a code comment saying exit 2 and prose saying drift exits 3. This evaluation measures retrieval of the supplied target; it does not resolve that documentation inconsistency. No query or target was changed.
