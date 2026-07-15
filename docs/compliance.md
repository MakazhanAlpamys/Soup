# Compliance & governance quickstart

[← Back to the docs index](README.md)

Soup ships an end-to-end compliance workflow: start from a regulation-shaped
config, train with provenance capture, sign + attest the artifact, keep an audit
trail, and move it across an air gap — then publish a documented model card and
gate future changes in CI.

- **Pre-wired configs:** `soup init --template hipaa|soc2|eu-ai-act|sr-11-7`
- **Provenance:** `soup train --repro-receipt` · `soup bom emit` · `soup attest emit`
- **Integrity:** `soup adapters sign` / `verify` · `soup adapters scan`
- **Audit trail:** the HIPAA/SOC2 audit log is on by default (`soup audit-log tail`)
- **Air gap:** `soup airgap-bundle`
- **Publish:** `soup card <registry-id>` → `MODELCARD.md` (or `soup push --card`)
- **CI gate:** `soup ci init` → a PR workflow that runs validate → expect → ship

---

## 1. Start from a compliance template

Each template is a normal training config plus header comments listing the exact
compliance commands to run around it. Pick the regime you operate under:

```bash
soup init --template hipaa      # Protected Health Information
soup init --template soc2       # SOC 2 Trust Services Criteria
soup init --template eu-ai-act  # EU AI Act Annex XI/XII
soup init --template sr-11-7    # SR 11-7 Model Risk Management
```

The compliance controls are Soup **CLI flags/commands, not config keys** — the
template header documents which ones apply. The steps below are the common path.

## 2. Clean the data before training

```bash
soup data pii ./data/train.jsonl            # flag emails / phones / SSNs / MRNs
soup data decontaminate ./data/train.jsonl  # drop public-benchmark overlap
```

## 3. Train with a reproducibility receipt (+ Annex XI / energy for the EU)

```bash
# SR 11-7 / SOC 2 / HIPAA: capture seeds, kernels, GPU, OS
soup train --config soup.yaml --repro-receipt receipt.json

# EU AI Act: auto-generate the Annex XI/XII documentation + measure energy
soup train --config soup.yaml \
    --annex-xi annex_xi.md \
    --track-energy --energy-country DEU --energy-out energy.json
```

The audit log records every command automatically:

```bash
soup audit-log tail          # review the trail
soup audit-log rotate        # force a rotation pass
```

## 4. Register the run, then emit BOM + attestation

```bash
soup registry push --run-id <run-id> --name my-model --tag v1

soup bom emit --name my-model --base-sha <hex> --config-sha <hex> \
    --energy energy.json --format both       # CycloneDX + SPDX
soup attest emit --stage train --subject my-model --sha <hex> \
    --sign ed25519 --key key.pem             # in-toto + SLSA-3
```

## 5. Sign, scan, and verify the artifact

```bash
soup adapters scan ./output                              # weight-space backdoor scan
soup adapters sign ./output --backend ed25519 --generate-key key.pem
soup adapters verify ./output --strict --public-key key.pub.pem
```

## 6. Air-gap transfer (optional)

```bash
soup airgap-bundle --model ./output --output my-model.tar --repro-receipt receipt.json
```

## 7. Generate a documented model card

Turn the registry entry into a provenance-rich `MODELCARD.md` — base model,
training config, eval scorecard, config/data hashes, lineage, and every
registered artifact:

```bash
soup card my-model:v1 -o MODELCARD.md
# or, when uploading to the Hub, override the auto-generated card:
soup push --model ./output --repo you/my-model --card my-model:v1
```

## 8. Gate future changes in CI

Write a GitHub Actions workflow that blocks a PR unless the data validates, the
expectations suite passes, and the SHIP verdict is green:

```bash
soup ci init --data data/train.jsonl --suite expectations.yaml --evidence ship_evidence.json
# writes .github/workflows/soup-gate.yml
```

The generated job runs, in order:

```
soup data validate <data>       # dataset format compliance
soup expect <data> <suite>      # PII / token-length / refusal / judge expectations
soup ship --evidence <ev.json>  # SHIP / DON'T-SHIP (exit 2 blocks the merge)
```

A minimal `expectations.yaml` for the second step:

```yaml
expectations:
  - name: expect_no_pii
  - name: expect_token_length_between
    min_tokens: 1
    max_tokens: 512
```

Supported names: `expect_no_pii`, `expect_token_length_between`,
`expect_no_refusal_pattern`, `expect_chosen_preferred_over_rejected_by_judge`.

Every path is shell-quoted and validated to stay under the repo root, so the
rendered workflow is injection-safe. Edit the paths to match your repo.

---

See also: [Adapters, registry & governance](adapters-and-governance.md) for the
full supply-chain command set, and [Evaluation & probes](evaluation.md) for the
`soup ship` verdict engine.
