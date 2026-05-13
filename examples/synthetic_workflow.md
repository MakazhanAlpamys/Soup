# Synthetic data workflow (end-to-end)

Generate, filter, score, and train on synthetic data — all with `soup`. Pairs
with the bundled `synthetic_workflow.yaml` recipe.

## 1. Generate

Spin up a local Ollama model and ask it for 200 instruction/response pairs
around a topic:

```bash
soup data generate \
  --provider ollama \
  --model llama3.2:3b \
  --topic "Python error handling" \
  --count 200 \
  --output ./synth_raw.jsonl
```

For Anthropic / vLLM / server providers, swap `--provider` and follow the
`soup data generate --help` matrix.

## 2. Filter for quality

Drop low-perplexity / low-coherence rows:

```bash
soup data filter \
  --input ./synth_raw.jsonl \
  --output ./synth_filtered.jsonl \
  --min-coherence 0.5
```

## 3. Score for safety + diversity

Run the v0.47.0 quality moat to fingerprint PII / toxicity / language /
educational value, and decontaminate against your downstream evals:

```bash
soup data score --input ./synth_filtered.jsonl --output ./synth_scored.jsonl
soup data decontaminate \
  --input ./synth_scored.jsonl \
  --output ./synth_clean.jsonl \
  --benchmarks mmlu,gsm8k
```

## 4. Train

Point `soup train` at `synthetic_workflow.yaml`:

```bash
soup train --config examples/synthetic_workflow.yaml --yes
```

That recipe references `./synth_clean.jsonl`, picks `TinyLlama-1.1B-Chat`
as the base, and runs an SFT job with LoRA r=8.

## 5. Watch progress live (v0.53.9)

In another terminal:

```bash
soup ui --public --no-browser
```

Scan the printed QR code from your phone to monitor the loss curve and
live SSE training stream while the job runs.

---

This is a thin walkthrough — for deeper coverage see
[examples/README.md](README.md) and [examples/configs/](configs/).
