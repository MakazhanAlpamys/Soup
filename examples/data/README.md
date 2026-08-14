# Sample data

**These are format examples and test fixtures — not training data.** Each file
is a handful of rows, there to show what a format looks like and to give the
parser and the test suite something to chew on. Nothing useful will train on
this volume. Treat them as a shape to copy, not a starting corpus.

| File | Format | Rows | Demonstrates |
|------|--------|-----:|--------------|
| `alpaca_tiny.jsonl` | alpaca | 10 | `instruction` / `input` / `output` instruction pairs |
| `chat_preferences.jsonl` | dpo | 5 | `prompt` / `chosen` / `rejected`, messages as chat turns |
| `dpo_sample.jsonl` | dpo | 8 | Same shape, longer answers — used by the DPO example |
| `reasoning_math.jsonl` | alpaca | 5 | Step-by-step worked solutions for GRPO/reasoning |

`soup data inspect <file>` prints the detected format and stats for any of them.

## Getting real data

```bash
soup data search "instruction tuning"      # find datasets on the HF Hub
soup data preview <dataset-id>             # splits + features, before downloading
soup data download <dataset-id> -o out.jsonl
soup data demo                             # list bundled demo fixtures (same files as these)
```

## How much do you actually need

- **A format or a style** (JSON output, a house tone, a response shape): hundreds of rows.
- **A task the model already half-knows** (your flavour of summarisation, classification): thousands.
- **New facts** the model has never seen: fine-tuning is usually the wrong tool — retrieval (RAG)
  puts the facts in the prompt instead, and stays correct when they change.

`soup advise <data> --goal "..."` reads your data and says which of these you're in.

More on formats, conversion, and the data pipeline: [docs/data.md](../../docs/data.md).
