# One-step CPU fixture for Hugging Face Hub QA

`hf_hub_cpu_fixture.py` prepares an opt-in, one-step SFT run for the `soup train --push-as` leg of [Issue #74](https://github.com/MakazhanAlpamys/Soup/issues/74).

## Scope and evidence boundary

The fixture supplies a public tiny causal-LM, one valid ChatML row, batch size one, one epoch, and `save_steps: 1`. Its training child hides CUDA and ROCm devices, so a successful `--train` invocation proves only that Soup can create a local `checkpoint-1` and a loadable adapter using this CPU fixture.

It is not a Hugging Face upload result, a model-quality result, or evidence that a private Space can build or serve. Those require the live checks in Issue #74.

The fixture refuses an MPS-capable macOS host rather than labelling an MPS execution as CPU: `soup train` has no explicit CPU device selector.

## Local-only preparation and CPU run

Install the repository with its training dependencies, then use a new output path:

```bash
pip install -e ".[train]"
python benchmarks/harness/hf_hub_cpu_fixture.py \
  --output-dir ./hf-hub-cpu-fixture \
  --train --json
```

The path must not already exist. The harness writes only within that user-named directory:

- `train.jsonl` — one non-sensitive local training row;
- `soup.yaml` — the exact tiny-model CPU fixture config;
- `output/` — the local adapter and `checkpoint-1`;
- `cpu-train-command.txt` and `cpu-fixture-report.json` — commands and local result metadata.

Without `--train`, it prepares those inputs only. It never reads, writes, or stores an HF token in either local mode.

## Live checkpoint-branch verification

This is intentionally an explicit, credentialed operation. Before running it:

1. Create a **new private model repository** on Hugging Face yourself.
2. Configure a write token through `HF_TOKEN` or `huggingface-cli login`; never put the token in a shell history, fixture file, or issue comment.
3. Pass the existing repo name below. The harness checks that it is private before training because Soup's current auto-push callback calls `create_repo(..., private=False, exist_ok=True)`. Pre-creating the private repository preserves its visibility.

```bash
python benchmarks/harness/hf_hub_cpu_fixture.py \
  --output-dir ./hf-hub-cpu-fixture-live \
  --train \
  --push-as USER/unique-private-cpu-fixture \
  --verify-hub --json
```

`--push-as` is refused unless `--verify-hub` is also present. The latter checks both that the target is already private and that the remote `checkpoint-1` branch contains `adapter_config.json`, `trainer_state.json`, and checkpoint weights after the run. A zero training exit status is not accepted as a Hub-push pass.

Record the command, Soup commit, Python and package versions, local result JSON, remote branch/file listing, and elapsed time in the Issue #74 report. The Space and dataset legs remain separate live checks.
