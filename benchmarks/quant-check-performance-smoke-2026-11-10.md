# quant-check performance: real CUDA and CPU path validation (#2677)

These are functional smokes of the optional combined quality/performance report,
not a speedup benchmark or a model-quality evaluation. Three toy prompts are
insufficient to establish retained model quality. In particular, identical tiny
CPU checkpoints can show a latency difference in such a short measurement.

Runtime: Soup source `26ffea17461d0d14a0d3641b93435343b54c29e1` plus this PR's
performance changes, Python 3.12.3, torch 2.14.1+cu130, Transformers 5.19.0,
bitsandbytes 0.50.2. H200 physical GPU 2 selected by UUID; another idle process
held 49850 MiB. Allocated memory belongs to this process, not to the whole card.

These smokes and `validation.json` describe the earlier tree: its 12-test file
was named `tests/test_issue2677_quant_performance.py`, later renamed to
`tests/test_quant_performance.py` and extended, while CLI follow-ups and merges
changed `src/soup_cli/commands/eval.py` afterwards; the retained hashes, counts
and measurements are historical records, not hashes of the current head.

## What was exercised

- CUDA: `Qwen/Qwen2.5-0.5B-Instruct` at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, compared to a locally exported
  bitsandbytes NF4 copy (FP16 compute). Three chat-templated prompts, contains
  scoring, one first-prompt warm-up and three measured repeats per prompt.
  Both models score 3/3 in this fixture. Both CLI calls finish with exit 0;
  outputs include actual generated token counts, latencies and nonzero allocated
  CUDA peaks, with the NF4 peak below the FP16 one.
- CPU: seeded random one-layer Llama, hidden=32, intermediate=64, vocab=8,
  two heads/one KV head, tokenizer saved with the checkpoint, seed=2677. Two
  identical saved copies, FP32, one prompt, one warm-up and three repeats.
  Both score 0 in the deliberately random fixture; the delta is 0 and exit is 0.
  CUDA memory and its percentage change are JSON null, not zero.
- Each device was exercised twice: first directly, then through the checked-in
  reproduction harness using the prepared checkpoints. All four CLI outputs
  are retained under [`evidence/quant-check-2677/`](evidence/quant-check-2677/).
- The new tests first failed on the unchanged source (9 failures), then pass.
  They cover fake 10 ms/5 ms timings (-50%), a 10-point quality loss still
  returning exit 2, warm-up/extra repeats excluded from scoring, sequential
  object lifetime, cleanup on exceptions, CPU null memory, noise labels,
  invalid budgets and all report formats.
- Full CPU project suite after the runtime/docs changes: **52650 passed,
  663 skipped**, no failures, 322.90 seconds. Whole-tree Ruff passes. The
  reproduction harness was added afterwards and exercised separately; it
  changes no runtime code.

All latency samples and generated token counts are in the JSON. A short first
repeat can retain initialization effects for prompts other than the warm-up
prompt. The observed-range label is deliberately descriptive; it is not a
claim of statistical significance. No speed/memory metric changes the quality
gate or exit status.

## Reproduce

Install this checkout's `[train,dev]` dependencies, select one GPU and run from
the checkout root. The script prepares the two model pairs in the ignored
`.codex` directory, then invokes the real CLI in fresh subprocesses:

```bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python benchmarks/harness/issue2677_smoke.py
```

`--reuse --root .codex/quant-check-smoke` reuses an already prepared pair.
Prepared weights and stderr logs remain local; the public JSON reports and
validation manifest contain the measured values and relevant provenance.
