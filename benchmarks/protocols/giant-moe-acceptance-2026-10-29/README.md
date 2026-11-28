# Giant MoE acceptance protocol

The frozen protocol evaluates the pinned DeepSeek-V3 checkpoint before and after two
attention-LoRA updates. Each arm uses the same eight WikiText-2 test fragments, token IDs,
and model/cache dtype. Each fragment is truncated independently to 64 tokens. The split,
revision, selection rule, row IDs, and checksums are in [evaluation-data.json](evaluation-data.json).
The developer-authored training and held-out probe strings are in
[giant-moe-pilot-data.json](../../../examples/giant-moe-pilot-data.json).

Record per-fragment causal loss, predicted-token count, and token digest. Report the
token-weighted mean negative log-likelihood, perplexity, and paired deltas, including
degradation. Exact adapter and logits reload is a separate requirement. The results
cover the selected fragments and two optimizer updates.

Use native BF16 when supported and FP32 otherwise. Use the same per-expert NF4 cache with
double quantization disabled for both arms.

## RAM accounting

Run the download, preparation, training, both evaluation arms, and reload under an
observed hard limit of at most 32 GiB. The Linux monitor accepts a bounded cgroup or a
physical host within the pilot's RAM range with host swap disabled. It resolves the
current process's controller against its mounts and records RAM, swap, OOM and RSS
accounting. Keep the monitor and all measured phases in the same constrained scope.
The measured command must write these phase names to the phase file as it runs:
`download`, `preparation`, `training`, `evaluation_before`, `evaluation_after`,
and `reload`.

```bash
python -m benchmarks.harness.giant_moe_ram \
  --output "$RAM_OUTPUT" --phase-file "$PHASE_FILE" -- "$WORKLOAD"
```

## Replay recorded evidence

Run these modules from the repository root. The basic verdict consumes the event
journal, cache index, and one-row physical GPU CSV. The acceptance verdict also
requires RAM samples, the workload exit code, and the exact protocol and evaluation
files referenced by the journal's digests. Supply the recorded physical VRAM in MiB.

```bash
python -m benchmarks.harness.giant_moe_verdict \
  --events "$EVENTS" --index "$INDEX" --gpu "$GPU_CSV" --output pilot-verdict.json
python -m benchmarks.harness.giant_moe_acceptance_verdict \
  --events "$EVENTS" --index "$INDEX" --ram-samples "$RAM_SAMPLES" \
  --protocol "$PROTOCOL" --evaluation-data "$EVALUATION_DATA" \
  --physical-vram-mib "$PHYSICAL_VRAM_MIB" --returncode "$WORKLOAD_EXIT_CODE" \
  --output acceptance-verdict.json
```

Each verdict exits with status 1 when its checks fail. Final acceptance requires
external review. Historical replay needs the original protocol and evaluation file
bytes from the recorded run. The checked-in fixtures retain the texts, row order, and
per-text hashes, with updated whole-file digests after metadata cleanup.

## Dataset attribution

WikiText was compiled by Stephen Merity, Caiming Xiong, James Bradbury, and Richard
Socher (2016) from Wikipedia contributors. The fragments retain the dataset's license
notices. Its pinned dataset card identifies CC BY-SA 3.0 and GFDL in its metadata and
references CC BY-SA 4.0 in its licensing section. The fixture records the pinned dataset
revision and source Parquet checksum.
