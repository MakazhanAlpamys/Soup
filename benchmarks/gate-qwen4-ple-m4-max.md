# Qwen4-Exp PLE / oQ gate on M4 Max — partial pass, production smoke stopped

Date: 2026-08-31

Hardware: MacBook Pro M4 Max, 128 GiB unified memory, external SSD measured at
approximately 946 MB/s sequential read. Software: macOS, Python 3.12, PyTorch
2.13.0, Transformers training floor from the Soup development environment.

This record separates architecture correctness from the feasibility of training
the full production checkpoint. It must not be read as a throughput, peak-memory,
or production-trainability claim.

## Checkpoint

The local source was `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`, an oMLX/oQ affine
Qwen4 bundle:

- `model_type=qwen4_exp`, 48 decoder layers;
- 106.29 GB of source safetensors;
- mixed 4/5/6/8-bit affine matrices with BF16 scales and biases;
- a 320,001,536 x 160 packed PLE N-gram table split across 128 parts;
- vision-tower and MTP components outside the text-only CausalLM.

## Passed gates

1. The oQ-to-Transformers shard map was checked against the real text-decoder
   meta skeleton: 1,167 expected tensors, 1,167 cached tensors, zero missing,
   unexpected, or shape-mismatched entries.
2. The reusable decoder cache completed at 48 layers and 251.47 GB on disk.
   The packed PLE source remained external and read-only; the projected dense
   102.40 GB table was not copied into that cache.
3. Independent oQ affine decoder vectors matched MLX for 4, 5, 6, and 8 bits.
4. The tiny native Qwen4 resident-versus-streamed CPU gate remained bit-exact
   for rows, logits, loss, and LoRA gradients. Both MPS variants passed locally:

   ```text
   2 passed, 21 deselected in 5.94s
   ```

5. The production checkpoint reached tokenizer load, cache reuse, PLE mmap,
   text-only model construction, LoRA injection, dataset preparation, and
   `Training started!` without a key, shape, dtype, or module-routing error.

## Production smoke result — STOPPED / not validated

The one-step BF16 MPS SFT did not complete. It repeatedly stopped during the
first streamed forward without a Python traceback or adapter output. The host
owner reported that this workload can destabilize and shut down the Mac, and
requested that no further full-checkpoint training be attempted.

At the last setup point Soup reported:

```text
48 layers, 251.47 GB disk stream cache
2 x 5357 MB decoder buffers + 1 x 1271 MB large-layer slot
21,214,912 trainable / 176,943,899,555 total parameters
```

System memory sampled after the stop was 96% free with no swap I/O, so this
record does not attribute the stop to host-RAM exhaustion. No macOS diagnostic
report or Python fatal traceback was produced; the precise low-level cause is
therefore unresolved.

## Verdict

- **PASS:** checkpoint discovery, oQ affine decoding, fused-expert mapping,
  external read-only PLE rows, cache construction, tiny CPU/MPS parity, and
  production-checkpoint setup.
- **NOT VALIDATED:** a complete optimizer step for the 176.9B-parameter
  production checkpoint on M4 Max.
- **PENDING:** Qwen4 resident-versus-streamed BF16 parity on CUDA.

Do not market this record as proof that the full model is trainable on a
128 GiB Mac. Smaller Qwen4 checkpoints and CUDA BF16 parity remain the safe next
validation targets.
