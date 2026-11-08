# Fast-LoRA MLP on a Colab T4

Measured 2026-11-08 in Google Colab on one Tesla T4 (sm_75, 15 GiB, driver 580.82.07). The fixture is one synthetic Llama-3.1-8B-shaped SwiGLU MLP (`N=2048`, `d=4096`, `f=14336`, `r=64`) with adapters on all three projections (`lora_alpha=2r`, dropout 0, seed 837). NF4 uses double quantization and BF16 compute. It does not load a language model and is not an end-to-end training result. BF16 on this T4 is emulated, not native.

Each dtype ran in its own fresh Python process (emulated BF16 first, then NF4). Within a process, each arm received five warm-ups and 30 forward-plus-backward CUDA-event measurements. The harness rotates through all six permutations of `fast`, `plain`, and `liger`; over 30 repetitions each permutation occurs five times. Saved-tensor hooks capture unique tensor views, deduplicated by storage and view metadata. The compact record in [`evidence/issue-837-cuda-t4.json`](evidence/issue-837-cuda-t4.json) was assembled by hand from that run's Colab notebook output; the original raw harness JSON was not retained. Per-arm output equality and gradient L2 figures come from the notebook's parity-capture output; the JSON condenses each comparison to a boolean and a maximum. [`harness/fast_lora_mlp_report.py`](harness/fast_lora_mlp_report.py) converts future raw harness files to a similar, but not identical, schema.

| Dtype | Path | Median fwd+bwd | IQR (p25–p75) | Hook-visible saved tensors (raw) | Peak allocated | SM clock min / median / max |
|---|---|---:|---:|---:|---:|---:|
| Emulated BF16 (sm_75) | Fast-LoRA | 595.26 ms | 582.61–610.80 ms | 494,141,440 B | 1,306.2 MiB | 855 / 945 / 1,470 MHz |
| Emulated BF16 (sm_75) | Plain PEFT | 582.25 ms | 569.57–595.87 ms | 611,844,096 B | 745.8 MiB | 855 / 945 / 1,470 MHz |
| Emulated BF16 (sm_75) | PEFT + Liger | 582.86 ms | 566.64–595.36 ms | 553,123,840 B | 689.8 MiB | 855 / 945 / 1,470 MHz |
| NF4 | Fast-LoRA | 673.56 ms | 654.93–688.39 ms | 232,699,084 B | 1,128.9 MiB | 735 / 870 / 960 MHz |
| NF4 | Plain PEFT | 652.18 ms | 640.02–667.04 ms | 259,522,560 B | 720.4 MiB | 735 / 870 / 960 MHz |
| NF4 | PEFT + Liger | 656.89 ms | 637.21–668.26 ms | 200,802,304 B | 664.4 MiB | 735 / 870 / 960 MHz |

## How to read saved-tensor bytes

The raw count is exactly what `saved_tensors_hooks` observed; it is not a like-for-like activation-memory comparison for NF4. In bitsandbytes 0.50.2, `MatMul4Bit.forward` retains the packed weight on its autograd context (`ctx.tensors`) and the quantization metadata in `ctx.state`, outside `save_for_backward`. Those objects are therefore absent from the hook in the PEFT and PEFT + Liger arms. Fast-LoRA explicitly passes packed NF4 weight parts to `ctx.save_for_backward`, so the hook counts them. This accounts for 88,080,384 B of packed weights and 2,798,796 B of quantization state in Fast-LoRA's NF4 raw total. The implementation is documented in [bitsandbytes 0.50.2 `MatMul4Bit`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py).

Subtracting those 90,879,180 B from Fast-LoRA's NF4 raw total gives a derived count of 141,819,904 B excluding base weights and quantization state. The corresponding PEFT and Liger hook totals are 259,522,560 B and 200,802,304 B. For emulated BF16, all three paths expose the same 352,321,536 B base-weight total to the hook; subtracting that common amount gives the same derived counts: Fast-LoRA 141,819,904 B, PEFT 259,522,560 B, and Liger 200,802,304 B. These figures are arithmetic derived from the recorded hook totals and source inspection, not separate measurements; they still include LoRA weights. Fast-LoRA's derived count excluding base weights and quantization state is 117,702,656 B lower than PEFT and 58,982,400 B lower than Liger in both dtypes.

## Timing and limits

Fast-LoRA's median was 13.0 ms higher than plain PEFT in emulated BF16 and 21.4 ms higher in NF4. Its own IQR widths were 28.2 ms and 33.5 ms, respectively. Under the Q/K/V timing rule—call faster/slower only if the gap exceeds the spread inside either arm and the ratios agree across three processes—these results show **no measurable difference**. There was only one process per dtype, so process-to-process variability was not measured and there is no speed verdict. No noise threshold was specified before this run.

The raw saved-tensor counts show fewer hook-visible bytes for Fast-LoRA than plain PEFT in both dtypes; the NF4 raw difference is confounded by the different autograd retention paths described above. Fast-LoRA also had higher measured peak allocated memory than both comparison paths. This run did not capture an allocation breakdown that explains the extra peak, so its cause remains unknown.

Outputs matched plain PEFT exactly for both Fast-LoRA and Liger in both dtypes. The largest relative L2 difference across input and LoRA gradients was 0.00432 for Fast-LoRA and 0.00468 for Liger in emulated BF16; in NF4 the maxima were 0.00431 and 0.00467. Each arm's autograd path proof passed all 30 repetitions. These results describe this single-card, single-block fixture only.
