# Gate record - NF4 layer-major arithmetic on an RTX 3060

**Verdict (2026-10-24) — Mixed.** The first probe failed bitwise parity. The follow-up passed the BF16 noise-floor and FP32 error limits on all three updates.

Measured by @M1kado0.

## 1. Setup

- NVIDIA GeForce RTX 3060, 12 GB. The October 23 host exposed 755 GiB RAM with a cgroup limit of 275,432,603,648 bytes (256 GiB) and a 100 GB overlay. Ubuntu 24.04.2 LTS, CUDA 13.0 development image, driver 595.104.02 reporting CUDA 13.2. The October 24 run recorded the same GPU, image and driver, with Linux 5.4.0-216-generic and glibc 2.39. A separate host RAM inventory is unavailable for October 24.
- Python 3.12.3, torch 2.14.1+cu130, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0 and bitsandbytes 0.50.2.
- `mistralai/Mistral-7B-v0.1`, 32 layers, hidden size 4096, nested NF4 weights and BF16 or FP32 compute and activation storage. Ten synthetic token rows up to 256 tokens cover masking and a partial final window. Both arms accumulate four microbatches per update. `training.l2l` is pending.

## 2. Method and results

Can layer-major microbatching reproduce ordinary streamed gradient accumulation (GA) on real NF4 weights? Two identical GA runs had to be bitwise equal before the layer-major comparison could proceed. Both comparison arms used deterministic MATH SDPA. The follow-up applied a per-update BF16 limit of 1.5 times the relative error between GA with MATH and EFFICIENT_ATTENTION, and a fixed FP32 limit of `1e-4`. The same rules applied to loss and relative L2 gradient error. Step counts, learning rates and log lengths had to match exactly.

| Update | BF16 gradient error / floor / limit | BF16 relative loss error / floor | FP32 gradient error | FP32 relative loss error |
|---|---|---|---:|---:|
| 0 | 1.80% / 1.44% / 2.16% | 3.9e-05 / 9.8e-05 | 2.1e-06 | 7.5e-08 |
| 1 | 1.62% / 1.54% / 2.31% | 6.8e-05 / 7.7e-05 | 3.1e-06 | 1.6e-07 |
| 2 | 5.30% / 4.49% / 6.74% | 3.7e-04 / 6.7e-04 | 8.3e-06 | 1.5e-07 |

Both GA repeat checks passed bitwise. In BF16, GA took the custom NF4 GEMM at M=64 (896 calls) and M=128 (640 calls), and used dequantization at larger M. The layer-major arm used 4,704 dequantized projections with windows padded to M=256. Both FP32 arms used dequantization.

## 3. Commands

The harness and diagnostic wrappers live at the evidence commit.

```bash
export PYTHONPATH=$PWD/src
.venv/bin/python benchmarks/harness/l2l_mistral_stand.py --out /root/runs/mistral7b-01
.venv/bin/python benchmarks/harness/l2l_mistral_stand.py --out /root/runs/mistral7b-07-bf16
.venv/bin/python benchmarks/harness/l2l_mistral_stand.py --dtype float32 --out /root/runs/mistral7b-07-fp32
```

The first command used the October 23 checkout. Its sibling runs used `diag_nocustom.py`, `diag_equal_len.py`, `diag_dump_grads.py`, `diag_dump_grads_efficient.py` and `diag_dump_grads_fp32.py` in the saved results.

## 4. Earlier runs and limits

The follow-up superseded the first probe for tolerance and path accounting. The original bitwise failure stands. The first BF16 run had maximum absolute gradient difference 0.539, and the dump repeat reproduced it. Disabling the custom GEMM, equalizing sequence lengths and switching both arms to Efficient SDPA left differences of 0.336, 1.016 and 0.445, respectively. The FP32 diagnostic measured 4.5e-5 absolute difference. The first run counted 184 decoder-layer loads, 7 large-module loads and zero NF4 restores. The zero restore counter and the load comparison at accumulation one remain unresolved.

The BF16 floor changes the attention backend within GA and leaves the projection-path difference unisolated. These synthetic runs establish a finite arithmetic tolerance, with quality and training on real data unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/l2l-mistral7b-a3-vast-rtx3060-ticket07
