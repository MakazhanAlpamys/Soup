<!--
Measurement record, published as measured.

Hardware: RTX 5070 Laptop GPU 8 GB (Blackwell sm_120; torch reports 8,518,041,600 bytes),
32 GB RAM, Windows 11 (Windows-11-10.0.26200-SP0). A NEW baseline, never compared with
RTX 3050 numbers.
Stack: Python 3.12.10, torch 2.14.0+cu130.
Soup: PR #2510 (`feat/streamed-inference-mode`), the three frozen files at f6997ec62; the
rule was committed on top of it as 4fea4c4fc.
Harness: benchmarks/harness/stream_infer_throughput.py (re-runnable).
Rule: benchmarks/protocol-2509-stream-inference-throughput.md, written and committed before
the first timed run and not edited after it.
Raw output: benchmarks/results/2509-stream-infer-throughput-m7.json and
benchmarks/results/2509-stream-infer-throughput-q14.json, the same JSON as the harness wrote
it, with LF line ends and a final newline (see Files).
-->

# Gate record: forward-only throughput of the streamed inference mode (#2509)

**Status: MEASURED on one laptop (RTX 5070 Laptop GPU 8 GB), 2026-11-06, one run per arm.
G1 PASS. G2 PASS.** M7 (Mistral-7B NF4, pinned RAM tier) at batch 1 x 512 reached 1854.8
tok/s against a rule of at least 972. The best Q14 cell (Qwen2.5-14B NF4 on the disk tier,
batch 4 x 2048, peak VRAM 4.149 GiB) reached 1186.3 tok/s against a rule of at least 500. No
cell ran out of memory and no arm was void.

These are timing figures only. The input ids are uniform random from a fixed generator, so
nothing here is a quality claim. Logit correctness is covered by the CPU bit-exact tests in
`tests/test_layer_stream_inference_mode.py`, not by this record.

Unit convention: peak VRAM in the tables is in GiB (bytes / 2^30); "GB" in the text is
decimal (10^9 bytes), as in the other records. Every Q14 cell is far below the 7.5 GB limit
of G2 in either unit (the largest peak is 4,454,512,128 bytes = 4.149 GiB = 4.455 GB).

---

## 1. Code identity

The rule froze three files by sha256 before the first run. The maintainer verified them
equal before the run. They were checked again for this record with `sha256sum` on the
worktree of PR #2510 (head `4fea4c4fc`, the commit that adds only the rule file on top of
`f6997ec62`):

| file | sha256 in the rule | sha256 re-computed | equal |
|---|---|---|---|
| `benchmarks/harness/stream_infer_throughput.py` | `50f30b2ab6062e6f5f50cc30f68349ef231de80ffa18b105d6969ef53bf2a1be` | `50f30b2ab6062e6f5f50cc30f68349ef231de80ffa18b105d6969ef53bf2a1be` | yes |
| `src/soup_cli/utils/stream_inference.py` | `81fe702773a2fbba64ab60ca49c97e467341eadd43b074d684cc4a6c263f4674` | `81fe702773a2fbba64ab60ca49c97e467341eadd43b074d684cc4a6c263f4674` | yes |
| `src/soup_cli/utils/layer_stream_runtime.py` | `e318e5554abd41e2810df8b1027d5ea513695531fd0de182a0522324837df93b` | `e318e5554abd41e2810df8b1027d5ea513695531fd0de182a0522324837df93b` | yes |

The re-check proves the files are unchanged now; the equality at run time rests on the
maintainer's check before the run. The first line of the run log names the `soup_cli`
import path, and it is inside the `src` of the PR worktree, not a copy in `site-packages`.

## 2. Commands as run

One process per arm, M7 first, then Q14. Each grid cell: 2 untimed warm-up forwards, then 5
timed forwards with `torch.cuda.synchronize()` around each; the median is reported.

```
# M7: RAM tier (pinned), two buffers
python benchmarks/harness/stream_infer_throughput.py \
  --model unsloth/mistral-7b-instruct-v0.3 \
  --shard-dir ~/.soup/layer-stream/unsloth__mistral-7b-instruct-v0.3 \
  --tier ram --buffers 2 --batches 1,2,4 --seq-lens 128,512,2048 \
  --warmup 2 --repeats 5 --out m7.json

# Q14: disk tier (direct I/O), two buffers
python benchmarks/harness/stream_infer_throughput.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --shard-dir ~/.soup/layer-stream/Qwen__Qwen2.5-14B-Instruct \
  --tier disk --buffers 2 --batches 1,2,4 --seq-lens 128,512,2048 \
  --warmup 2 --repeats 5 --out q14.json
```

`--out` pointed at a scratch folder outside the repository; the two JSON files were copied
into `benchmarks/results/`; only the line ends and a final newline differ (see Files). Settings the JSON records besides the flags: `pin`
auto, `read_ahead` and `dtype` at their defaults (null), vocabulary size 32768 (M7) and
152064 (Q14). Both shard indexes report quant nf4 and dtype bfloat16: 32 layers and
7,248,023,552 parameters (M7), 48 layers and 14,770,033,664 parameters (Q14).

The harness times one full forward, `stream.logits(input_ids)`, with ids drawn by
`torch.randint` from a generator seeded with 17. Peak VRAM is `torch.cuda.max_memory_allocated`,
reset per cell: the allocated peak, not the reserved one and not an `nvidia-smi` reading.

Both arms exited 0 (exit code 3 would mean an out-of-memory cell). Stdout carried the nine
result lines per arm, a `wrote` line and two warnings (unauthenticated Hub requests;
triton not found for flop counting).

## 3. Validity

Checked against each void condition of the rule, per arm, before the arm started:

```
PRE M7 10:08:44Z apps=[] gpu_used_MiB=0 free_ram_GB=8.52 valid=True
M7 exit=0 10:10:24Z
PRE Q14 10:10:26Z apps=[] gpu_used_MiB=0 free_ram_GB=7.9 valid=True
Q14 exit=0 10:13:35Z
```

| void condition | how it was checked | M7 | Q14 | result |
|---|---|---|---|---|
| another process held GPU memory (`nvidia-smi --query-compute-apps` not empty, or more than 600 MiB used before the arm) | the query and the used-memory reading, immediately before each arm | list empty, 0 MiB | list empty, 0 MiB | not void |
| free RAM below 6 GB | free physical RAM, immediately before each arm | 8.52 GB | 7.90 GB | not void |
| the box went to sleep during the run | the Windows System log between 10:08Z and 10:14Z on 2026-11-06 searched for Kernel-Power and Power-Troubleshooter events; none found. The JSON timestamps are continuous: 96.0 s for M7 and 184.5 s for Q14, each ending about 2 s before the exit time in the run log | none | none | not void |

Also recorded: the pytest suite was not running during either arm. The hold on other work on
the box was announced at about 10:07Z, before the first arm, and lifted at 10:15Z. The
announcement named about 10:15Z as its start, and the arms ran earlier than that, from
10:08:44Z to 10:13:35Z. One CPU-only job of another user of the box ran until 10:11:15Z, so it
overlapped M7 and the first 49 s of Q14; it held no GPU memory, and it is not one of the rule's
void conditions. No arm was void and none was re-run.

## 4. The reference

From `probe-rtx5070-what-bounds-streaming.md` section 3, the training step of Mistral-7B NF4
on the pinned RAM tier at batch 1 x 512 tokens (LoRA r=8, synthetic ids): **1.053 s** per
step (8 timed steps, spread 1.041 to 1.069 s), **486.0 tok/s**, moving 7.40 GB through 61
layer loads and 2 large loads at 16.7 GB/s on the copy stream. G1 takes 2 x 486.0 = 972
tok/s from it.

## 5. M7 grid: Mistral-7B NF4, RAM tier, 2 buffers

| batch x seq | tokens | tok/s (median) | median s | min-max of 5 runs (s) | peak VRAM (GiB) | layer loads per forward |
|---|---|---|---|---|---|---|
| 1 x 128 | 128 | 552.3 | 0.2318 | 0.2056 - 0.4110 | 0.611 | 32 + 2 large |
| 1 x 512 | 512 | 1854.8 | 0.2760 | 0.2730 - 0.2785 | 0.643 | 32 + 2 large |
| 1 x 2048 | 2048 | 2300.9 | 0.8901 | 0.8897 - 0.8953 | 0.773 | 32 + 2 large |
| 2 x 128 | 256 | 1185.2 | 0.2160 | 0.2040 - 0.2227 | 0.618 | 32 + 2 large |
| 2 x 512 | 1024 | 2252.5 | 0.4546 | 0.4536 - 0.4635 | 0.686 | 32 + 2 large |
| 2 x 2048 | 4096 | 2311.5 | 1.7720 | 1.7632 - 1.7814 | 0.945 | 32 + 2 large |
| 4 x 128 | 512 | 1841.4 | 0.2780 | 0.2749 - 0.2829 | 0.643 | 32 + 2 large |
| 4 x 512 | 2048 | 2386.2 | 0.8583 | 0.8560 - 0.8590 | 0.772 | 32 + 2 large |
| 4 x 2048 | 8192 | 2353.1 | 3.4814 | 3.4737 - 3.4986 | 1.398 | 32 + 2 large |

- Every forward reloaded every layer (32 layer loads and 2 large loads per forward, the
  harness's own counters), so the stream was measured, not a pool that kept the model
  resident. The harness's upper-bound byte estimate is 4,137,812,864 bytes per forward.
- The first cell (batch 1 x 128) is the first timed cell of the first arm. Its five runs in
  order were 0.4110, 0.2459, 0.2318, 0.2184, 0.2056 s: still falling after two warm-up
  forwards, and the widest spread of the 18 cells. The median 0.2318 s is reported as the
  rule states and feeds no gate.

## 6. Q14 grid: Qwen2.5-14B NF4, disk tier, 2 buffers

| batch x seq | tokens | tok/s (median) | median s | min-max of 5 runs (s) | peak VRAM (GiB) | layer loads per forward |
|---|---|---|---|---|---|---|
| 1 x 128 | 128 | 70.7 | 1.8115 | 1.7923 - 1.8333 | 1.788 | 48 + 2 large |
| 1 x 512 | 512 | 284.3 | 1.8008 | 1.7144 - 1.8111 | 1.929 | 48 + 2 large |
| 1 x 2048 | 2048 | 1070.0 | 1.9140 | 1.8912 - 1.9325 | 2.350 | 48 + 2 large |
| 2 x 128 | 256 | 145.9 | 1.7552 | 1.7006 - 1.7773 | 1.903 | 48 + 2 large |
| 2 x 512 | 1024 | 562.5 | 1.8205 | 1.8051 - 1.9393 | 2.050 | 48 + 2 large |
| 2 x 2048 | 4096 | 1167.4 | 3.5087 | 3.5010 - 3.5154 | 2.949 | 48 + 2 large |
| 4 x 128 | 512 | 294.3 | 1.7397 | 1.6820 - 1.7999 | 1.929 | 48 + 2 large |
| 4 x 512 | 2048 | 1095.6 | 1.8693 | 1.8652 - 1.9184 | 2.350 | 48 + 2 large |
| 4 x 2048 | 8192 | 1186.3 | 6.9056 | 6.8396 - 6.9585 | 4.149 | 48 + 2 large |

- Every forward reloaded every layer (48 + 2 large per forward). The harness's upper-bound
  byte estimate is 9,931,636,032 bytes per forward.
- No cell ran out of memory. The page-cache state of the Q14 store was not recorded.

## 7. Gate verdicts

Computed as the rule states.

**G1, overhead of the inference path: PASS.**
- Rule: M7 at batch 1 x 512 reaches at least 972 tok/s (2 x 486.0).
- Measured: 1854.8 tok/s (median 0.2760 s over 5 timed forwards, 0.2730 to 0.2785 s).
- Margin: +882.8 tok/s, 1.91x the threshold (+90.8%). The slowest of the five runs, 0.2785 s,
  is 1838.7 tok/s and also clears it; the threshold corresponds to a median of 0.5267 s.
- For scale: 3.82x the 486.0 tok/s of the training step that the rule was derived from.

**G2, usefulness on a model larger than VRAM: PASS.**
- Rule: the best Q14 grid cell with peak VRAM at or below 7.5 GB reaches at least 500 tok/s.
- Cells at or below 7.5 GB: all nine (largest peak 4.455 GB). The best of them is batch 4 x
  2048 at 1186.3 tok/s (median 6.9056 s).
- Margin: +686.3 tok/s, 2.37x the threshold (+137.3%). The slowest of the five runs, 6.9585 s,
  is 1177.3 tok/s and also clears it.
- Five of the nine cells reach 500 tok/s: 2 x 512 (562.5), 1 x 2048 (1070.0), 4 x 512
  (1095.6), 2 x 2048 (1167.4) and 4 x 2048 (1186.3). The smallest of these, 2 x 512, clears
  it by 12.5%.
- No cell is within 5% of a threshold, so the rule's near-miss clause does not apply.

For scale, as arithmetic on the grid and not a measured evaluation pass: the rule sizes 500
tok/s as 512k tokens (1,000 sequences of 512 tokens) in 17.1 minutes. At the 4 x 512 rate
of 1095.6 tok/s the same 512k tokens take 7.8 minutes; at the 2 x 512 rate of 562.5 tok/s,
15.2 minutes.

## 8. Informational items (no verdict)

**Scaling curve.** Throughput follows the tokens per forward, not the split of batch and
sequence length:

| tokens per forward | M7 tok/s | Q14 tok/s |
|---|---|---|
| 128 | 552.3 | 70.7 |
| 256 | 1185.2 | 145.9 |
| 512 | 1854.8 (1 x 512), 1841.4 (4 x 128) | 284.3 (1 x 512), 294.3 (4 x 128) |
| 1024 | 2252.5 | 562.5 |
| 2048 | 2300.9 (1 x 2048), 2386.2 (4 x 512) | 1070.0 (1 x 2048), 1095.6 (4 x 512) |
| 4096 | 2311.5 | 1167.4 |
| 8192 | 2353.1 | 1186.3 |

At equal tokens the two splits are within 0.7% to 3.7% of each other on M7 and 2.4% to 3.5%
on Q14. M7 flattens from 1024 tokens up (2252.5 to 2386.2 tok/s, 5.9% apart over five
cells); Q14 flattens from 4096 tokens up (1167.4 and 1186.3).

**Peak VRAM.** M7: 0.611 to 1.398 GiB (0.656 to 1.501 GB). Q14: 1.788 to 4.149 GiB (1.920
to 4.455 GB). The peak grows with tokens per forward and is the largest at batch 4 x 2048 in
both arms. No cell ran out of memory, so none was retried.

**Q14 at batch 1 x 128, the read floor of one forward.** 1.8115 s per forward (70.7 tok/s),
min-max 1.7923 to 1.8333 s. The lowest median anywhere in the Q14 grid is 1.7397 s (batch 4
x 128). On the harness's byte estimate for one forward (9.93 GB) that is 5.5 GB/s at 1.8115
s and 5.7 GB/s at 1.7397 s. This is arithmetic on an upper-bound byte count, not an
instrumented read rate.

**Read-bound or compute-bound, from batch 1 x 128 against batch 4 x 2048.**
- Q14: 1.8115 s for 128 tokens, 6.9056 s for 8192 tokens. 64x the tokens cost 3.81x the time,
  and the 128-token forward is 26% of the 8192-token forward.
- Q14 cells of at most 2048 tokens (seven cells) keep their median between 1.7397 s and
  1.9140 s, a 10% spread over a 16x range of tokens: read-bound. At 4096 and 8192 tokens the
  time is proportional to tokens (0.857 and 0.843 ms per token): compute-bound. The knee lies
  between 2048 and 4096 tokens per forward.
- M7: 0.2318 s for 128 tokens, 3.4814 s for 8192 tokens, 15.0x the time for 64x the tokens.
  The time is about 0.22 to 0.23 s at 128 and 256 tokens and 0.28 s at 512 tokens, then
  proportional to tokens (0.42 to 0.44 ms per token from 1024 tokens up). The knee lies between 256 and
  512 tokens per forward.
- At batch 4 x 2048, where both arms are compute-bound, Q14 takes 1.98x the time of M7
  (6.9056 s against 3.4814 s), against a parameter ratio of 2.04 (14.77 B against 7.25 B).

**A note on copy rates, not a finding.** The same byte arithmetic gives 17.9 GB/s for the M7
batch 1 x 128 median (0.2318 s) and 19.2 GB/s for the 2 x 128 median (0.2160 s), above the
16.7 GB/s copy-stream rate that the reference measured with CUDA events on the training
step. This harness does not instrument the copy, and the byte figure is an upper-bound
estimate (a load counter times the slot size), so no copy rate is claimed.

## 9. What this record does NOT say

- No quality claim: uniform random ids, timing only. Logit correctness is the CPU tests'
  job (`tests/test_layer_stream_inference_mode.py`).
- One laptop, one run per arm. The five timed forwards per cell are repeats inside one
  process, not independent runs.
- New numbers for this box. Nothing is compared with an RTX 3050 record. The only earlier
  figure used is the reference of section 4, the training step measured on this card on
  2026-09-12; the G1 comparison with it is the rule's own.
- Not measured: batch above 4 or sequence length above 2048; a cold page cache or a
  different drive for Q14 (its page-cache state was not recorded); other models; any
  instrumented copy or disk read rate; reserved VRAM.
- The batch 1 x 128 cell of M7 was still speeding up when the timed runs ended (section 5).
  It feeds no gate.

## 10. Consequence under the protocol

G1 passes, so the inference path does not first need an overhead investigation. G2 passes,
and by the protocol a PASS of G2 means step 1 of #2509 gets built: wiring the forward-only
mode into evaluation and judge scoring, and into batch forward passes on models larger than
VRAM.

## Files

| file | committed bytes | committed sha256 | harness output bytes | harness output sha256 |
|---|---|---|---|---|
| `benchmarks/results/2509-stream-infer-throughput-m7.json` | 5227 | `15681169bf8af33ae66b6bbe05a7b71a97c925015b65a709a550edfdb3622c74` | 5429 | `6c0072ac469469d0149bdc709e6051b20d614e6cbc1fef67d57c2e07a12c04db` |
| `benchmarks/results/2509-stream-infer-throughput-q14.json` | 5197 | `cb9904ae6e2d45bb64dc269f587d2a22198f9f6421251f629a01dd570b0f1227` | 5399 | `37a38de5536b26f1c4f3aa883daab2f317f44e2b6c6558af89afee14a0f05a07` |

The harness wrote CRLF line ends and no final newline. The committed files hold the same JSON with LF line ends and a final newline (the repository's line-end normalisation and end-of-file hook); converting them back to CRLF and dropping the final newline reproduces the harness output hashes.
