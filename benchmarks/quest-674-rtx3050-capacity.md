# Bounded QuEST mixed-route capacity run on an RTX 3050 (#674)

**Eight of eight optimizer updates completed at batch size 1 on 32 synthetic ChatML rows (45 maximum formatted tokens), without OOM. The sampled whole-card usage reached 8032/8192 MiB. PyTorch's allocator peak counters exceeded physical VRAM and are unexplained; no reliable process-specific peak is claimed. This is not a parity, quality, or efficiency result.**

This is the bounded capacity check requested in [#674](https://github.com/MakazhanAlpamys/Soup/issues/674). It is separate from the evaluation-only and strict-W4A4 records. It does not touch FINAL256 or OASST256 and does not establish QuEST numerical parity, training quality, or general hardware support.

## Setup

| Item | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 8192 MiB, compute capability 8.6 |
| OS | Windows 11 Pro, build 26200 |
| NVIDIA driver | 616.56 |
| Soup | 0.77.2, training extra |
| Python | 3.13.12 |
| PyTorch / CUDA build | 2.15.0+cu132 / 13.2 |
| Model | ahxt/LiteLlama-460M-1T, revision 77b8a976440e7d1ea5a890eaf1e0175b1cac0078 |
| Data | 32 locally generated synthetic arithmetic ChatML rows; longest formatted row 45 tokens |
| Run ID | run_20261111_204514_8f2f2efd |

Measured configuration (local input/output paths shortened):

~~~yaml
base: ahxt/LiteLlama-460M-1T
task: sft
backend: transformers
modality: text

data:
  train: ./train.jsonl
  format: chatml
  chat_template: chatml
  val_split: 0
  max_length: 2048

training:
  epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 4
  quantization_aware: quest
  quantization: none
  lora:
    r: 0

output: ./output
~~~

Soup reported 168 W4 weights, with 161 A4 and 7 A16 activation paths. This experimental route is neither pure W4A4 nor packed INT4.

## First run and follow-up

The initial run reported 8/8 optimizer updates in 10 minutes 41 seconds on 32
synthetic ChatML rows, with a longest formatted row of 59 tokens. Its environment
reported PyTorch `2.14.1+cu130`. Soup printed `GPU peak: 11.2/8.0 GB`; direct
`max_memory_allocated` and `max_memory_reserved` readings were not captured, so
this remains an unexplained summary value rather than a physical VRAM measurement.

The follow-up report says the first draft exceeded the requested 59-token row
limit, so the data was shortened and the run repeated with rows up to 45 tokens.
That run used PyTorch `2.15.0+cu132` and is the measured run detailed below. The
two runs therefore differ in row length and PyTorch build; this record does not
treat them as a controlled comparison. The first-run result reported a 59-token
maximum, while the follow-up comment describes the draft that prompted the rerun
as over the limit; both statements are retained as reported rather than
reconciled without the original row-level artifacts.

## Result

- 8/8 optimizer updates completed without OOM. Trainer runtime: 370.8 seconds; total harness time from preflight through command exit: 429.98 seconds.
- Immediately before training, after CUDA initialization, `torch.cuda.mem_get_info(0)` returned 7,562,330,112 free bytes of 8,589,410,304 total bytes.
- `nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total` was sampled every three seconds. Sampled total-card usage reached 8032/8192 MiB (160 MiB available); sampled temperature peaked at 66 °C. This is whole-card usage, not process-specific.
- After resetting CUDA peak stats immediately before the Soup command, `torch.cuda.max_memory_allocated(0)` returned 11,963,378,688 bytes and `torch.cuda.max_memory_reserved(0)` returned 12,354,322,432 bytes. Both exceed physical device memory and conflict with the completed run and sampled `nvidia-smi` readings. Soup also printed `GPU peak: 11.1/8.0 GB`. These readings are recorded as anomalous, not as physical VRAM usage.

No quality evaluation, parity check, packed-kernel run, or comparison with a full-precision training control was performed. The run shows only that this bounded configuration completed on this GPU under the recorded stack.
