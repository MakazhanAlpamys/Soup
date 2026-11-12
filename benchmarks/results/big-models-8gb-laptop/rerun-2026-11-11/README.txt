Result files of section 13 of benchmarks/run-big-models-8gb-laptop.md: three runs through `soup train` on the
RTX 5070 Laptop, 2026-11-11. File names carry the configuration: qwen3-32b, mistral123-128, mistral123-512.

Instrument files
  train-<config>.log               console log of `soup train` and of the wrapper's logits capture
  driver-<config>.log              driver summary: wall time, exit code, whole-device peak read from gpu-<config>.csv
  gpu-<config>.csv                 one row per second from nvidia-smi, no header; columns: time (local time,
                                   UTC+5), memory.used (MiB), memory.total (MiB), utilization.gpu (%), power.draw
                                   (W), temperature.gpu (C)
  host-<config>.csv                one row about every 2.4 s, no header; columns: UTC time, then free physical
                                   memory, free commit and the commit limit in GiB, each printed with a decimal
                                   comma by the machine's locale (11,37 is 11.37)
  step-losses.json                 the exact step losses, read from each run's checkpoint-3/trainer_state.json
  <config>.soup.yaml               the config of the run (paths shortened)
  reload-compare-vs-live-<config>.json   the comparison script's result against the training process's capture
  reload-two-process-<config>.log        the two-process fallback result
  live-vs-reload-posthoc-<config>.log    the CPU comparison of the saved live and reload logits
  import-summary-mistral123.json   the import summary of the Mistral Large shard store
Discarded Mistral Large 128-token attempt
  discarded-attempt-driver-mistral123-128.log, discarded-attempt-gpu-mistral123-128.csv,
  discarded-attempt-host-mistral123-128.csv
Earlier-runner Qwen3-32B reload-only run (section 13.2) and waiting
  earlier-harness-q32-reload-check.txt       excerpt of the wrapper log
  wait-checks-mistral123-512.txt, wait-checks-earlier-runner-q32-reload.txt,
  wait-checks-earlier-runner-q32-repeat2.txt  the operator's wait logs, scrubbed
Operator's notes (statements no instrument file shows)
  operator-notes-2026-11-11.txt

Preparation
  Paths were shortened: <ASSETS> (the folder of the published checkpoints), <CACHE> (the shard-store folder),
  <OUT> (the output folder) and the directory token of the kit's config files, now `kit\`. Box-drawing
  characters of the console panels are ASCII (two logs had been written through a console code page and were
  converted back first). Two driver lines that print the lowest free host memory and commit are dropped: the
  decimal comma made them wrong; the section 13 figures were computed again from the host-*.csv files. In the
  wait files the list of programs using the GPU is removed.
  Measurement anomalies: utilization.gpu reads [Unknown Error] in 9 rows (1 in gpu-qwen3-32b.csv, 8 in
  gpu-mistral123-512.csv) and power.draw holds impossible values, up to 1804.68 W, in 11 rows at the start of
  those two runs; memory.used parses in every row, so the peaks are unaffected.
