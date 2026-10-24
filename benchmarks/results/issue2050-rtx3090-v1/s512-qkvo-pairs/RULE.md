label: 2050-s512-qkvo
A args: '--policy full'
B args: '--policy qkvo'
common: '--seq 512 --warmup 10 --steps 50 --no-ceiling'
pairs: 5
written: 2026-10-23T10:53:34+0100

Balanced AB/BA pairs of benchmarks/harness/issue361_nf4_throughput.py, one fresh
process per arm, with admission guards checked before EVERY arm (no waiting: a failed
guard stops the run and keeps what was measured).

Rule (written before measuring, also copied into <out>/RULE.md):
  metric   = measured-window seconds of an arm (sum of the harness's measured step times)
  per pair = reduction % = (A - B) / A * 100, A = control arm, B = candidate arm
  verdict  = CONFIRMED if median >= 2.0 and the 95% whole-pair bootstrap lower bound > 0
             (10,000 draws, seed 42); NO GAIN if the upper bound < 0; else INCONCLUSIVE.
  No exclusion on timing magnitude; only an operational failure (non-zero exit) voids a pair.

