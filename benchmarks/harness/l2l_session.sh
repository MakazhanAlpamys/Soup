#!/usr/bin/env bash
# L2L step 0 — the four blocks of benchmarks/probe-rtx5070-l2l-step0.md §1,
# one process each, every log to its own file. Run from the worktree root, on
# AC with the lid open, after every live session granted the hold.
set -u
cd "$(dirname "$0")/../.."
OUT=benchmarks/results/probe-rtx5070/l2l
LOG="$OUT/logs"
mkdir -p "$LOG"
PY="${PYTHON:-python}"
export PYTHONPATH="$(pwd -W 2>/dev/null || pwd)/src"
H=benchmarks/harness/l2l_probe.py
F1=unsloth/mistral-7b-instruct-v0.3
F2=D:/synth/llama-70b-shape-v32k-f4

"$PY" -c "import soup_cli; print(soup_cli.__file__)" | tee "$LOG/soup_cli_file.txt"
case "$(cat "$LOG/soup_cli_file.txt")" in
  *Soup-l2l*) ;;
  *) echo "REFUSED: soup_cli does not resolve under Soup-l2l"; exit 2 ;;
esac

a_vs_a_exact() {
  "$PY" -c "import json,sys; r=json.load(open(sys.argv[1]))['records'][0]; \
print('yes' if r.get('a_vs_a',{}).get('equal') and r.get('a_vs_a_losses_equal') else 'no')" "$1"
}

run_correctness() {  # $1 = f1|f2, $2 = weights, $3 = tier, $4 = k
  "$PY" "$H" --mode correctness --label "${1^^}" --weights "$2" --tier "$3" --k "$4" \
    --out "$OUT/correctness_$1.json" > "$LOG/correctness_$1.log" 2>&1
  if [ "$(a_vs_a_exact "$OUT/correctness_$1.json")" != "yes" ]; then
    "$PY" "$H" --mode correctness --label "${1^^}" --weights "$2" --tier "$3" --k "$4" \
      --deterministic --out "$OUT/correctness_$1_det.json" > "$LOG/correctness_$1_det.log" 2>&1
  fi
}

date -u +"block 1 start %FT%TZ" | tee -a "$LOG/session.txt"
run_correctness f1 "$F1" ram 4
date -u +"block 2 start %FT%TZ" | tee -a "$LOG/session.txt"
run_correctness f2 "$F2" disk 2
date -u +"block 3 start %FT%TZ" | tee -a "$LOG/session.txt"
"$PY" "$H" --mode timing --label F2 --weights "$F2" --tier disk \
  --out "$OUT/timing.json" > "$LOG/timing.log" 2>&1
if [ $? -eq 3 ]; then
  echo "STOPPED: block 3 refused at its start (V1/V3), see $LOG/timing.log" | tee -a "$LOG/session.txt"
  exit 3
fi
date -u +"block 4 start %FT%TZ" | tee -a "$LOG/session.txt"
"$PY" "$H" --mode spill --label F2 --weights "$F2" --tier disk \
  --out "$OUT/spill.json" > "$LOG/spill.log" 2>&1
if [ $? -eq 3 ]; then
  echo "STOPPED: block 4 refused at its start (V1/V3), see $LOG/spill.log" | tee -a "$LOG/session.txt"
  exit 3
fi
date -u +"done %FT%TZ" | tee -a "$LOG/session.txt"
"$PY" benchmarks/harness/l2l_rule.py "$OUT" | tee "$OUT/verdict.md"
