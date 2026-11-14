#!/usr/bin/env bash
# ds_compare, parts 2 and 3 in one rented instance (one setup, one pull).
# Part 2 = Shapes B and C at 80 GB; part 3 = the 24 / 16 GB caps on Shape A. Both start from
# part 1's results, shipped in the payload. After both parts the job waits up to 20 minutes for
# /workspace/out/PULLED (written by the operator after copying the results over ssh), so the
# results are copied while the instance still runs; then it exits and the guard stops the box.
set -uo pipefail

SKIP_S0=1
JOB_BUDGET_S="${JOB_BUDGET_S:-11400}"
PAYLOAD_DIR="${PAYLOAD_DIR:-/workspace/payload}"
HARNESS="$PAYLOAD_DIR/benchmarks/harness/ds_compare"
[ -f "$HARNESS/job_common.sh" ] || { echo "[job23] ERROR: job_common.sh not found" >&2; exit 1; }
# shellcheck source=/dev/null
source "$HARNESS/job_common.sh"

OUT="${VK_OUT:-/workspace/out}"
mkdir -p "$OUT"
( PART=part2; job_main ); rc2=$?
echo "[job23 $(date -u +%TZ)] part2 rc=$rc2" | tee -a "$OUT/job23.txt"
( PART=part3; job_main ); rc3=$?
echo "[job23 $(date -u +%TZ)] part3 rc=$rc3" | tee -a "$OUT/job23.txt"
touch "$OUT/RUNS_DONE"
for _ in $(seq 1 40); do
    [ -e "$OUT/PULLED" ] && break
    sleep 30
done
echo "[job23 $(date -u +%TZ)] exit (pulled: $([ -e "$OUT/PULLED" ] && echo yes || echo no))" | tee -a "$OUT/job23.txt"
exit 0
