#!/usr/bin/env bash
# ds_compare, part 1 of 3, as a job of the self-stopping vast kit (one rented 1x H100 instance).
#
# Part 1 = Shape A at 80 GB, every arm of protocol section 2 except S0 (DEVIATIONS.md D-2):
# S1..S8, SB (derived, then run), D1..D4, R1, R2, the prefetch variants by their rule, the two
# warm-ups, 3 repeats. It holds G1, the ablation and the G2 / G3 values at 80 GB. Part 2 and
# part 3 start from its results.
#
#   bash make_payload.sh part1 <out_dir>
#   python vk.py launch ... --max-hours <dry_run_parts.py --part part1> \
#       --payload <out_dir> --job <out_dir>/benchmarks/harness/ds_compare/job_part1.sh
#
# The kit runs this file as /root/vk/job.sh, so the harness is found in the payload, not next to
# this file. Everything else (results folder, deadline, set-up, run) is in job_common.sh.
set -euo pipefail

PART=part1
SKIP_S0=1  # S0 is out of the plan (DEVIATIONS.md D-2)
# Used only when the kit offers no deadline: the kit's --max-hours of this part (4.75 h, from
# dry_run_parts.py) minus 15 minutes that pass before the job starts (boot, image, payload upload).
JOB_BUDGET_S="${JOB_BUDGET_S:-16200}"

PAYLOAD_DIR="${PAYLOAD_DIR:-/workspace/payload}"
HARNESS="$PAYLOAD_DIR/benchmarks/harness/ds_compare"
[ -f "$HARNESS/job_common.sh" ] || HARNESS="$PAYLOAD_DIR"
[ -f "$HARNESS/job_common.sh" ] || { echo "[job] ERROR: job_common.sh not found under $PAYLOAD_DIR" >&2; exit 1; }
# shellcheck source=job_common.sh
source "$HARNESS/job_common.sh"
job_main
