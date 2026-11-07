#!/usr/bin/env bash
# ds_compare, part 2 of 3, as a job of the self-stopping vast kit (one rented 1x H100 instance).
#
# Part 2 = Shapes B and C at 80 GB: arms SB, S1, D-best, D1, R1; repeat 1 of both shapes, then
# repeat 2. It needs part 1's results (SB overrides, D-best): ship them in the payload,
#
#   bash make_payload.sh part2 <out_dir> <part 1 results dir>
#   python vk.py launch ... --max-hours <dry_run_parts.py --part part2> \
#       --payload <out_dir> --job <out_dir>/benchmarks/harness/ds_compare/job_part2.sh
#
# Without them the job refuses before the set-up starts. Only part 1 runs arm S0, so the S0
# venv is skipped here. The kit runs this file as /root/vk/job.sh, so the harness is found in the
# payload, not next to this file. Everything else is in job_common.sh.
set -euo pipefail

PART=part2
SKIP_S0=1
# Used only when the kit offers no deadline: the kit's --max-hours of this part (2.25 h, from
# dry_run_parts.py) minus 15 minutes that pass before the job starts (boot, image, payload upload).
JOB_BUDGET_S="${JOB_BUDGET_S:-7200}"

PAYLOAD_DIR="${PAYLOAD_DIR:-/workspace/payload}"
HARNESS="$PAYLOAD_DIR/benchmarks/harness/ds_compare"
[ -f "$HARNESS/job_common.sh" ] || HARNESS="$PAYLOAD_DIR"
[ -f "$HARNESS/job_common.sh" ] || { echo "[job] ERROR: job_common.sh not found under $PAYLOAD_DIR" >&2; exit 1; }
# shellcheck source=job_common.sh
source "$HARNESS/job_common.sh"
job_main
