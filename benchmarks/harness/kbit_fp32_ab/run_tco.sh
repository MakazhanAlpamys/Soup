#!/usr/bin/env bash
# run_tco.sh - one command per (case, framework) on the RENTED box.  Pre-registration: protocol.md
#
#   bash run_tco.sh <case> <framework|all> [--repeats N] [--mode window|epoch] [--card "RTX 4090"]
#                   [--price-json price.json | --usd-per-hour X --price-source "..." --price-date UTC]
#                   [--actual-usd-per-hour X] [--synthetic]
#   bash run_tco.sh plan <case> <framework>      # print the plan, install nothing, touch nothing
#   bash run_tco.sh setup                         # only the pinned install
#   bash run_tco.sh status                        # what is running / what is on disk
#
#   case:      smoke | 1 | 2a | 2b          framework: soup | hf | unsloth | all
#
# What it does:  preflight -> pinned install (once) -> data -> N measured repeats -> summary.
# What it never does:  start, stop or destroy the instance; read or store any API key.  Teardown is
# the operator's step (protocol.md section 10).  It detaches itself (setsid + nohup) so that a
# dropped ssh session or a closed laptop does not kill the run; every run writes append-only files
# as it goes, so `pull` after a stop still returns partial data.  TCO_FG=1 keeps it in the foreground.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TCO_OUT="${TCO_OUT:-/workspace/tco-out}"
TCO_VENV="${TCO_VENV:-$TCO_OUT/venv}"
PYBIN="${TCO_PYTHON:-python3}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PIN="torch==2.11.0"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-}"
case "$cmd" in
  ""|-h|--help|help) usage; exit 0 ;;
  plan)
    shift
    # plan needs no GPU and no venv: pure argument handling and arithmetic
    exec "$PYBIN" -I "$HERE/tco_run.py" plan "$@" ;;
  status)
    echo "TCO_OUT=$TCO_OUT"
    [ -f "$TCO_OUT/run.pid" ] && { pid="$(cat "$TCO_OUT/run.pid")"; kill -0 "$pid" 2>/dev/null && echo "running: pid $pid" || echo "not running (pid $pid gone)"; }
    [ -f "$TCO_OUT/progress.log" ] && tail -n 15 "$TCO_OUT/progress.log"
    find "$TCO_OUT/runs" -name result.json 2>/dev/null | sort
    exit 0 ;;
esac

# ---- detach unless asked not to --------------------------------------------------------------
if [ "${TCO_FG:-0}" != "1" ] && [ -z "${TCO_DETACHED:-}" ]; then
  mkdir -p "$TCO_OUT"
  export TCO_DETACHED=1
  setsid nohup bash "$0" "$@" > "$TCO_OUT/run.log" 2>&1 < /dev/null &
  echo $! > "$TCO_OUT/run.pid"
  say "detached as pid $(cat "$TCO_OUT/run.pid"); follow with:  tail -f $TCO_OUT/run.log"
  say "results appear under $TCO_OUT/runs/<case>/<framework>/<card>/rep<k>/ (partial files are valid)"
  exit 0
fi

mkdir -p "$TCO_OUT"
PROGRESS="$TCO_OUT/progress.log"
log() { say "$*" | tee -a "$PROGRESS"; }

# ---- setup ----------------------------------------------------------------------------------
do_setup() {
  if [ "${TCO_SKIP_SETUP:-0}" = "1" ]; then log "setup: SKIPPED (TCO_SKIP_SETUP=1, testing only; versions are checked against the pins per run)"; TCO_VENV=""; return 0; fi
  if [ -f "$TCO_OUT/setup.done" ]; then log "setup: already done ($(cat "$TCO_OUT/setup.done"))"; return 0; fi
  local t0 t1; t0=$(date +%s)
  "$PYBIN" - <<'EOF' || { log "FATAL: python must be 3.11-3.13"; return 1; }
import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)
EOF
  "$PYBIN" -m venv "$TCO_VENV" || return 1
  local pip="$TCO_VENV/bin/python -m pip"
  $pip install --quiet --upgrade pip wheel || return 1
  log "setup: torch from $TORCH_INDEX"
  $pip install --quiet "$TORCH_PIN" --index-url "$TORCH_INDEX" || return 1
  local torch_installed; torch_installed="$($TCO_VENV/bin/python -c 'import torch;print(torch.__version__)')"
  printf 'torch==%s\n' "$torch_installed" > "$TCO_OUT/constraints.txt"
  log "setup: pinned stack (requirements-tco.txt) with torch held at $torch_installed"
  $pip install --quiet -c "$TCO_OUT/constraints.txt" --extra-index-url "$TORCH_INDEX" \
      -r "$HERE/requirements-tco.txt" || return 1
  $pip check > "$TCO_OUT/pip-check.txt" 2>&1 || log "setup: pip check reported problems (see pip-check.txt); drift is recorded in every result.json"
  $pip freeze > "$TCO_OUT/freeze.txt"
  t1=$(date +%s)
  echo "$((t1 - t0)) s" > "$TCO_OUT/setup.done"
  printf '{"setup_s": %s}\n' "$((t1 - t0))" > "$TCO_OUT/setup.json"
  log "setup: done in $((t1 - t0)) s"
}

if [ "$cmd" = "setup" ]; then do_setup; exit $?; fi

# ---- run --------------------------------------------------------------------------------------
case_id="$1"; fw_arg="${2:-}"; shift 2 || { usage; exit 2; }
repeats=""; mode="window"; passthru=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repeats) repeats="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    *) passthru+=("$1"); shift ;;
  esac
done
case "$case_id" in smoke|1|2a|2b) ;; *) echo "unknown case '$case_id'"; usage; exit 2 ;; esac

do_setup || { log "FATAL: setup failed"; exit 1; }
if [ -n "$TCO_VENV" ]; then PY="$TCO_VENV/bin/python"; else PY="$PYBIN"; fi   # TCO_SKIP_SETUP: current interpreter
setup_s="$(cat "$TCO_OUT/setup.json" 2>/dev/null | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["setup_s"])' 2>/dev/null || echo "")"
[ -n "$setup_s" ] && passthru+=(--setup-s "$setup_s")

# frameworks and the default repeat count come from tco_common.CASES (one source of truth)
read -r default_reps fws < <(TCO_HERE="$HERE" "$PY" - "$case_id" <<'EOF'
import os, sys
sys.path.insert(0, os.environ["TCO_HERE"])
import tco_common as C
c = C.CASES[sys.argv[1]]
print(c["repeats"], ",".join(c["frameworks"]))
EOF
)
repeats="${repeats:-$default_reps}"
if [ "$fw_arg" = "all" ]; then IFS=',' read -r -a fw_list <<< "$fws"; else fw_list=("$fw_arg"); fi

log "case=$case_id frameworks=${fw_list[*]} repeats=$repeats mode=$mode TCO_OUT=$TCO_OUT"
if [ "${TCO_SKIP_PREFLIGHT:-0}" = "1" ]; then
  log "preflight: SKIPPED (TCO_SKIP_PREFLIGHT=1, testing only; a real run must never set this)"
else
  log "preflight (a box that fails it is not a valid host; destroy it and rent another)"
  "$PY" "$HERE/tco_run.py" preflight "$case_id" "${fw_list[0]}" >> "$PROGRESS" 2>&1 \
    || { log "PREFLIGHT FAILED: see $TCO_OUT/preflight-$case_id-${fw_list[0]}.json"; exit 3; }
fi

# Order inside a repeat: Soup FIRST (its recorded batch order is the reference the other arms replay,
# protocol.md section 7 rule C1), then the remaining arms rotated by repeat so none always runs last.
order_for_repeat() {
  local rep=$1 f; local -a others=()
  for f in "${fw_list[@]}"; do if [ "$f" = "soup" ]; then echo soup; else others+=("$f"); fi; done
  local n=${#others[@]} i
  for ((i = 0; i < n; i++)); do echo "${others[$(((i + rep - 1) % n))]}"; done
}

failed=0
for ((rep = 1; rep <= repeats; rep++)); do
  for fw in $(order_for_repeat "$rep"); do
    log "RUN case=$case_id framework=$fw repeat=$rep/$repeats"
    if "$PY" "$HERE/tco_run.py" run "$case_id" "$fw" --repeat "$rep" --mode "$mode" "${passthru[@]}" >> "$PROGRESS" 2>&1; then
      :
    else
      failed=$((failed + 1)); log "run exited non-zero (kept: its result.json says why)"
    fi
  done
done

"$PY" "$HERE/summarize.py" "$TCO_OUT/runs" --out "$TCO_OUT/summary" >> "$PROGRESS" 2>&1 || true
log "DONE ($failed run(s) exited non-zero). Summary: $TCO_OUT/summary.md"
log "OPERATOR NEXT: pull $TCO_OUT (result.json, steps.jsonl, logs), then destroy the instance and log the cost: protocol.md section 10"
