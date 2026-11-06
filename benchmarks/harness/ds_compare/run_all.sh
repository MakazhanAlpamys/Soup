#!/usr/bin/env bash
# One command repeats the whole ds_compare series on a prepared H100 box:
#
#     bash setup.sh && bash run_all.sh --all
#
#   --all              run everything (resumes: a finished run's json is not run again)
#   --dry-run          print every command, write every soup.yaml, run nothing
#   --only S1,D1,...   restrict to these arm ids (SB and DBEST are valid ids)
#
# Environment:
#   DEADLINE_EPOCH            unix time after which no run may start; the budget rule drops
#                             groups in protocol order when the rest does not fit
#   BUDGET_USD, RENTAL_RATE_USD_PER_HOUR (alias VAST_RATE_USD_PER_HOUR)
#                             optional cost cap (same drop rule) and the cost log
#   DS_COMPARE_RESULTS        results directory (default: ./results next to this script)
#   DS_COMPARE_MODEL_DIR      local copy of the pinned model (default: ./model)
#   BENCH_GPU                 GPU index (default 0)
#
# The planning, ordering, derivations, void checks and budget rule live in orchestrate.py
# (a script in bash would be untestable); this file only prepares the environment. Every
# run writes results/<shape>/<budget_gb>/<arm>/rep<k>.json and its sampler CSVs as soon as it
# ends; results/DONE is written last.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment written by setup.sh (PATH with the venv, CUDA_HOME, SOUP_S0_BIN, SETUP_SECONDS).
if [ -f "$HERE/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$HERE/env.sh"
fi

pick_python() {
    local candidate
    for candidate in "${PY:-}" python3 python; do
        [ -n "$candidate" ] || continue
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import sys" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    echo "no working python found" >&2
    return 1
}
PY="$(pick_python)"

# These must not leak into any arm: S1 means "whatever soup train exports itself", S5 sets the
# opposite explicitly, and the cap / trace variables are set per run by orchestrate.py.
unset PYTORCH_CUDA_ALLOC_CONF SOUP_BENCH_VRAM_CAP_GB SOUP_BENCH_TRACE_OUT
# The series never uses a Hugging Face token.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN

export DS_COMPARE_RESULTS="${DS_COMPARE_RESULTS:-$HERE/results}"
export DS_COMPARE_MODEL_DIR="${DS_COMPARE_MODEL_DIR:-$HERE/model}"
if [ -z "${SETUP_SECONDS:-}" ] && [ -f "$DS_COMPARE_RESULTS/setup_seconds.txt" ]; then
    SETUP_SECONDS="$(cat "$DS_COMPARE_RESULTS/setup_seconds.txt")"
    export SETUP_SECONDS
fi
export TOKENIZERS_PARALLELISM=false

exec "$PY" "$HERE/orchestrate.py" "$@"
