#!/usr/bin/env bash
# Shared by job_part1.sh, job_part2.sh and job_part3.sh. Each of them sets PART, SKIP_S0 and
# JOB_BUDGET_S, finds this file and calls job_main, so the three cannot drift apart.
#
# What the self-stopping vast kit gives a job: the job script runs as /root/vk/job.sh with the
# shipped payload unpacked in /workspace/payload (the working directory); $VK_OUT (/workspace/out)
# is tarred and pulled back when the job ends; the variables of --env are set; and two files
# exist on the instance: /root/vk/deadline (the job's deadline as unix time, rewritten when the
# owner extends it) and /root/vk/vk.env (VK_DEADLINE_EPOCH, VK_RESERVE_S, VK_LAUNCH_EPOCH). The
# kit does NOT export VK_DEADLINE_EPOCH into the job's environment, so job_resolve_deadline
# reads the variable if the owner passed one with --env and otherwise those two files. At
# deadline - VK_RESERVE_S the kit kills the job.
#
# job_main, in order:
#   1. log to $VK_OUT/job_<part>.log, cd to the payload;
#   2. DS_COMPARE_RESULTS=$VK_OUT/results, then copy the shipped results (payload/results) into it;
#   3. preflight: a part that needs an earlier part's results refuses HERE, before any set-up
#      time is spent;
#   4. DEADLINE_EPOCH (no new run starts after it): the kit's deadline minus its reserve minus a
#      10-minute finish margin; without a kit deadline, the job start + JOB_BUDGET_S minus the
#      same margin;
#   5. setup.sh (SETUP_PART=<part>, SKIP_S0 as the job script says), then run_all.sh --part <part>.
#
# Testing without a rented box: PAYLOAD_DIR, VK_OUT, VK_ROOT (a folder standing in for
# /root/vk), JOB_PYTHON (the interpreter) and JOB_DRY_RUN=1 (log the two last commands instead
# of running them) are honoured.

JOB_FINISH_MARGIN_S="${JOB_FINISH_MARGIN_S:-600}"

job_log() { printf '[job %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
job_die() { printf '[job] ERROR: %s\n' "$*" >&2; exit 1; }

# The digits of KEY=value in a kit env file (nothing when the file or the key is missing).
job_env_file_value() {
    local file="$1" key="$2"
    [ -r "$file" ] || return 0
    sed -n "s/^${key}=//p" "$file" | tail -n 1 | tr -cd '0-9'
}

# Sets DEADLINE_EPOCH (exported) and DEADLINE_SOURCE. Needs JOB_START_EPOCH and JOB_BUDGET_S.
job_resolve_deadline() {
    local kit_root="${VK_ROOT:-/root/vk}" kit_deadline="" reserve="" now
    now="$(date +%s)"
    DEADLINE_SOURCE=""
    if [ -n "${VK_DEADLINE_EPOCH:-}" ]; then
        kit_deadline="$VK_DEADLINE_EPOCH"
        DEADLINE_SOURCE="environment VK_DEADLINE_EPOCH"
    elif [ -r "$kit_root/deadline" ]; then
        kit_deadline="$(tr -cd '0-9' < "$kit_root/deadline")"
        DEADLINE_SOURCE="$kit_root/deadline"
    elif [ -r "$kit_root/vk.env" ]; then
        kit_deadline="$(job_env_file_value "$kit_root/vk.env" VK_DEADLINE_EPOCH)"
        DEADLINE_SOURCE="$kit_root/vk.env"
    fi
    case "$kit_deadline" in *[!0-9]*) kit_deadline="" ;; esac
    if [ -n "$kit_deadline" ]; then
        reserve="${VK_RESERVE_S:-$(job_env_file_value "$kit_root/vk.env" VK_RESERVE_S)}"
        case "$reserve" in '' | *[!0-9]*) reserve=600 ;; esac
        [ "$kit_deadline" -gt "$now" ] || job_die "the kit deadline ($kit_deadline) is already past (now $now)"
        DEADLINE_EPOCH=$((kit_deadline - reserve - JOB_FINISH_MARGIN_S))
        DEADLINE_SOURCE="$DEADLINE_SOURCE ($kit_deadline) - kit reserve ${reserve} s - finish margin ${JOB_FINISH_MARGIN_S} s"
    else
        DEADLINE_EPOCH=$((JOB_START_EPOCH + JOB_BUDGET_S - JOB_FINISH_MARGIN_S))
        DEADLINE_SOURCE="no kit deadline found: job start + JOB_BUDGET_S ${JOB_BUDGET_S} s - finish margin ${JOB_FINISH_MARGIN_S} s"
    fi
    export DEADLINE_EPOCH
}

# The first working python (the image's python3 before set-up; orchestrate.py is stdlib only).
job_pick_python() {
    local candidate
    for candidate in "${JOB_PYTHON:-}" python3 python; do
        [ -n "$candidate" ] || continue
        if command -v "$candidate" > /dev/null 2>&1 && "$candidate" -c 'import sys' > /dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Copy the shipped results into this instance's results folder; never overwrite a file this
# instance already wrote (the job may be started again after a container restart).
job_copy_shipped_results() {
    local source="$1" target="$2" count
    mkdir -p "$target"
    if [ ! -d "$source" ]; then
        job_log "no shipped results ($source does not exist)"
        return 0
    fi
    cp -a -n "$source/." "$target/" || job_log "note: cp reported a file that already exists in $target (kept)"
    count="$(find "$source" -type f -name 'rep*.json' | wc -l | tr -d ' ')"
    job_log "shipped results copied into $target: $count run jsons"
}

job_main() {
    : "${PART:?PART is not set}" "${JOB_BUDGET_S:?JOB_BUDGET_S is not set}" "${HARNESS:?HARNESS is not set}"
    case "$PART" in part1 | part2 | part3) ;; *) job_die "PART must be part1, part2 or part3 (got $PART)" ;; esac
    case "$JOB_BUDGET_S" in '' | *[!0-9]*) job_die "JOB_BUDGET_S must be whole seconds (got $JOB_BUDGET_S)" ;; esac
    local out="${VK_OUT:-/workspace/out}" payload="${PAYLOAD_DIR:-/workspace/payload}" python_bin
    JOB_START_EPOCH="$(date +%s)"
    mkdir -p "$out"
    exec > >(tee -a "$out/job_${PART}.log") 2>&1
    job_log "$PART starts; payload $payload, output $out"
    cd "$payload" || job_die "the payload folder $payload does not exist"

    export DS_COMPARE_RESULTS="$out/results"
    job_copy_shipped_results "$payload/results" "$DS_COMPARE_RESULTS"
    if [ -f "$payload/PAYLOAD.txt" ]; then cp "$payload/PAYLOAD.txt" "$out/payload_${PART}.txt"; fi

    python_bin="$(job_pick_python)" || job_die "no working python found"
    "$python_bin" "$HARNESS/orchestrate.py" --part "$PART" --preflight --results "$DS_COMPARE_RESULTS" \
        || job_die "$PART refused to start; no set-up time was spent (ship the earlier parts' results with the payload)"

    job_resolve_deadline
    job_log "DEADLINE_EPOCH=$DEADLINE_EPOCH ($(date -u -d "@$DEADLINE_EPOCH" +%H:%M:%SZ 2> /dev/null || echo '?')); source: $DEADLINE_SOURCE"
    export SETUP_PART="$PART" SKIP_S0

    if [ "${JOB_DRY_RUN:-0}" = "1" ]; then
        job_log "dry run: would run: bash $HARNESS/setup.sh (SETUP_PART=$PART SKIP_S0=$SKIP_S0) && bash $HARNESS/run_all.sh --part $PART"
        return 0
    fi
    bash "$HARNESS/setup.sh"
    bash "$HARNESS/run_all.sh" --part "$PART"
    job_log "$PART finished"
}
