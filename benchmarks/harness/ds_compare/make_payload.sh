#!/usr/bin/env bash
# Build the payload folder of one part of the ds_compare series for the self-stopping vast kit.
#
#     bash make_payload.sh <part> <out_dir> [<results_dir_to_ship>]
#
#   <part>                 part1, part2 or part3
#   <out_dir>              the payload folder to create (it must not exist, or be empty)
#   <results_dir_to_ship>  the results folder of the earlier part(s): part 2 and part 3 need
#                          part 1's, a part 1 that continues an interrupted one may ship its own
#
# The harness files come from `git archive` of the committed tree (HEAD, or $PAYLOAD_REF), never
# from the working tree, with core.autocrlf off, so every file is LF only. What is shipped of the
# results is exactly what a later part reads: <shape>/<budget>/<arm>/rep<k>.json (the run jsons)
# and sb_overrides.json / d_best.json when they exist. The sampler CSVs, the console logs
# (progress bars write carriage returns) and the traces stay behind: nothing in a later part reads
# them, and they would break the rule below.
#
# The script refuses (and leaves no payload at <out_dir>) if
#   * the harness directory has uncommitted changes (ALLOW_DIRTY=1 overrides), because the payload
#     would not be what the owner is looking at;
#   * any file of the payload contains a carriage-return byte (a CR in a shell script breaks it);
#   * any file of the payload is over 50 MB (PAYLOAD_MAX_FILE_MB overrides);
#   * a results folder is given but holds no run json, or the part's own preflight
#     (orchestrate.py --part <part> --preflight) finds the earlier part's results incomplete.
#
# The payload also gets PAYLOAD.txt (part, commit, build time). __pycache__ is never included.
# Other overrides: PAYLOAD_REPO (a git repository other than this one), PAYLOAD_PYTHON.
set -euo pipefail

HARNESS_REL="benchmarks/harness/ds_compare"

die() { printf 'make_payload: ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    printf 'usage: bash make_payload.sh <part1|part2|part3> <out_dir> [<results_dir_to_ship>]\n' >&2
    exit 2
}

[ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage
PART="$1"
OUT="${2%/}"
SHIP="${3:-}"
SHIP="${SHIP%/}"
case "$PART" in part1 | part2 | part3) ;; *) die "the part must be part1, part2 or part3 (got $PART)" ;; esac
[ -n "$OUT" ] || die "the output folder is empty"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PAYLOAD_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}"
REF="${PAYLOAD_REF:-HEAD}"
MAX_MB="${PAYLOAD_MAX_FILE_MB:-50}"
case "$MAX_MB" in '' | *[!0-9]*) die "PAYLOAD_MAX_FILE_MB must be a whole number (got $MAX_MB)" ;; esac
COMMIT="$(git -C "$REPO" rev-parse --verify "$REF^{commit}")" || die "cannot resolve $REF in $REPO"

pick_python() {
    local candidate
    for candidate in "${PAYLOAD_PYTHON:-}" python3 python; do
        [ -n "$candidate" ] || continue
        if command -v "$candidate" > /dev/null 2>&1 && "$candidate" -c 'import sys' > /dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# --- the output folder -------------------------------------------------------------------
if [ -e "$OUT" ]; then
    [ -d "$OUT" ] || die "$OUT exists and is not a folder"
    [ -z "$(ls -A "$OUT")" ] || die "$OUT is not empty: give a new folder (a payload is never built over another)"
fi
if [ -n "$SHIP" ]; then
    [ -d "$SHIP" ] || die "the results folder to ship does not exist: $SHIP"
fi

# --- the harness must be what is committed -------------------------------------------------
if [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    if ! git -C "$REPO" diff --quiet "$REF" -- "$HARNESS_REL" \
        || [ -n "$(git -C "$REPO" ls-files --others --exclude-standard -- "$HARNESS_REL")" ]; then
        die "$HARNESS_REL has uncommitted changes (commit them first; ALLOW_DIRTY=1 builds from $REF anyway)"
    fi
fi

# --- build in a scratch folder next to the target; move it into place only when every check passed
mkdir -p "$(dirname "$OUT")"
STAGE="$(mktemp -d "$OUT.building.XXXXXX")"
refused() {
    printf 'make_payload: ERROR: %s\n' "$*" >&2
    printf 'make_payload: no payload was created; the unfinished build is in %s (inspect it, then delete it)\n' "$STAGE" >&2
    exit 1
}

git -c core.autocrlf=false -C "$REPO" archive "$COMMIT" "$HARNESS_REL" | (cd "$STAGE" && tar -x) \
    || refused "git archive of $HARNESS_REL at $COMMIT failed"
[ -f "$STAGE/$HARNESS_REL/orchestrate.py" ] || refused "$HARNESS_REL/orchestrate.py is not in $COMMIT"

# --- the earlier part's results: the run jsons and the two derivation files only --------------
SHIPPED=0
if [ -n "$SHIP" ]; then
    while IFS= read -r file; do
        relative="${file#"$SHIP"/}"
        # rep<k>.json only: not rep<k>.raw.json, rep<k>.void1.json, rep<k>.oom.json
        [[ "$(basename "$relative")" =~ ^rep[0-9]+\.json$ ]] || continue
        mkdir -p "$STAGE/results/$(dirname "$relative")"
        cp "$file" "$STAGE/results/$relative"
        SHIPPED=$((SHIPPED + 1))
    done < <(find "$SHIP" -mindepth 4 -maxdepth 4 -type f -name 'rep*.json' | sort)
    for derived in sb_overrides.json d_best.json; do
        if [ -f "$SHIP/$derived" ]; then cp "$SHIP/$derived" "$STAGE/results/$derived"; fi
    done
    [ "$SHIPPED" -gt 0 ] || refused "no run json (<shape>/<budget>/<arm>/rep<k>.json) under $SHIP: give the results folder itself, e.g. <pulled output>/results"
fi

{
    printf 'part: %s\n' "$PART"
    printf 'harness_commit: %s\n' "$COMMIT"
    printf 'built_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'shipped_run_jsons: %s\n' "$SHIPPED"
    if [ -n "$SHIP" ]; then printf 'shipped_from: %s\n' "$SHIP"; fi
} > "$STAGE/PAYLOAD.txt"

# --- the part's own preflight (stdlib only; no bytecode is written into the payload) -----------
PYTHON_BIN="$(pick_python)" || refused "no working python to run the preflight with (PAYLOAD_PYTHON)"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" "$STAGE/$HARNESS_REL/orchestrate.py" --part "$PART" --preflight --results "$STAGE/results" \
    || refused "$PART would refuse to start with these results (see above)"

# --- no CR byte anywhere, no file over the limit ---------------------------------------------
# a tracked __pycache__ (there should be none) must not ship
find "$STAGE" -type d -name __pycache__ -prune -exec rm -r {} +
# Counted with tr and wc: Git Bash's grep reads files as text and drops the CR it should find.
CR_FILES=""
while IFS= read -r file; do
    if [ "$(LC_ALL=C tr -cd '\r' < "$file" | wc -c | tr -d ' ')" -gt 0 ]; then
        CR_FILES="${CR_FILES}${file}"$'\n'
    fi
done < <(find "$STAGE" -type f | sort)
CR_FILES="${CR_FILES%$'\n'}"
if [ -n "$CR_FILES" ]; then
    refused "these files contain a carriage-return byte (payloads are LF only):
$(printf '%s\n' "$CR_FILES" | sed "s|^$STAGE/|  |")"
fi
BIG_FILES="$(find "$STAGE" -type f -size +"$((MAX_MB * 1024 * 1024))c")"
if [ -n "$BIG_FILES" ]; then
    refused "these files are over $MAX_MB MB:
$(printf '%s\n' "$BIG_FILES" | sed "s|^$STAGE/|  |")"
fi

# --- in place ----------------------------------------------------------------------------------
chmod 755 "$STAGE"
if [ -d "$OUT" ]; then rmdir "$OUT"; fi
mv "$STAGE" "$OUT"
FILES="$(find "$OUT" -type f | wc -l | tr -d ' ')"
KIB="$(du -sk "$OUT" | cut -f1)"
HOURS="$("$PYTHON_BIN" "$OUT/$HARNESS_REL/dry_run_parts.py" --part "$PART" --max-hours-only)"
printf 'payload for %s built in %s\n' "$PART" "$OUT"
printf '  harness: %s (git archive, LF only), %s files, %s KiB\n' "${COMMIT:0:12}" "$FILES" "$KIB"
printf '  shipped results: %s run jsons\n' "$SHIPPED"
printf '  suggested --max-hours: %s (dry_run_parts.py)\n' "$HOURS"
printf '  launch with: --payload %s --job %s/%s/job_%s.sh --max-hours %s\n' "$OUT" "$OUT" "$HARNESS_REL" "$PART" "$HOURS"
