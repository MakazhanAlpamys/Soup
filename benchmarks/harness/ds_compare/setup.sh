#!/usr/bin/env bash
# Prepare a fresh rented box for the ds_compare series (about one hour, mostly downloads and
# the DeepSpeed build).
#
#     bash setup.sh && bash run_all.sh --all
#
# Running the series in parts (README.md "Running in parts"): every part is its own rented
# instance and runs this script once. Two switches:
#     SETUP_PART=part1|part2|part3   write this instance's records under results/setup_<part>/
#                                    instead of results/ (nothing of an earlier part is overwritten)
#     SKIP_S0=1                      skip step 4, the S0 venv (no arm of the plan uses it: D-2)
# Every other step stays: the hardware record, the cpu_adam build and its log, the model and its
# hashes, the dataset check, the environment record.
#
# Target: a rented PyTorch + CUDA *devel* image (it must have nvcc: DeepSpeed's cpu_adam is
# compiled here) with Python 3.11 to 3.13 and torch >= 2.6 already installed. The image's own
# torch is reused; both virtual environments are created with --system-site-packages so a
# 2 GB torch wheel is not downloaded twice.
#
# What it does, in order (everything it records goes to results/):
#   1. hardware.txt   nvidia-smi, lscpu, free -g, ulimit -l, df -h, disk type, nvcc, OS
#   2. main venv      soup-cli (train extras) from the git commit in $SOUP_COMMIT
#   3. DeepSpeed      built with DS_BUILD_CPU_ADAM=1, build log kept; cpu_adam import verified
#   4. S0 venv        soup-cli at commit 2c9a078 with the stack of the earlier record
#                     (skipped with SKIP_S0=1)
#   5. model          NousResearch/Meta-Llama-3.1-8B-Instruct at the pinned revision, no token;
#                     sha256 of every downloaded file -> results/model_hashes.txt
#   6. datasets       make_rows.py --check against the model's tokenizer
#   7. env.txt        pip freeze of both environments, versions, protocol sha256
#
# No Hugging Face token is read, used or written anywhere in this file.
# Never run with `set -x`: it would echo the environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${DS_COMPARE_RESULTS:-$HERE/results}"
MODEL_DIR="${DS_COMPARE_MODEL_DIR:-$HERE/model}"
SETUP_PART="${SETUP_PART:-}"
SKIP_S0="${SKIP_S0:-0}"
case "$SETUP_PART" in
    "" | part1 | part2 | part3) ;;
    *) printf '[setup] ERROR: SETUP_PART must be part1, part2 or part3 (got %s)\n' "$SETUP_PART" >&2; exit 1 ;;
esac
case "$SKIP_S0" in
    0 | 1) ;;
    *) printf '[setup] ERROR: SKIP_S0 must be 0 or 1 (got %s)\n' "$SKIP_S0" >&2; exit 1 ;;
esac
# REC: where this instance's own records go (results/ itself when no part is named).
REC="$RESULTS"
if [ -n "$SETUP_PART" ]; then REC="$RESULTS/setup_$SETUP_PART"; fi
mkdir -p "$RESULTS" "$REC"
exec > >(tee -a "$REC/setup.log") 2>&1

SETUP_T0="$(date +%s)"
# soup-cli commit under test: origin/main when this harness was written (override with
# SOUP_COMMIT=<sha or ref>). The record states the commit it actually ran.
SOUP_COMMIT="${SOUP_COMMIT:-9ca9819747a624da4cbbff03cc8e1a7d2a81100d}"
SOUP_OLD_COMMIT="${SOUP_OLD_COMMIT:-2c9a078}"
SOUP_REPO_URL="${SOUP_REPO_URL:-https://github.com/MakazhanAlpamys/Soup.git}"
MODEL_ID="NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_REVISION="d10aef7999a2b5ba950ab3974312feeedbfe0b77"
# DEEPSPEED_SPEC may pin a version, e.g. "deepspeed==0.19.4" (the version of the earlier record).
DEEPSPEED_SPEC="${DEEPSPEED_SPEC:-deepspeed}"
# Stack of the earlier record for arm S0 (H100 record, STEP 3 section "The installed stack").
S0_PINS=("transformers==4.57.6" "trl==0.26.2" "peft==0.20.0" "accelerate==1.14.0")

log() { printf '[setup %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# This series never uses a Hugging Face token; make sure nothing picks one up by accident.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# --- 1. hardware ------------------------------------------------------------------------
log "recording the hardware"
{
    echo "## date";            date -u +%Y-%m-%dT%H:%M:%SZ
    echo "## uname";           uname -a
    echo "## os-release";      cat /etc/os-release 2>/dev/null || true
    echo "## nvidia-smi";      nvidia-smi || echo "nvidia-smi failed"
    echo "## nvidia-smi query"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.sm,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current --format=csv || true
    echo "## nvidia-smi topo"; nvidia-smi topo -m 2>/dev/null || true
    echo "## nvcc";            nvcc --version 2>&1 || echo "nvcc NOT FOUND"
    echo "## lscpu";           lscpu
    echo "## free -g";         free -g
    echo "## ulimit -l (KB)";  ulimit -l
    echo "## df -h";           df -h
    echo "## block devices (ROTA=1 means the guest reports a rotational disk)"
    lsblk -d -o NAME,ROTA,TYPE,SIZE,MODEL 2>/dev/null || true
    for queue in /sys/block/*/queue/rotational; do echo "$queue: $(cat "$queue")"; done 2>/dev/null || true
    echo "## root mount";      mount | grep ' / ' || true
    echo "## python";          command -v python3 || true; python3 --version 2>&1 || true
    echo "## g++";             g++ --version 2>&1 | head -1 || echo "g++ NOT FOUND"
} > "$REC/hardware.txt" 2>&1 || true
command -v nvcc >/dev/null 2>&1 || die "nvcc not found: use a CUDA *devel* image (DeepSpeed cpu_adam needs it)"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
PINNED_KB="$(ulimit -l)"
if [ "$PINNED_KB" != "unlimited" ] && [ "$PINNED_KB" -lt 33554432 ]; then
    log "WARNING: ulimit -l is $PINNED_KB KB (< 32 GB): the bf16 store (16 GB) may not be pinned and"
    log "         the streaming arms would then run from pageable memory. Recorded; not raised."
fi

# --- python -----------------------------------------------------------------------------
PYBIN=""
for candidate in "${PYBIN_OVERRIDE:-}" python3.12 python3.11 python3.13 python3; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null; then
        PYBIN="$(command -v "$candidate")"
        break
    fi
done
[ -n "$PYBIN" ] || die "no Python 3.11 to 3.13 found (soup-cli requires it)"
log "python: $PYBIN ($("$PYBIN" --version 2>&1))"
"$PYBIN" - <<'PY' || die "the image's torch is missing or older than 2.6 (soup-cli requires >= 2.6): use a newer PyTorch devel image"
import sys
import torch
major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
sys.exit(0 if (major, minor) >= (2, 6) and torch.cuda.is_available() else 1)
PY

# CUDA_HOME for the DeepSpeed build and its runtime JIT.
if [ -z "${CUDA_HOME:-}" ]; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export CUDA_HOME
log "CUDA_HOME=$CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | tail -1))"

# --- 2. main venv + soup-cli ------------------------------------------------------------
VENV_MAIN="$HERE/venv_main"
if [ ! -x "$VENV_MAIN/bin/python" ]; then
    log "creating $VENV_MAIN"
    "$PYBIN" -m venv --system-site-packages "$VENV_MAIN"
fi
# shellcheck disable=SC1091
source "$VENV_MAIN/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel
python -m pip install --quiet psutil pyyaml ninja packaging hjson py-cpuinfo
log "installing soup-cli [train] at $SOUP_COMMIT"
python -m pip install "soup-cli[train] @ git+${SOUP_REPO_URL}@${SOUP_COMMIT}"
python - <<'PY'
import importlib.metadata as metadata
import json
direct = metadata.distribution("soup-cli").read_text("direct_url.json")
print("soup-cli", metadata.version("soup-cli"), (json.loads(direct).get("vcs_info") or {}).get("commit_id") if direct else None)
PY

# --- 3. DeepSpeed with cpu_adam ---------------------------------------------------------
log "building ${DEEPSPEED_SPEC} with DS_BUILD_CPU_ADAM=1 (log: $REC/deepspeed_build.log)"
if ! DS_BUILD_CPU_ADAM=1 python -m pip install "$DEEPSPEED_SPEC" --no-cache-dir --no-build-isolation -v \
    > "$REC/deepspeed_build.log" 2>&1; then
    tail -40 "$REC/deepspeed_build.log"
    die "the DeepSpeed build failed (full log: $REC/deepspeed_build.log)"
fi
python -c "import torch; import deepspeed; from deepspeed.ops.adam import DeepSpeedCPUAdam; DeepSpeedCPUAdam([torch.nn.Parameter(torch.zeros(1))]); print('deepspeed', deepspeed.__version__, 'cpu_adam OK')" \
    || die "DeepSpeedCPUAdam could not be created: the cpu_adam build is not usable"
command -v deepspeed >/dev/null 2>&1 || die "the deepspeed launcher is not on PATH"

# --- 4. S0 venv (earlier revision and its stack) ----------------------------------------
VENV_S0="$HERE/venv_s0"
if [ "$SKIP_S0" = "1" ]; then
    log "SKIP_S0=1: the S0 venv is not built (arm S0 is not in the plan)"
else
    if [ ! -x "$VENV_S0/bin/python" ]; then
        log "creating $VENV_S0"
        "$PYBIN" -m venv --system-site-packages "$VENV_S0"
    fi
    "$VENV_S0/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
    "$VENV_S0/bin/python" -m pip install --quiet psutil pyyaml
    log "installing soup-cli [train] at $SOUP_OLD_COMMIT with ${S0_PINS[*]}"
    "$VENV_S0/bin/python" -m pip install "soup-cli[train] @ git+${SOUP_REPO_URL}@${SOUP_OLD_COMMIT}" "${S0_PINS[@]}"
    "$VENV_S0/bin/soup" --help >/dev/null || die "the S0 venv's soup does not start"
    "$VENV_S0/bin/python" -m pip freeze > "$REC/env_s0.txt"
fi

# --- 5. model (no token) ----------------------------------------------------------------
log "downloading $MODEL_ID @ ${MODEL_REVISION:0:12} to $MODEL_DIR"
python - "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR" <<'PY'
import sys

from huggingface_hub import HfApi, snapshot_download

repo, revision, target = sys.argv[1:4]
resolved = HfApi().model_info(repo, revision=revision, token=False).sha
if resolved != revision:
    raise SystemExit(f"revision {revision} resolves to {resolved}: refusing to continue")
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=target,
    token=False,
    allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    ignore_patterns=["original/*"],
)
print("downloaded", repo, revision)
PY
{
    echo "# $MODEL_ID  revision $MODEL_REVISION"
    echo "# sha256  bytes  file   (every file under the model directory; written $(date -u +%Y-%m-%dT%H:%M:%SZ))"
    ( cd "$MODEL_DIR" && find . -path ./.cache -prune -o -type f -print | sort | while read -r file; do
        printf '%s  %s  %s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(stat -c %s "$file")" "${file#./}"
    done )
    echo
    echo "# files compared by hand against the published revision:"
    ( cd "$MODEL_DIR" && for file in config.json tokenizer.json tokenizer_config.json generation_config.json; do
        printf '%s  %s  %s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(stat -c %s "$file")" "$file"
    done )
} > "$REC/model_hashes.txt"
ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || die "no safetensors shards in $MODEL_DIR"
[ "$(ls "$MODEL_DIR"/model-*.safetensors | wc -l)" = "4" ] || die "expected 4 safetensors shards"
log "model_hashes.txt written ($REC)"

# --- 6. datasets ------------------------------------------------------------------------
log "checking the three dataset files against the model's tokenizer"
python "$HERE/make_rows.py" --check --tokenizer "$MODEL_DIR" || die "the dataset files do not tokenize to their target lengths"

# --- 7. environment record --------------------------------------------------------------
python -m pip freeze > "$REC/env.txt"
{
    echo "## versions"
    python - <<'PY'
import importlib.metadata as metadata
for name in ("torch", "transformers", "peft", "accelerate", "deepspeed", "trl", "datasets", "soup-cli"):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, "not installed")
import torch
print("cuda runtime", torch.version.cuda, "| device", torch.cuda.get_device_name(0))
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as MAPPING
print("peft default LoRA targets for llama:", MAPPING["llama"])
PY
    echo "## nvcc"; nvcc --version | tail -2
    echo "## harness"; ( cd "$HERE" && sha256sum PROTOCOL.md ds_compare_rows_*.jsonl ds_*.json 2>/dev/null || true )
} >> "$REC/env.txt"
if [ -f "$HERE/PROTOCOL.md" ]; then
    sha256sum "$HERE/PROTOCOL.md" > "$REC/protocol.sha256"
else
    log "note: no PROTOCOL.md next to the harness; its sha256 is not recorded"
fi
python - <<'PY'
import torch
assert torch.cuda.is_available()
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
print("cuda smoke OK on", torch.cuda.get_device_name(0), float((x @ x).float().abs().mean()) >= 0)
PY

# env.sh: sourced by run_all.sh (no SOUP_S0_BIN when the S0 venv was skipped)
S0_EXPORT="export SOUP_S0_BIN=\"$VENV_S0/bin/soup\""
if [ "$SKIP_S0" = "1" ]; then S0_EXPORT="# SKIP_S0=1: no S0 venv"; fi
cat > "$HERE/env.sh" <<EOF
# written by setup.sh
export PATH="$VENV_MAIN/bin:\$PATH"
export CUDA_HOME="$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export PY="$VENV_MAIN/bin/python"
$S0_EXPORT
export DS_COMPARE_MODEL_DIR="$MODEL_DIR"
export DS_COMPARE_RESULTS="$RESULTS"
export HF_HUB_OFFLINE=1
EOF
echo "$(( $(date +%s) - SETUP_T0 ))" > "$REC/setup_seconds.txt"
NEXT="bash run_all.sh --all"
if [ -n "$SETUP_PART" ]; then NEXT="bash run_all.sh --part $SETUP_PART"; fi
log "setup done in $(cat "$REC/setup_seconds.txt") s; next: $NEXT"
