#!/usr/bin/env bash
# job_p6.sh - vk job for issue #2609 step 1 (scratch A/B on one RTX 4090; no product change).  cwd = /workspace/payload.
#   A  soup-cli 0.77.2 default            B  A + embed_tokens/lm_head/norms back to bf16 after prepare_model_for_kbit_training
#   C  prepare_model_for_kbit_training replaced (freeze + enable_input_require_grads + gradient checkpointing, bf16 kept)
#   D  A + training.use_liger: true (only if liger-kernel installs and imports)
# Per round r (seed 3406+r): for each arm  P-cell (TCO_PROFILE=1: 5 warm-up + 3 profiled + 1 snapshot step, then stop; rounds in
#   $P6_PROF_REPS only)  then T-cell (no profiler: 5 warm-up + 100 measured steps, the 840-row window).
# Harness = case1-profile payload (soup-8a-prof) + the p6 edits listed in harness-diff.txt.  Results -> $VK_OUT.  No secrets here.
set -u
: "${VK_OUT:?need VK_OUT}"
PAYLOAD="$(pwd)"
SHARED=/workspace/tco-shared
CELLS=/workspace/tco-cells
ARMS="${P6_ARMS:-A B C D}"
REPS="${P6_REPS:-1 2 3}"
PROF_REPS="${P6_PROF_REPS:-1 2}"
EXPECT_DATA_SHA=eba8a5bc6074eced5adb255f3f8872d942779978171421e13f9c670494479c8a
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1
export TCO_HF_HOME="$SHARED/hf" TCO_OUT="$SHARED"
mkdir -p "$SHARED" "$CELLS" "$VK_OUT/runs"
CARD="NVIDIA GeForce RTX 4090"
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$VK_OUT/job_wrapper.log"; }
now() { date +%s; }

finish() {
  [ -n "${SMI_PID:-}" ] && kill "$SMI_PID" 2>/dev/null
  say "job wrapper end rc=${1:-?}"
  exit "${1:-0}"
}
trap 'finish 143' TERM INT

say "== environment"
{ date -u; uname -a; nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit --format=csv;
  free -g; df -h /workspace /tmp; nproc; ulimit -l; command -v python3 python3.11 python3.12 python3.13 uv gcc; python3 --version; cat /root/vk/deadline 2>/dev/null; } > "$VK_OUT/env.txt" 2>&1
head -40 "$VK_OUT/env.txt"
nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu --format=csv > "$VK_OUT/gpu-idle-before.txt" 2>&1

pick_python() {
  local c
  for c in python3.12 python3.11 python3.13 /venv/main/bin/python python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys,venv,ensurepip; raise SystemExit(0 if (3,11)<=sys.version_info[:2]<=(3,13) else 1)" 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}
PY="$(pick_python)" || {
  say "no usable system python 3.11-3.13; installing 3.12 via uv"
  command -v uv >/dev/null 2>&1 || python3 -m pip install --quiet uv || pip install --quiet uv
  uv python install 3.12 && PY="$(uv python find 3.12)"
}
[ -n "${PY:-}" ] || { say "FATAL: no python 3.11-3.13"; finish 11; }
say "python: $PY $($PY --version 2>&1)"
export TCO_PYTHON="$PY"

T_SETUP0=$(now)
if ! TCO_FG=1 bash run_tco.sh setup > "$VK_OUT/setup.log" 2>&1; then say "FATAL: setup failed"; tail -30 "$VK_OUT/setup.log"; cp -f "$SHARED"/*.txt "$VK_OUT/" 2>/dev/null; finish 12; fi
cp -f "$SHARED"/freeze.txt "$SHARED"/pip-check.txt "$SHARED"/setup.json "$VK_OUT/" 2>/dev/null
if ! grep -q "No broken requirements found" "$SHARED/pip-check.txt"; then
  say "GATE: pip check FAILED -> stop"; head -40 "$SHARED/pip-check.txt"; finish 13
fi
say "pip check: clean; setup took $(( $(now) - T_SETUP0 )) s"
VPY="$SHARED/venv/bin/python"

# ---- liger-kernel for arm D: installed against the frozen stack (-c freeze.txt) so it cannot move any pinned package
LIGER_OK=0
{
  "$VPY" -m pip install -c "$SHARED/freeze.txt" liger-kernel
  echo "pip rc=$?"
} > "$VK_OUT/liger-install.txt" 2>&1
"$VPY" - >> "$VK_OUT/liger-install.txt" 2>&1 <<'EOF' && LIGER_OK=1
import importlib.metadata as md
import liger_kernel
from liger_kernel.transformers import monkey_patch
print("liger-kernel", md.version("liger-kernel"), "qwen2 supported:", "qwen2" in monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN)
assert "qwen2" in monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN
EOF
"$VPY" -m pip freeze > "$VK_OUT/freeze-with-liger.txt" 2>&1
"$VPY" -m pip check > "$VK_OUT/pip-check-with-liger.txt" 2>&1
say "liger-kernel usable: LIGER_OK=$LIGER_OK ($(tail -2 "$VK_OUT/liger-install.txt" | tr '\n' ' ' | cut -c1-200))"

# ---- CUPTI smoke
"$VPY" - > "$VK_OUT/cupti-smoke.txt" 2>&1 <<'EOF'
import torch
from torch.profiler import profile, ProfilerActivity
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(5):
        (a @ a).sum().item()
    torch.cuda.synchronize()
ev = [e for e in prof.events() if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type)]
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), "cuda events:", len(ev))
print("CUPTI_OK" if ev else "CUPTI_NO_CUDA_EVENTS")
EOF
tee -a "$VK_OUT/job_wrapper.log" < "$VK_OUT/cupti-smoke.txt"
HF_HOME="$TCO_HF_HOME" "$VPY" tco_run.py preflight 1 soup > "$VK_OUT/preflight.txt" 2>&1 || say "preflight non-zero (informational: RAM bar is the TCO record's 48 GB)"

# ---- model download (timed) and the 840-row window (sha256 checked against the record)
T_DL0=$(now)
HF_HOME="$TCO_HF_HOME" "$VPY" - > "$VK_OUT/model-download.txt" 2>&1 <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("snapshot", path)
EOF
say "model download rc=$? in $(( $(now) - T_DL0 )) s; hf cache $(du -sh "$TCO_HF_HOME" 2>/dev/null | cut -f1)"
T_DATA0=$(now)
HF_HOME="$TCO_HF_HOME" "$VPY" tco_run.py prepare-data 1 > "$VK_OUT/data-prepare.txt" 2>&1
say "data prepare rc=$? in $(( $(now) - T_DATA0 )) s"
cp -f "$SHARED/data/1/manifest.json" "$VK_OUT/data-manifest.json" 2>/dev/null
GOT_SHA="$(sha256sum "$SHARED/data/1/train.jsonl" 2>/dev/null | cut -d' ' -f1)"
if [ "$GOT_SHA" = "$EXPECT_DATA_SHA" ]; then say "DATA sha256 matches the record's eba8a5bc...: $GOT_SHA"; else say "DATA sha256 DIFFERS from the record: got '$GOT_SHA' want $EXPECT_DATA_SHA (arms stay comparable with each other, not with the record's data)"; fi
printf 'got=%s\nwant=%s\n' "$GOT_SHA" "$EXPECT_DATA_SHA" > "$VK_OUT/data-sha-check.txt"
[ -s "$SHARED/data/1/train.jsonl" ] || { say "FATAL: no data"; finish 14; }

# ---- throttle-reason sampler (separate from the harness poller); field name differs by driver
THR_FIELD=""
for f in clocks_throttle_reasons.active clocks_event_reasons.active; do
  if nvidia-smi --query-gpu=$f --format=csv,noheader >/dev/null 2>&1; then THR_FIELD=$f; break; fi
done
if [ -n "$THR_FIELD" ]; then
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,$THR_FIELD --format=csv,noheader -l 2 > "$VK_OUT/smi_throttle.csv" 2>/dev/null &
  SMI_PID=$!
fi
say "throttle sampler field: ${THR_FIELD:-none}"

collect_cell() {   # copy the small files + traces of one finished cell into $VK_OUT/runs/<cell>/
  local cell="$1" dir dst f
  dir="$(ls -d "$CELLS/$cell"/runs/1/soup/*/rep* 2>/dev/null | head -1)"
  [ -n "$dir" ] || { say "collect: no run dir for $cell"; return 0; }
  dst="$VK_OUT/runs/$cell"; mkdir -p "$dst"
  for f in worker.log effective_config.json run_meta.json result.json gpu.json gpu_samples.json steps.jsonl log.jsonl soup.yaml status.json \
           train_begin.json train_end.json meter_installed.json meter_error.json profiler_error.json capped.json arm_info.json; do
    [ -f "$dir/$f" ] && cp -f "$dir/$f" "$dst/"
  done
  [ -d "$dir/prof" ] && mkdir -p "$dst/prof" && cp -f "$dir/prof"/* "$dst/prof/" 2>/dev/null
  return 0
}

DEADLINE="$(tr -cd '0-9' < /root/vk/deadline 2>/dev/null)"
run_cell() {   # arm rep kind(P|T) est_seconds
  local arm="$1" rep="$2" kind="$3" est="$4" cell rc t0
  cell="${arm}-r${rep}-${kind}"
  if [ -n "$DEADLINE" ] && [ $(( DEADLINE - $(now) )) -lt $(( est + 900 )) ]; then
    say "SKIP $cell: $(( DEADLINE - $(now) )) s to the deadline, need ~$est s + 900 s reserve"
    return 9
  fi
  rm -rf "$CELLS/$cell"; mkdir -p "$CELLS/$cell"
  ln -sfn "$SHARED/data" "$CELLS/$cell/data"
  export TCO_ARM="$arm"
  if [ "$kind" = P ]; then export TCO_PROFILE=1; else unset TCO_PROFILE; fi
  t0=$(now)
  say "RUN cell=$cell arm=$arm repeat=$rep kind=$kind"
  TCO_OUT="$CELLS/$cell" "$VPY" tco_run.py run 1 soup --repeat "$rep" --card "$CARD" --pretrain-cap-s 3600 --setup-s 0 >> "$VK_OUT/job_wrapper.log" 2>&1
  rc=$?
  say "DONE cell=$cell rc=$rc in $(( $(now) - t0 )) s"
  printf '%s\t%s\t%s\t%s\t%s\n' "$cell" "$t0" "$(now)" "$rc" "$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader | tr -d ' ')" >> "$VK_OUT/cell_times.tsv"
  collect_cell "$cell"
  unset TCO_PROFILE
  sleep 3
  return 0
}

for rep in $REPS; do
  for arm in $ARMS; do
    if [ "$arm" = D ] && [ "$LIGER_OK" != 1 ]; then
      say "SKIP arm D round $rep: liger-kernel not usable (see liger-install.txt)"; echo "arm D not run: liger-kernel not usable" > "$VK_OUT/liger-not-run.txt"; continue
    fi
    case " $PROF_REPS " in *" $rep "*) run_cell "$arm" "$rep" P 330 ;; esac
    run_cell "$arm" "$rep" T 540
  done
done

"$VPY" -m pip freeze > "$VK_OUT/freeze-final.txt" 2>&1
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv >> "$VK_OUT/env.txt" 2>&1
say "all cells done; du of runs:"; du -sh "$VK_OUT"/runs/* 2>&1 | tee -a "$VK_OUT/job_wrapper.log" | tail -30
n=0; while [ -f /workspace/hold ] && [ $n -lt 360 ]; do sleep 5; n=$((n+1)); done
finish 0
