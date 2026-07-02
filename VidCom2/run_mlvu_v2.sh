#!/usr/bin/env bash
# Relaunch the 4 MLVU evals. Root cause of the previous failure: mlvu_dev.yaml has no
# local_files_only flag, so lmms_eval calls dataset_info -> huggingface.co, which dies on the
# broken proxy's TLS (WRONG_VERSION_NUMBER). Fix: unset proxy + no_proxy='*' hard guard +
# HF_ENDPOINT=hf-mirror (verified working), xet off. Videos are cached; dataset build only
# needs to resolve metadata + finish extracting zips into $HF_HOME/mlvu.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
# Hard-guard the network BEFORE sourcing anything that inspects http_proxy.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy='*' NO_PROXY='*'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/msj_team/.cache/huggingface
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0

G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2   # nvidia-smi GPU1 (free)
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751   # nvidia-smi GPU3 (free)
LOG=logs/mlvu_run2; mkdir -p "$LOG"; OL="$LOG/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

has_results(){ local base; [ "$1" = ov ] && base=logs/repro/ov-7b || base=logs/repro/vid-7b
  find "${base}/${2}_${3}" -name '*results*.json' 2>/dev/null | grep -v submission | grep -q .; }

run_job(){ # gpu model ratio label name
  local gpu=$1 model=$2 ratio=$3 label=$4 name=$5
  if has_results "$model" mlvu_dev "$label"; then log "SKIP done $model mlvu $label"; return 0; fi
  log "RUN $model mlvu R=$ratio ($label) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 no_proxy='*' NO_PROXY='*' HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    bash run_eval.sh "$model" mlvu_dev "$ratio" "$label" > "$LOG/$name.log" 2>&1
  local rc=$?
  if has_results "$model" mlvu_dev "$label"; then log "OK $model mlvu $label"; else log "FAIL(rc=$rc) $model mlvu $label"; fi
}

wait_infer(){ local f=$1 i=0
  while [ $i -lt 360 ]; do
    grep -qE "Model Responding|Running generate_until" "$f" 2>/dev/null && return 0
    grep -qE "Error during evaluation|Traceback|does not exist" "$f" 2>/dev/null && return 1
    sleep 20; i=$((i+1))
  done; return 1; }

log "=== MLVU v2 start (pid $$) ==="
# 1) First job alone on G1: resolves metadata + finishes zip extraction (avoid concurrent race).
run_job "$G1" ov 0.25 0p25 ov_mlvu_0p25 &
P1=$!
if wait_infer "$LOG/ov_mlvu_0p25.log"; then
  log "dataset ready -> start vid 0.25 on G3 in parallel"
  run_job "$G3" vid 0.25 0p25 vid_mlvu_0p25 &
  P3=$!
else
  log "WARN first job not at inference yet; will still continue after it finishes"
fi
wait $P1
# remaining
run_job "$G1" ov 0.15 0p15 ov_mlvu_0p15 & Pa=$!
if ! kill -0 ${P3:-0} 2>/dev/null && ! has_results vid mlvu_dev 0p25; then
  run_job "$G3" vid 0.25 0p25 vid_mlvu_0p25 & P3=$!
fi
wait ${P3:-$$} 2>/dev/null || true
run_job "$G3" vid 0.15 0p15 vid_mlvu_0p15 & Pb=$!
wait $Pa $Pb
touch "$LOG/ALL_DONE"
log "=== ALL DONE ==="