#!/usr/bin/env bash
# Auto-run the 4 MLVU evals needed for Table 2 Average(%), AFTER the dataset finishes
# downloading. Waits for logs/.mlvu_download_done, then runs on nvidia-smi GPU3 + GPU1
# (pinned by UUID because CUDA enumeration != nvidia-smi order). run_missing.sh only ever
# lands on nvidia-smi GPU0, so GPU1/GPU3 are safe. Never puts 2 jobs on the same GPU.
# Uses hf-mirror.com DIRECT (proxy HK01 node's TLS is broken) for any residual metadata.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
source vidcom2_env.sh 2>/dev/null || true
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/msj_team/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0

G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751   # nvidia-smi GPU3
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2   # nvidia-smi GPU1
LOG=logs/mlvu_run
mkdir -p "$LOG"
OL="$LOG/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

has_results(){ local base; [ "$1" = ov ] && base=logs/repro/ov-7b || base=logs/repro/vid-7b
  find "${base}/${2}_${3}" -name '*results*.json' 2>/dev/null | grep -q .; }

run_job(){ # gpu model task ratio label name
  local gpu=$1 model=$2 task=$3 ratio=$4 label=$5 name=$6
  if has_results "$model" "$task" "$label"; then log "SKIP done $model $task $label"; return 0; fi
  log "RUN $model $task R=$ratio ($label) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    HF_ENDPOINT=https://hf-mirror.com \
    bash run_eval.sh "$model" "$task" "$ratio" "$label" > "$LOG/$name.log" 2>&1
  log "END(exit=$?) $model $task $label"
}

wait_extract(){ # logfile  -> return when dataset load/extraction is past (inference started)
  local f=$1 i=0
  while [ $i -lt 240 ]; do   # up to ~2h for zip extraction
    grep -qE "Model Responding|Running generate_until|Building contexts for mlvu" "$f" 2>/dev/null && return 0
    grep -qE "Traceback|Error during evaluation" "$f" 2>/dev/null && return 1
    sleep 30; i=$((i+1))
  done; return 1; }

# 1) Wait for the download to finish.
log "=== waiting for MLVU download (logs/.mlvu_download_done) ==="
w=0
while [ ! -f logs/.mlvu_download_done ]; do
  sleep 60; w=$((w+1))
  [ $((w % 10)) -eq 0 ] && log "  ...still waiting ($w min); incomplete=$(find /home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev -name '*.incomplete' 2>/dev/null | wc -l)"
  [ $w -ge 600 ] && { log "TIMEOUT waiting for download"; exit 1; }
done
log "download done after ~$w min"

log "=== MLVU eval orchestrator start (pid $$) ==="
# 2) First job alone on G3 -> triggers zip extraction. Overlap G1 once inference begins.
run_job "$G3" ov mlvu_dev 0.25 0p25 ov_mlvu_0p25 &
P3=$!
if wait_extract "$LOG/ov_mlvu_0p25.log"; then
  log "extraction/load done -> start ov 0.15 on G1 in parallel"
  run_job "$G1" ov mlvu_dev 0.15 0p15 ov_mlvu_0p15 &
  P1=$!
else
  log "WARN: first job did not reach inference; waiting for it before continuing"
fi

wait $P3
run_job "$G3" vid mlvu_dev 0.25 0p25 vid_mlvu_0p25 &
P3b=$!
# ensure ov 0.15 was started even if wait_extract failed
if ! has_results ov mlvu_dev 0p15 && ! kill -0 ${P1:-0} 2>/dev/null; then
  run_job "$G1" ov mlvu_dev 0.15 0p15 ov_mlvu_0p15 & P1=$!
fi
wait ${P1:-$$} 2>/dev/null || true
run_job "$G1" vid mlvu_dev 0.15 0p15 vid_mlvu_0p15 &
P1b=$!

wait $P3b $P1b
touch "$LOG/ALL_DONE"
log "=== ALL DONE ==="
