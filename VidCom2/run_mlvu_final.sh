#!/usr/bin/env bash
# Final MLVU pipeline. The real blocker was never the network for inference: the 4 evals only
# need (a) local metadata (mlvu_dev.yaml now points at ~/.cache/huggingface/mlvu_meta) and
# (b) the videos extracted to $HF_HOME/mlvu. This script extracts all zips then runs the 4
# evals FULLY OFFLINE (HF_HUB_OFFLINE=1), so the broken proxy is irrelevant.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOG=logs/mlvu_final; mkdir -p "$LOG"; OL="$LOG/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

MLVU_DIR=/home/msj_team/.cache/huggingface/mlvu
SNAP=/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/96207eb9aa7101e2a495dd147684a7e618c79e12
mkdir -p "$MLVU_DIR"

log "=== extract MLVU zips (skip existing) ==="
for z in "$SNAP"/video_part_*.zip; do
  log "unzip $(basename "$z")"
  unzip -n -q "$z" -d "$MLVU_DIR" 2>>"$LOG/unzip.err" || log "  warn: unzip issue on $(basename "$z")"
done
N=$(ls "$MLVU_DIR" | grep -c '\.mp4$' || true)
log "extracted videos now: $N (need 1122)"

G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2   # nvidia-smi GPU1
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751   # nvidia-smi GPU3

has_results(){ local base; [ "$1" = ov ] && base=logs/repro/ov-7b || base=logs/repro/vid-7b
  find "${base}/mlvu_dev_${2}" -name '*results*.json' 2>/dev/null | grep -v submission | grep -q .; }
run_job(){ # gpu model ratio label name
  local gpu=$1 model=$2 ratio=$3 label=$4 name=$5
  if has_results "$model" "$label"; then log "SKIP done $model $label"; return 0; fi
  log "RUN $model mlvu R=$ratio ($label) on ${gpu:0:20}"
  # FULLY OFFLINE: no network, proxy irrelevant. run_eval.sh defaults HF_HUB_OFFLINE=1.
  GPUS="$gpu" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh "$model" mlvu_dev "$ratio" "$label" > "$LOG/$name.log" 2>&1
  local rc=$?
  if has_results "$model" "$label"; then log "OK $model $label"; else log "FAIL(rc=$rc) $model $label"; fi
}

log "=== run 4 MLVU evals on G1+G3 ==="
# pair 1
run_job "$G1" ov  0.25 0p25 ov_mlvu_0p25 &  A=$!
run_job "$G3" vid 0.25 0p25 vid_mlvu_0p25 & B=$!
wait $A $B
# pair 2
run_job "$G1" ov  0.15 0p15 ov_mlvu_0p15 &  C=$!
run_job "$G3" vid 0.15 0p15 vid_mlvu_0p15 & D=$!
wait $C $D
touch "$LOG/ALL_DONE"
log "=== ALL DONE ==="