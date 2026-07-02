#!/usr/bin/env bash
# Run the 4 MLVU jobs needed for Table 2's Average(%) column.
# IMPORTANT: this machine's CUDA enumeration is FASTEST_FIRST and does NOT match
# nvidia-smi indices, so we pin GPUs by UUID (unambiguous).
#   G3 = nvidia-smi GPU3 (fd2e342a, free)   G1 = nvidia-smi GPU1 (a373a2c2, free)
# GPU0 is used by run_missing.sh; GPU2 is used by another process. We do NOT touch them.
# We deliberately do NOT create logs/.mlvu_ready, so run_missing.sh keeps skipping MLVU
# (no double-run). videomme vid 0.15 is left to run_missing.sh.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
source vidcom2_env.sh 2>/dev/null || true
# MLVU dataset (sy1998/MLVU_dev) needs HF resolution -> use proxy + disable offline mode.
source /home/msj_team/Jacob/0/env.sh 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
LOG=logs/mlvu_table2
mkdir -p "$LOG"
OL="$LOG/orchestrator.log"
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

has_results(){ # model task label
  local base; [ "$1" = ov ] && base=logs/repro/ov-7b || base=logs/repro/vid-7b
  find "${base}/${2}_${3}" -name '*results*.json' 2>/dev/null | grep -q .
}
run_job(){ # gpu model task ratio label logname
  local gpu=$1 model=$2 task=$3 ratio=$4 label=$5 name=$6
  if has_results "$model" "$task" "$label"; then log "SKIP done $model $task $label"; return 0; fi
  log "RUN $model $task R=$ratio ($label) on ${gpu:0:16}..."
  GPUS="$gpu" NPROC=1 bash run_eval.sh "$model" "$task" "$ratio" "$label" > "$LOG/$name.log" 2>&1
  log "END(exit=$?) $model $task $label"
}

log "=== MLVU orchestrator start (pid $$) ==="

# 1) First MLVU alone on GPU3 -> triggers + completes MLVU dataset extraction (avoid race)
run_job "$G3" ov mlvu_dev 0.25 0p25 ov_mlvu_0p25

# 2) Extraction done -> parallelize remaining 3 across GPU3 + GPU1
run_job "$G3" vid mlvu_dev 0.25 0p25 vid_mlvu_0p25 &
P3=$!
run_job "$G1" ov mlvu_dev 0.15 0p15 ov_mlvu_0p15 &
P1=$!
wait $P1
run_job "$G1" vid mlvu_dev 0.15 0p15 vid_mlvu_0p15 &
P1b=$!

wait $P3 $P1b
touch "$LOG/ALL_DONE"
log "=== ALL DONE ==="
