#!/usr/bin/env bash
# Orchestrate the 5 eval jobs needed to complete Table 2's Short/Medium/Long/Average columns.
# GPUs: 1 and 3 (GPU0 busy with longvideobench R15 vid, GPU2 busy with other proc).
# Strategy: run the FIRST MLVU job alone to completion so the MLVU dataset (video zip)
# is extracted exactly once (no concurrent-extraction race); then parallelize the rest.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOG=logs/table2_complete
mkdir -p "$LOG"
OL="$LOG/orchestrator.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

log "=== orchestrator start (pid $$) ==="

# GPU3: VideoMME for Video R=15% (independent of MLVU; fills Overall + Short/Med/Long)
( GPUS=3 NPROC=1 bash run_eval.sh vid videomme 0.15 0p15 > "$LOG/vid_videomme_0p15.log" 2>&1 ) &
P_VMME=$!
log "launched vid videomme 0.15 on GPU3 (pid $P_VMME)"

# GPU1: first MLVU job ALONE -> triggers + completes dataset extraction
log "running ov mlvu_dev 0.25 on GPU1 (extraction pass, foreground)"
GPUS=1 NPROC=1 bash run_eval.sh ov mlvu_dev 0.25 0p25 > "$LOG/ov_mlvu_0p25.log" 2>&1
log "finished ov mlvu_dev 0.25 (exit $?)"

# extraction done -> GPU1 free; start next MLVU there
( GPUS=1 NPROC=1 bash run_eval.sh ov mlvu_dev 0.15 0p15 > "$LOG/ov_mlvu_0p15.log" 2>&1 ) &
P_G1=$!
log "launched ov mlvu_dev 0.15 on GPU1 (pid $P_G1)"

# wait for GPU3 videomme to free, then run a vid MLVU there
wait $P_VMME
log "vid videomme 0.15 done; launching vid mlvu_dev 0.25 on GPU3"
( GPUS=3 NPROC=1 bash run_eval.sh vid mlvu_dev 0.25 0p25 > "$LOG/vid_mlvu_0p25.log" 2>&1 ) &
P_G3=$!

# when GPU1 frees, run the last vid MLVU there
wait $P_G1
log "ov mlvu_dev 0.15 done; launching vid mlvu_dev 0.15 on GPU1"
GPUS=1 NPROC=1 bash run_eval.sh vid mlvu_dev 0.15 0p15 > "$LOG/vid_mlvu_0p15.log" 2>&1
log "vid mlvu_dev 0.15 done (exit $?)"

wait $P_G3
log "vid mlvu_dev 0.25 done"

touch "$LOG/ALL_DONE"
log "=== ALL DONE ==="
