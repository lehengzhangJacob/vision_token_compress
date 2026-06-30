#!/bin/bash
# Waits for the running MVBench OV batch to finish, then runs EgoSchema + VideoMME (OV, 4 ratios each).
cd /home/msj_team/Jacob/nk/AOT
MV_PID=${1:-554448}
echo "[followon] waiting for MVBench batch PID $MV_PID to finish... $(date '+%F %T')"
while kill -0 "$MV_PID" 2>/dev/null; do sleep 120; done
echo "[followon] MVBench done. Starting EgoSchema $(date '+%F %T')"
bash run_ov_batch.sh egoschema 0.3
echo "[followon] Starting VideoMME $(date '+%F %T')"
bash run_ov_batch.sh videomme 0.1
echo "[followon] ===== EgoSchema + VideoMME DONE $(date '+%F %T') ====="
