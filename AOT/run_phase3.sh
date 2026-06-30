#!/bin/bash
# Phase 3: after OV non-gated benchmarks finish -> extract LongVideoBench videos,
# run LongVideoBench OV (4 ratios), then run the full LLaVA-Video-7B grid.
cd /home/msj_team/Jacob/nk/AOT
FOLLOWON_PID=${1:-570389}
echo "[phase3] $(date '+%F %T') waiting for OV followon PID $FOLLOWON_PID to finish..."
while kill -0 "$FOLLOWON_PID" 2>/dev/null; do sleep 180; done
echo "[phase3] $(date '+%F %T') extracting LongVideoBench videos..."
if [ ! -d /home/msj_team/.cache/huggingface/longvideobench/videos ]; then
  bash extract_longvideobench.sh > extract_lvb.log 2>&1
fi
echo "[phase3] $(date '+%F %T') LongVideoBench OV (4 ratios)..."
bash run_ov_batch.sh longvideobench_val_v 0.1
echo "[phase3] $(date '+%F %T') LLaVA-Video-7B grid (4 benchmarks x 15/25%)..."
for t in mvbench egoschema videomme longvideobench_val_v; do
  bash run_vid_batch.sh "$t"
done
echo "[phase3] ===== ALL PHASE 3 DONE $(date '+%F %T') ====="
