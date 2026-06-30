#!/bin/bash
# LLaVA-Video-7B AOT reproduction (paper Table 2: ratios 15% & 25%).
# Intra anchors (64 frames): 15->144, 25->198. Authors disable inter-frame compression for VID.
# Usage: bash run_vid_batch.sh <task>
set -u
TASK=$1
cd /home/msj_team/Jacob/nk/AOT
declare -A VTN=( [15]=144 [25]=198 )
for LABEL in 15 25; do
  echo "##### $(date '+%F %T')  VID $TASK ratio=${LABEL}%  VTN=${VTN[$LABEL]} #####"
  GPUS=0,1 NPROC=2 bash run_eval.sh vid "$TASK" "${VTN[$LABEL]}" 0.4 "$LABEL" False || echo "!!!! FAILED VID $TASK $LABEL !!!!"
done
echo "##### VID $TASK ALL RATIOS DONE $(date '+%F %T') #####"
