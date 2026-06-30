#!/bin/bash
# Orchestrate LLaVA-OneVision-7B AOT reproduction across ratios for a given benchmark.
# Usage: bash run_ov_batch.sh <task> <keep_ratio>
#   task: mvbench | egoschema | videomme | longvideobench_val_v
# Ratios -> VISUAL_TOKEN_NUM (OV, 32 frames): 10->126, 15->144, 20->196, 25->205
set -u
TASK=$1
KR=$2
cd /home/msj_team/Jacob/nk/AOT

declare -A VTN=( [10]=126 [15]=144 [20]=196 [25]=205 )
for LABEL in 10 15 20 25; do
  echo "##### $(date '+%F %T')  OV $TASK ratio=${LABEL}%  VTN=${VTN[$LABEL]}  KR=$KR #####"
  GPUS=0,1 NPROC=2 bash run_eval.sh ov "$TASK" "${VTN[$LABEL]}" "$KR" "$LABEL" || echo "!!!! FAILED $TASK $LABEL !!!!"
done
echo "##### OV $TASK ALL RATIOS DONE $(date '+%F %T') #####"
