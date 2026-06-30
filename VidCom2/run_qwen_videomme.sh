#!/bin/bash
# Qwen2-VL VideoMME on VidCom2-qwen worktree
set -euo pipefail
QWEN_ROOT=/home/msj_team/Jacob/nk/VidCom2-qwen
source /home/msj_team/Jacob/nk/VidCom2/vidcom2_env.sh
cd "$QWEN_ROOT"
export PYTHONPATH=$QWEN_ROOT:$QWEN_ROOT/lmms-eval:${PYTHONPATH:-}

GPUS=${GPUS:-1}
NPROC=${NPROC:-1}
PORT=$((27000 + RANDOM % 2000))
OUT_BASE=./logs/repro/qwen2vl

run_qwen() {
  local R=$1
  local LABEL=$2
  unset COMPRESSOR
  if [ "$R" = "baseline" ]; then
    export R_RATIO=1.0
  else
    export COMPRESSOR=vidcom2
    export R_RATIO=$R
  fi
  OUT="$OUT_BASE/videomme_${LABEL}"
  CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --num_processes=$NPROC --main_process_port=$PORT \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args "pretrained=Qwen/Qwen2-VL-7B-Instruct,max_num_frames=32" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl \
    --output_path "$OUT"
}

run_qwen baseline baseline
run_qwen 0.25 r25
echo "Qwen2-VL videomme done"
