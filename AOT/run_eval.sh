#!/bin/bash
# AOT reproduction eval runner (2x RTX 4090; direct HF through proxy when available)
# Usage: bash run_eval.sh <ov|vid> <task> <VISUAL_TOKEN_NUM> <KEEP_RATIO> <label> [inter_compress: True|False]
#   e.g. bash run_eval.sh ov mvbench 126 0.3 10
# Optional env: LIMIT (e.g. 4 for a smoke test), GPUS (default 0,1), NPROC (default 2)
set -e
source /home/msj_team/Jacob/nk/AOT/aot_env.sh

MODEL=$1; TASK=$2; VTN=$3; KR=$4; LABEL=$5; IC=${6:-}

export GLOBAL_RATIO=0.5
export INTRA_SCALE=1.0
export INTER_SCALE=1.0
export VISUAL_TOKEN_NUM=$VTN
export KEEP_RATIO=$KR
unset EGOSCHEMA_UNPAD DYNAMIC_SEGMENTS INTER_COMPRESS

case "$TASK" in
  egoschema*) export EGOSCHEMA_UNPAD=True ;;
esac

GPUS=${GPUS:-0,1}
NPROC=${NPROC:-2}
PORT=$((25000 + RANDOM % 2000))
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="--limit $LIMIT"; fi

if [ "$MODEL" = "ov" ]; then
  export INTER_COMPRESS=${IC:-True}
  MODEL_NAME=llava_onevision
  MARGS="pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32"
  SUFFIX=llava_onevision
  OUT=./logs/repro/ov-7b/${TASK}_${LABEL}
else
  # LLaVA-Video: authors disable inter-frame compression for VID
  export INTER_COMPRESS=${IC:-False}
  MODEL_NAME=llava_vid
  MARGS="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64"
  SUFFIX=llava_vid
  OUT=./logs/repro/vid-7b/${TASK}_${LABEL}
fi

echo "[run_eval] model=$MODEL task=$TASK VTN=$VTN KR=$KR label=$LABEL INTER_COMPRESS=$INTER_COMPRESS EGOSCHEMA_UNPAD=${EGOSCHEMA_UNPAD:-unset} GPUS=$GPUS limit=${LIMIT:-none}"
echo "[run_eval] output -> $OUT"

CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --num_processes=$NPROC --main_process_port=$PORT \
  -m lmms_eval \
  --model $MODEL_NAME \
  --model_args $MARGS \
  --tasks $TASK \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix $SUFFIX \
  $LIMIT_ARG \
  --output_path $OUT
