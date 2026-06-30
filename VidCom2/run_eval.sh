#!/bin/bash
# VidCom2 eval runner. Usage:
#   bash run_eval.sh <ov|vid> <task> <r_ratio> <label>
#   r_ratio: baseline | 0.25 | 0.15
# Env: LIMIT, GPUS, NPROC, FLASH_ATTN=1, EFFICIENCY=1
set -euo pipefail
source "$(dirname "$0")/vidcom2_env.sh"

MODEL=$1
TASK=$2
R_RATIO=$3
LABEL=$4

unset COMPRESSOR EGOSCHEMA_UNPAD
if [ "$R_RATIO" = "baseline" ] || [ "$R_RATIO" = "1.0" ]; then
  unset COMPRESSOR
  export R_RATIO=1.0
else
  export COMPRESSOR=vidcom2
  export R_RATIO=$R_RATIO
fi

case "$TASK" in
  egoschema*) export EGOSCHEMA_UNPAD=True ;;
esac

GPUS=${GPUS:-1}
NPROC=${NPROC:-1}
PORT=$((26000 + RANDOM % 2000))
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

FA2=""
if [ "${FLASH_ATTN:-0}" = "1" ] || [ "${EFFICIENCY:-0}" = "1" ]; then
  FA2=",attn_implementation=flash_attention_2"
fi

if [ "$MODEL" = "ov" ]; then
  MODEL_NAME=llava_onevision
  OV_PRETRAINED=${OV_PRETRAINED:-/home/msj_team/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd}
  MARGS="pretrained=${OV_PRETRAINED},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32${FA2}"
  SUFFIX=llava_onevision
  OUT=./logs/repro/ov-7b/${TASK}_${LABEL}
else
  MODEL_NAME=llava_vid
  VID_PRETRAINED=${VID_PRETRAINED:-/home/msj_team/.cache/huggingface/hub/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/013210b3aff822f1558b166d39c1046dd109520f}
  MARGS="pretrained=${VID_PRETRAINED},conv_template=qwen_1_5,model_name=llava_qwen,mm_spatial_pool_mode=average,max_frames_num=64${FA2}"
  SUFFIX=llava_vid
  OUT=./logs/repro/vid-7b/${TASK}_${LABEL}
fi

echo "[run_eval] model=$MODEL task=$TASK r_ratio=$R_RATIO label=$LABEL COMPRESSOR=${COMPRESSOR:-unset} GPUS=$GPUS"
echo "[run_eval] -> $OUT"

CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --num_processes=$NPROC --main_process_port=$PORT \
  -m lmms_eval \
  --model "$MODEL_NAME" \
  --model_args "$MARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "$SUFFIX" \
  $LIMIT_ARG \
  --output_path "$OUT"
