#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/prunevid_env.sh"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for VideoChatGPT-Bench GPT scoring"
  exit 1
fi

model_dir=MODELS/pllava-7b
weight_dir=MODELS/pllava-7b
num_frames=16
test_ratio=1

lora_alpha=4
selected_layer=5
alpha=0.4
tau=0.8
temporal_segment_ratio=0.25
cluster_ratio=0.5

SAVE_DIR=test_results/pllava-7b-prunevid-vcgbench
mkdir -p "${SAVE_DIR}"

python -m tasks.eval.vcgbench.pllava_eval_vcgbench \
    --pretrained_model_name_or_path "${model_dir}" \
    --save_path "${SAVE_DIR}/vcgbench" \
    --num_frames ${num_frames} \
    --weight_dir "${weight_dir}" \
    --pooling_shape 16-12-12 \
    --test_ratio ${test_ratio} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --selected_layer ${selected_layer} \
    --alpha ${alpha} \
    --tau ${tau} \
    --temporal_segment_ratio ${temporal_segment_ratio} \
    --cluster_ratio ${cluster_ratio}
