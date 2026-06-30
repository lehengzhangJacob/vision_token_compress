#!/usr/bin/env bash
# Paper-aligned EgoSchema: fullset 5031 + official validation server upload.
set -euo pipefail
source "$(dirname "$0")/prunevid_env.sh"

model_dir=MODELS/pllava-7b
weight_dir=MODELS/pllava-7b
num_frames=16

lora_alpha=14
selected_layer=10
alpha=0.4
tau=0.8
temporal_segment_ratio=0.25
cluster_ratio=0.5

SAVE_DIR=test_results/pllava-7b-prunevid-egoschema
RESULT_DIR="${SAVE_DIR}/egoschema"
mkdir -p "${SAVE_DIR}"

python3 scripts/prep_json.py egoschema

python -m tasks.eval.egoshcema.pllava_eval_egoschema \
    --pretrained_model_name_or_path "${model_dir}" \
    --save_path "${RESULT_DIR}" \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --top_p 1.0 \
    --temperature 1.0 \
    --weight_dir "${weight_dir}" \
    --pooling_shape 16-12-12 \
    --conv_mode eval_mvbench \
    --selected_layer ${selected_layer} \
    --alpha ${alpha} \
    --tau ${tau} \
    --temporal_segment_ratio ${temporal_segment_ratio} \
    --cluster_ratio ${cluster_ratio} \
    --eval_split fullset

echo "Uploading fullset predictions to EgoSchema validation server..."
python3 evaluate_egoschema_result.py --result_dir "${RESULT_DIR}"
