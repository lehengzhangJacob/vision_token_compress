#!/usr/bin/env bash
set -euo pipefail
cd /home/msj_team/Jacob/nk/PruneVid/third_party/ST-LLM

export PYTHONPATH="./:${PYTHONPATH:-}"
export PRUNEVID=True
export PRUNEVID_ALPHA="${PRUNEVID_ALPHA:-0.4}"
export PRUNEVID_TAU="${PRUNEVID_TAU:-0.8}"
export PRUNEVID_TEMPORAL_SEGMENT_RATIO="${PRUNEVID_TEMPORAL_SEGMENT_RATIO:-0.25}"
export PRUNEVID_CLUSTER_RATIO="${PRUNEVID_CLUSTER_RATIO:-0.5}"

CKPT_PATH="${STLLM_CONV_CKPT:-/home/msj_team/Jacob/nk/PruneVid/MODELS/st-llm/STLLM_conversation_weight}"
VIDEO_DIR="${VCGBENCH_VIDEO_DIR:-/home/msj_team/Jacob/nk/PruneVid/DATAS/VCGBench/Videos/Benchmarking}"
QA_DIR="${VCGBENCH_QA_DIR:-/home/msj_team/Jacob/nk/PruneVid/DATAS/VCGBench/Zero_Shot_QA/Benchmarking_QA}"
OUT_DIR="${OUT_DIR:-/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-vcgbench}"
PYTHON_BIN="${PYTHON_BIN:-/home/msj_team/.conda/envs/PruneVid/bin/python}"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" stllm/test/vcgbench/videochatgpt_benchmark_general.py \
    --cfg-path config/instructblipbase_stllm_conversation.yaml \
    --ckpt-path "${CKPT_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --gt_file "${QA_DIR}/generic_qa.json" \
    --output_dir "${OUT_DIR}" \
    --output_name stllm_prunevid_generic \
    --num-frames "${NUM_FRAMES:-64}" \
    --gpu-id "${GPU_ID:-0}"

"${PYTHON_BIN}" stllm/test/vcgbench/videochatgpt_benchmark_general.py \
    --cfg-path config/instructblipbase_stllm_conversation.yaml \
    --ckpt-path "${CKPT_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --gt_file "${QA_DIR}/temporal_qa.json" \
    --output_dir "${OUT_DIR}" \
    --output_name stllm_prunevid_temporal \
    --num-frames "${NUM_FRAMES:-64}" \
    --gpu-id "${GPU_ID:-0}"

"${PYTHON_BIN}" stllm/test/vcgbench/videochatgpt_benchmark_consist.py \
    --cfg-path config/instructblipbase_stllm_conversation.yaml \
    --ckpt-path "${CKPT_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --gt_file "${QA_DIR}/consistency_qa.json" \
    --output_dir "${OUT_DIR}" \
    --output_name stllm_prunevid_consistency \
    --num-frames "${NUM_FRAMES:-64}" \
    --gpu-id "${GPU_ID:-0}"
