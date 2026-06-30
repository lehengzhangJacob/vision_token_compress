#!/usr/bin/env bash
set -euo pipefail
cd /home/msj_team/Jacob/nk/PruneVid/third_party/ST-LLM

export PYTHONPATH="./:${PYTHONPATH:-}"
export PRUNEVID=True
export PRUNEVID_ALPHA="${PRUNEVID_ALPHA:-0.4}"
export PRUNEVID_TAU="${PRUNEVID_TAU:-0.8}"
export PRUNEVID_TEMPORAL_SEGMENT_RATIO="${PRUNEVID_TEMPORAL_SEGMENT_RATIO:-0.25}"
export PRUNEVID_CLUSTER_RATIO="${PRUNEVID_CLUSTER_RATIO:-0.5}"

CKPT_PATH="${STLLM_QA_CKPT:-/home/msj_team/Jacob/nk/PruneVid/MODELS/st-llm/instructblipbase_stllm_qa}"
ANNO_PATH="${MVBENCH_JSON:-/home/msj_team/Jacob/nk/PruneVid/DATAS/MVBench/json}"
OUT_DIR="${OUT_DIR:-/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-mvbench}"
PYTHON_BIN="${PYTHON_BIN:-/home/msj_team/.conda/envs/PruneVid/bin/python}"

"${PYTHON_BIN}" stllm/test/mvbench/mv_bench_infer.py \
    --cfg-path config/instructblipbase_stllm_qa.yaml \
    --ckpt-path "${CKPT_PATH}" \
    --anno-path "${ANNO_PATH}" \
    --output_dir "${OUT_DIR}" \
    --output_name stllm_prunevid_mvbench \
    --num-frames "${NUM_FRAMES:-16}" \
    --gpu-id "${GPU_ID:-0}" \
    --ask_simple
