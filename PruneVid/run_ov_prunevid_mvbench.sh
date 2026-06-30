#!/usr/bin/env bash
set -euo pipefail
cd /home/msj_team/Jacob/nk/AOT

export PRUNEVID=True
export PRUNEVID_ALPHA="${PRUNEVID_ALPHA:-0.4}"
export PRUNEVID_SELECTED_LAYER="${PRUNEVID_SELECTED_LAYER:-10}"
export PRUNEVID_TAU="${PRUNEVID_TAU:-0.8}"
export PRUNEVID_TEMPORAL_SEGMENT_RATIO="${PRUNEVID_TEMPORAL_SEGMENT_RATIO:-0.25}"
export PRUNEVID_CLUSTER_RATIO="${PRUNEVID_CLUSTER_RATIO:-0.5}"

env GPUS="${GPUS:-0}" NPROC="${NPROC:-1}" bash run_eval.sh ov mvbench 729 0.4 prunevid False
