#!/usr/bin/env bash
set -euo pipefail
cd /home/msj_team/Jacob/nk/PruneVid

LOG_DIR="${LOG_DIR:-/home/msj_team/Jacob/nk/PruneVid/logs}"
mkdir -p "${LOG_DIR}"

wait_gpu() {
    local threshold_mb="${GPU_FREE_THRESHOLD_MB:-18000}"
    local candidates="${GPU_CANDIDATES:-GPU-fd2e342a-3610-76c0-ef30-d087968b4751 2 1 0}"
    while true; do
        for gpu in ${candidates}; do
            if CUDA_VISIBLE_DEVICES="${gpu}" /home/msj_team/.conda/envs/AOT/bin/python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
            then
                used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
                if [ "${used}" -lt "${threshold_mb}" ]; then
                    echo "${gpu}"
                    return 0
                fi
            fi
        done
        echo "[queue] waiting for GPU memory < ${threshold_mb} MB" >> "${LOG_DIR}/prunevid_extension_queue.log"
        sleep 300
    done
}

run_ov() {
    local script="$1"
    local name="$2"
    if [ -e "test_results/ov-prunevid-${name}/DONE" ]; then
        echo "[queue] skip OV ${name}" >> "${LOG_DIR}/prunevid_extension_queue.log"
        return 0
    fi
    gpu=$(wait_gpu)
    echo "[queue] run OV ${name} on GPU ${gpu}" >> "${LOG_DIR}/prunevid_extension_queue.log"
    mkdir -p "test_results/ov-prunevid-${name}"
    set +e
    env GPUS="${gpu}" NPROC=1 OUT_DIR="/home/msj_team/Jacob/nk/PruneVid/test_results/ov-prunevid-${name}" \
        bash "${script}" >> "${LOG_DIR}/ov_prunevid_${name}.log" 2>&1
    status=$?
    set -e
    if [ "${status}" -ne 0 ]; then
        echo "[queue] FAILED OV ${name} status=${status}" >> "${LOG_DIR}/prunevid_extension_queue.log"
        return "${status}"
    fi
    if ! compgen -G "/home/msj_team/Jacob/nk/AOT/logs/repro/ov-7b/${name}_prunevid/*/*_results.json" > /dev/null; then
        echo "[queue] FAILED OV ${name} no results json" >> "${LOG_DIR}/prunevid_extension_queue.log"
        return 1
    fi
    touch "test_results/ov-prunevid-${name}/DONE"
}

run_stllm() {
    local script="$1"
    local name="$2"
    if [ -e "test_results/stllm-prunevid-${name}/DONE" ]; then
        echo "[queue] skip ST-LLM ${name}" >> "${LOG_DIR}/prunevid_extension_queue.log"
        return 0
    fi
    while [ ! -f "MODELS/st-llm/DOWNLOAD_COMPLETE" ]; do
        echo "[queue] waiting ST-LLM weights" >> "${LOG_DIR}/prunevid_extension_queue.log"
        sleep 600
    done
    gpu=$(wait_gpu)
    echo "[queue] run ST-LLM ${name} on GPU ${gpu}" >> "${LOG_DIR}/prunevid_extension_queue.log"
    mkdir -p "test_results/stllm-prunevid-${name}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" GPU_ID=0 OUT_DIR="/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-${name}" \
        bash "${script}" >> "${LOG_DIR}/stllm_prunevid_${name}.log" 2>&1
    status=$?
    set -e
    if [ "${status}" -ne 0 ]; then
        echo "[queue] FAILED ST-LLM ${name} status=${status}" >> "${LOG_DIR}/prunevid_extension_queue.log"
        return "${status}"
    fi
    touch "test_results/stllm-prunevid-${name}/DONE"
}

run_ov /home/msj_team/Jacob/nk/PruneVid/run_ov_prunevid_mvbench.sh mvbench || true
run_ov /home/msj_team/Jacob/nk/PruneVid/run_ov_prunevid_videomme.sh videomme || true
run_ov /home/msj_team/Jacob/nk/PruneVid/run_ov_prunevid_egoschema.sh egoschema || true

run_stllm /home/msj_team/Jacob/nk/PruneVid/run_stllm_prunevid_mvbench.sh mvbench || true
run_stllm /home/msj_team/Jacob/nk/PruneVid/run_stllm_prunevid_videomme.sh videomme || true
run_stllm /home/msj_team/Jacob/nk/PruneVid/run_stllm_prunevid_egoschema.sh egoschema || true
run_stllm /home/msj_team/Jacob/nk/PruneVid/run_stllm_prunevid_vcgbench.sh vcgbench || true

echo "[queue] complete" >> "${LOG_DIR}/prunevid_extension_queue.log"
