#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/msj_team/Jacob/0/env.sh 2>/dev/null || true

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUT_DIR="${STLLM_WEIGHT_DIR:-/home/msj_team/Jacob/nk/PruneVid/MODELS/st-llm}"
mkdir -p "${OUT_DIR}"

download_file() {
    local rel="$1"
    local dst="${OUT_DIR}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    curl -L --fail --retry 20 --retry-delay 5 --connect-timeout 30 -C - \
        "${HF_ENDPOINT}/farewellthree/ST_LLM_weight/resolve/main/${rel}" \
        -o "${dst}"
}

for rel in \
    QA_weight/config.json \
    QA_weight/pytorch_model-00001-of-00002.bin \
    QA_weight/pytorch_model-00002-of-00002.bin \
    QA_weight/pytorch_model.bin.index.json \
    QA_weight/tokenizer.model \
    QA_weight/tokenizer_config.json \
    conversation_weight/config.json \
    conversation_weight/pytorch_model-00001-of-00002.bin \
    conversation_weight/pytorch_model-00002-of-00002.bin \
    conversation_weight/pytorch_model.bin.index.json \
    conversation_weight/tokenizer.model \
    conversation_weight/tokenizer_config.json
do
    download_file "${rel}"
done

ln -sfn "${OUT_DIR}/QA_weight" "${OUT_DIR}/instructblipbase_stllm_qa"
ln -sfn "${OUT_DIR}/conversation_weight" "${OUT_DIR}/STLLM_conversation_weight"
touch "${OUT_DIR}/DOWNLOAD_COMPLETE"
echo "${OUT_DIR}"
