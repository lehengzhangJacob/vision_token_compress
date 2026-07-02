#!/usr/bin/env bash
# Install the remaining ST-LLM deps into the dedicated `stllm` env
# (transformers 4.28 stack). Waits for the torch install to finish first.
set -uo pipefail
PIP=/home/msj_team/.conda/envs/stllm/bin/pip
PY=/home/msj_team/.conda/envs/stllm/bin/python
LOG=/home/msj_team/Jacob/nk/PruneVid/logs/stllm_env_setup.log
TORCH_LOG=/home/msj_team/Jacob/nk/PruneVid/logs/stllm_env_torch.log

echo "===== wait for torch install $(date -Is) =====" >> "$LOG"
while ! grep -q "TORCH_INSTALL_EXIT" "$TORCH_LOG" 2>/dev/null; do
    sleep 15
done
echo "torch log says: $(grep TORCH_INSTALL_EXIT "$TORCH_LOG")" >> "$LOG"

echo "===== verify torch $(date -Is) =====" >> "$LOG"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" >> "$LOG" 2>&1

echo "===== install core (transformers 4.28 stack) $(date -Is) =====" >> "$LOG"
"$PIP" install \
    "numpy<2" \
    "transformers==4.28.0" \
    "tokenizers==0.13.2" \
    "huggingface-hub==0.13.4" \
    "accelerate==0.20.3" \
    "timm==0.6.13" \
    "sentencepiece" \
    "ftfy" \
    "regex" \
    "einops==0.7.0" \
    "omegaconf==2.3.0" \
    "iopath==0.1.10" \
    "opencv-python==4.7.0.72" \
    "decord==0.6.0" \
    "imageio==2.33.1" \
    "webdataset==0.2.48" \
    "scipy" \
    "scikit-learn" \
    "pandas" \
    "matplotlib" \
    "tqdm" \
    "mmengine" >> "$LOG" 2>&1
echo "CORE_INSTALL_EXIT=$?" >> "$LOG"

echo "===== final versions $(date -Is) =====" >> "$LOG"
"$PY" - >> "$LOG" 2>&1 <<'PY'
import importlib
for m in ["torch","transformers","tokenizers","timm","decord","omegaconf","mmengine","cv2","sentencepiece"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "FAIL", e)
PY
echo "===== DONE $(date -Is) =====" >> "$LOG"
