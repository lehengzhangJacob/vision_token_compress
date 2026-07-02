#!/usr/bin/env bash
# Download MLVU_dev videos. The user's proxy exit node (OTE 香港HK01) currently fails ALL
# HTTPS (TLS handshake -> "wrong version number") for huggingface.co, google, etc., so the
# xet download stalled with 7 incomplete blobs (~230GB). Domestic mirror hf-mirror.com is
# reachable DIRECTLY (no proxy) -> use it. Classic HTTP LFS (xet disabled) resumes the
# *.incomplete blobs via Range requests. Network-only; does NOT use any GPU.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
# Direct connection to the mirror; DO NOT use the broken proxy.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/msj_team/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0
LOG=logs/mlvu_download.log
echo "[$(date '+%F %T')] start MLVU download via $HF_ENDPOINT (direct, xet off)" >> "$LOG"
for attempt in $(seq 1 20); do
  /home/msj_team/.conda/envs/VidCom2/bin/python3.10 - >> "$LOG" 2>&1 <<'PY'
import time
from huggingface_hub import snapshot_download
t = time.time()
path = snapshot_download("sy1998/MLVU_dev", repo_type="dataset", max_workers=4)
print("SNAPSHOT_DONE", path, "in", round(time.time() - t), "s", flush=True)
PY
  rc=$?
  echo "[$(date '+%F %T')] attempt $attempt exit=$rc" >> "$LOG"
  if [ $rc -eq 0 ]; then touch logs/.mlvu_download_done; break; fi
  sleep 10
done
