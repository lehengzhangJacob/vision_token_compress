#!/bin/bash
# Table 1 datasets only. MLVU / PerceptionTest / Qwen are separate.
set -euo pipefail
source "$(dirname "$0")/vidcom2_env.sh"

python - <<'PY'
import os
import time
from huggingface_hub import snapshot_download

print(f"http_proxy={os.environ.get('http_proxy', '(unset)')}", flush=True)
print(f"https_proxy={os.environ.get('https_proxy', '(unset)')}", flush=True)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
    os.environ.pop("HF_ENDPOINT", None)
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def grab(repo, repo_type="dataset", **kw):
    t = time.time()
    print(f"==== START {repo} ({repo_type}) ====", flush=True)
    p = snapshot_download(
        repo,
        repo_type=repo_type,
        etag_timeout=600,
        resume_download=True,
        max_workers=1,
        **kw,
    )
    print(f"==== DONE {repo} in {(time.time()-t)/60:.1f} min -> {p} ====", flush=True)
    return p

grab("OpenGVLab/MVBench", revision="video")
grab("lmms-lab/egoschema")
grab("lmms-lab/Video-MME")
print("==== TABLE1 PREFETCH DONE ====", flush=True)
PY

LVB=/home/msj_team/.cache/huggingface/longvideobench/videos
if [ ! -d "$LVB" ] || [ -z "$(ls -A "$LVB" 2>/dev/null)" ]; then
  echo "Extracting LongVideoBench..."
  bash /home/msj_team/Jacob/nk/AOT/extract_longvideobench.sh
fi
