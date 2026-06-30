#!/bin/bash
# MLVU_dev: small files via hf_hub; large xethub zips via curl resume (xethub_dl.py).
set -uo pipefail
cd "$(dirname "$0")"
source vidcom2_env.sh

LOG=logs/prefetch_mlvu.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "===== MLVU prefetch start $(date) ====="
echo "http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset}"

python - <<'PY' || exit 1
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from xethub_dl import download_hf_xethub, hf_hub_download_small

os.environ.pop("HF_ENDPOINT", None)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

REPO = "sy1998/MLVU_dev"
PARTS = [f"video_part_{i}.zip" for i in range(1, 9)]
META = ["mlvu/test-00000-of-00001.parquet", ".gitattributes"]
ZIP_DIR = Path(os.environ["HF_HOME"]) / "mlvu_zips"
MLVU_CACHE = Path(os.environ["HF_HOME"]) / "mlvu"

for f in META:
    p = hf_hub_download_small(REPO, f, repo_type="dataset")
    print(f"==== META OK {f} -> {p} ====", flush=True)

for z in PARTS:
    dest = ZIP_DIR / z
    download_hf_xethub(REPO, z, dest, repo_type="dataset")

MLVU_CACHE.mkdir(parents=True, exist_ok=True)
for z in PARTS:
    zp = ZIP_DIR / z
    print(f"==== UNZIP {z} -> {MLVU_CACHE} ====", flush=True)
    subprocess.run(["unzip", "-o", str(zp), "-d", str(MLVU_CACHE)], check=True)

n_videos = sum(1 for _ in MLVU_CACHE.rglob("*.mp4"))
if n_videos < 100:
    raise RuntimeError(f"Too few videos after unzip: {n_videos}")

print(f"==== MLVU DONE: {n_videos} mp4 under {MLVU_CACHE} =====", flush=True)
Path("logs/.mlvu_ready").write_text(str(MLVU_CACHE))
PY

echo "===== MLVU prefetch end $(date) ====="
