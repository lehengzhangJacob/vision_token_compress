#!/bin/bash
# PerceptionTest large zips: curl resume; Qwen2-VL direct via proxy when available.
set -uo pipefail
cd "$(dirname "$0")"
source vidcom2_env.sh

LOG=logs/prefetch_table2.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "===== Table2 prefetch start $(date) ====="
echo "http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset}"

python - <<'PY' || exit 1
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.getcwd())
from xethub_dl import download_hf_xethub

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

PT_REPO = "lmms-lab/PerceptionTest_Val"
PT_ZIPS = ["valid_audios.zip", "videos_chunked_01.zip", "videos_chunked_02.zip"]
PT_SMALL = [".gitattributes", "README.md", "mc_question_val/validation-00000-of-00001.parquet"]
ZIP_DIR = Path(os.environ["HF_HOME"]) / "perceptiontest_zips"
PT_CACHE = Path(os.environ["HF_HOME"]) / "perceptiontest_val"


def dl_small(repo, fn, repo_type, use_mirror=False, attempts=8):
    saved = os.environ.get("HF_ENDPOINT")
    try:
        endpoints = ["https://hf-mirror.com"] if use_mirror else [None, "https://hf-mirror.com"]
        for i in range(1, attempts + 1):
            for endpoint in endpoints:
                if endpoint is None:
                    os.environ.pop("HF_ENDPOINT", None)
                    source = "huggingface.co"
                else:
                    os.environ["HF_ENDPOINT"] = endpoint
                    source = endpoint
                try:
                    p = hf_hub_download(repo, fn, repo_type=repo_type)
                    print(f"==== OK {fn} via {source} -> {p} ====", flush=True)
                    return p
                except Exception as e:
                    print(f"==== FAIL {fn} via {source}: {type(e).__name__}: {e}", flush=True)
            if i < attempts:
                wait = min(60 * i, 300)
                print(f"==== RETRY in {wait}s ====", flush=True)
                time.sleep(wait)
    finally:
        if saved is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = saved
    raise RuntimeError(f"failed to download {repo}/{fn}")


print("==== PerceptionTest small files (huggingface.co + proxy) ====", flush=True)
for fn in PT_SMALL:
    dl_small(PT_REPO, fn, "dataset", use_mirror=False)

print("==== PerceptionTest large zips (curl resume) ====", flush=True)
os.environ.pop("HF_ENDPOINT", None)
for z in PT_ZIPS:
    download_hf_xethub(PT_REPO, z, ZIP_DIR / z, repo_type="dataset")

PT_CACHE.mkdir(parents=True, exist_ok=True)
for z in PT_ZIPS:
    zp = ZIP_DIR / z
    print(f"==== UNZIP {z} -> {PT_CACHE} ====", flush=True)
    subprocess.run(["unzip", "-o", str(zp), "-d", str(PT_CACHE)], check=True)

video_dir = PT_CACHE / "videos"
n_mp4 = sum(1 for _ in video_dir.rglob("*.mp4")) if video_dir.is_dir() else 0
if n_mp4 < 100:
    raise RuntimeError(f"Too few PerceptionTest videos: {n_mp4} under {video_dir}")

print(f"==== PerceptionTest OK: {n_mp4} mp4 under {video_dir} ====", flush=True)

print("==== Qwen2-VL-7B via huggingface.co/proxy ====", flush=True)
qwen_repo = "Qwen/Qwen2-VL-7B-Instruct"
files = [
    e.path
    for e in HfApi().list_repo_tree(qwen_repo, repo_type="model", recursive=True)
    if hasattr(e, "size")
]
print(f"==== {qwen_repo}: {len(files)} files ====", flush=True)
for fn in files:
    dl_small(qwen_repo, fn, "model", use_mirror=False)

Path("logs/.table2_data_ready").write_text("ok\n")
print("==== TABLE2 PREFETCH DONE ====", flush=True)
PY

echo "===== Table2 prefetch end $(date) ====="
