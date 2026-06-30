#!/usr/bin/env bash
# Download VCGBench videos + run inference (GPT scoring skipped without OPENAI_API_KEY).
set -uo pipefail
cd "$(dirname "$0")"
source /home/msj_team/Jacob/0/env.sh
source prunevid_env.sh

LOG=logs/vcgbench_pipeline.log
mkdir -p logs DATAS/VCGBench/Zero_Shot_QA/Benchmarking_QA
exec > >(tee -a "$LOG") 2>&1

echo "===== VCGBench prefetch start $(date) ====="

python - <<'PY' || exit 1
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/msj_team/Jacob/nk/VidCom2")
from huggingface_hub import hf_hub_download
from xethub_dl import download_hf_xethub

root = Path("/home/msj_team/Jacob/nk/PruneVid/DATAS/VCGBench")
qa = root / "Zero_Shot_QA/Benchmarking_QA"
qa.mkdir(parents=True, exist_ok=True)

import os
os.environ.pop("HF_ENDPOINT", None)
for fn in ["generic_qa.json", "temporal_qa.json", "consistency_qa.json"]:
    dest = qa / fn
    if not dest.exists():
        p = hf_hub_download("zrchen03/VCGBENCH", fn, repo_type="dataset")
        shutil.copy2(p, dest)
    print(f"==== OK {fn} ====", flush=True)

zip_path = root / "Test_Videos.zip"
download_hf_xethub("zrchen03/VCGBENCH", "Test_Videos.zip", zip_path, repo_type="dataset")

videos = root / "Videos"
videos.mkdir(parents=True, exist_ok=True)
marker = videos / ".unzipped"
if not marker.exists():
    print(f"==== UNZIP {zip_path} -> {videos} ====", flush=True)
    subprocess.run(["unzip", "-o", str(zip_path), "-d", str(videos)], check=True)
    # zrchen03 zip may unpack Benchmarking/, Test_Videos/, or flat mp4s
    bench = videos / "Benchmarking"
    bench.mkdir(parents=True, exist_ok=True)
    test_sub = videos / "Test_Videos"
    if test_sub.is_dir():
        for f in test_sub.glob("*.mp4"):
            dst = bench / f.name
            if not dst.exists():
                dst.symlink_to(f.resolve())
        for f in test_sub.glob("*.mkv"):
            dst = bench / f.name
            if not dst.exists():
                dst.symlink_to(f.resolve())
    for f in videos.glob("*.mp4"):
        dst = bench / f.name
        if not dst.exists():
            f.rename(dst)
    for f in videos.glob("*.mkv"):
        dst = bench / f.name
        if not dst.exists():
            f.rename(dst)
    marker.write_text("ok\n")

n_mp4 = sum(1 for _ in (videos / "Benchmarking").rglob("*.mp4"))
print(f"==== VCGBench videos: {n_mp4} mp4 ====", flush=True)
if n_mp4 < 10:
    raise RuntimeError(f"Too few videos: {n_mp4}")
PY

echo "===== VCGBench inference start $(date) ====="
CUDA_VISIBLE_DEVICES=2 bash run_vcgbench_infer.sh || exit 1
echo "===== VCGBench pipeline end $(date) ====="
