#!/bin/bash
# LongVideoBench parquet/json: ModelScope copy + optional HF metadata (videos already local).
set -euo pipefail
cd "$(dirname "$0")"
source vidcom2_env.sh

MS=/home/msj_team/.cache/huggingface/longvideobench_ms_src
SNAP_BASE=/home/msj_team/.cache/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots

if [ ! -f "$MS/validation-00000-of-00001.parquet" ]; then
  echo "===== Downloading LongVideoBench metadata from ModelScope ====="
  bash /home/msj_team/Jacob/nk/AOT/download_longvideobench.sh
fi

python - <<'PY'
import os
import shutil
from pathlib import Path

ms = Path("/home/msj_team/.cache/huggingface/longvideobench_ms_src")
snap_base = Path("/home/msj_team/.cache/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots")
snap_base.mkdir(parents=True, exist_ok=True)

# Reuse existing snapshot dir or create a stable one.
candidates = sorted(snap_base.iterdir()) if snap_base.is_dir() else []
snap = candidates[0] if candidates else snap_base / "ms_local"
snap.mkdir(parents=True, exist_ok=True)

for fn in ["validation-00000-of-00001.parquet", "test-00000-of-00001.parquet", "lvb_val.json", "README.md"]:
    src = ms / fn
    if src.is_file():
        shutil.copy2(src, snap / fn)
        print(f"==== copied {fn} ====", flush=True)

n_parquet = len(list(snap.glob("*.parquet")))
if n_parquet < 1:
    raise RuntimeError(f"No parquet in {snap}")

# Try HF metadata refresh (non-fatal if network is flaky).
os.environ.pop("HF_ENDPOINT", None)
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        "longvideobench/LongVideoBench",
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json", "README.md", ".gitattributes"],
        etag_timeout=120,
        resume_download=True,
        max_workers=1,
    )
    print(f"==== HF LVB metadata refreshed: {p} ====", flush=True)
    snap = Path(p)
except Exception as e:
    print(f"==== HF metadata skipped ({e}); using ModelScope copy at {snap} ====", flush=True)

Path("logs/.lvb_meta_ready").write_text(str(snap))
print(f"==== LVB metadata ready ({len(list(snap.glob('*.parquet')))} parquet) ====", flush=True)
PY

echo "===== LVB metadata prefetch done $(date) ====="
