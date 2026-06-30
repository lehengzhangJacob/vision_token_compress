#!/bin/bash
# LongVideoBench is gated on HF -> download from ModelScope (authors' approach).
source /home/msj_team/Jacob/nk/AOT/aot_env.sh
DEST=/home/msj_team/.cache/huggingface/longvideobench_ms_src
mkdir -p "$DEST"
python - <<PY
from modelscope import dataset_snapshot_download
import time
t=time.time()
p = dataset_snapshot_download('AI-ModelScope/LongVideoBench', local_dir="$DEST")
print(f"==== LONGVIDEOBENCH MS DONE in {(time.time()-t)/60:.1f} min -> {p} ====", flush=True)
PY
