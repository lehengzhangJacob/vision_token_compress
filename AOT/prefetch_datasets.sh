#!/bin/bash
source /home/msj_team/Jacob/nk/AOT/aot_env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
import time
def grab(repo, **kw):
    t=time.time()
    print(f"==== START {repo} {kw} ====", flush=True)
    p = snapshot_download(repo, repo_type="dataset", etag_timeout=60, **kw)
    print(f"==== DONE {repo} in {(time.time()-t)/60:.1f} min -> {p} ====", flush=True)

grab("OpenGVLab/MVBench", revision="video")
grab("lmms-lab/egoschema")
grab("lmms-lab/Video-MME")
print("==== ALL HF DATASETS PREFETCHED ====", flush=True)
PY
