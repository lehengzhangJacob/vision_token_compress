#!/usr/bin/env bash
# MLVU recovery + rerun.
# Phase 1: extract the 274 videos that `unzip` skipped (bad-offset) straight from the
#          existing zips via Python zipfile (per-file, streaming). No network needed.
# Phase 2: run the 4 MLVU evals (ov/vid x R25/R15) FULLY OFFLINE on GPU0+GPU1.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOGD=logs/mlvu_recover; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
PY=/home/msj_team/.conda/envs/VidCom2/bin/python3.10
export HF_HOME=/home/msj_team/.cache/huggingface

G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b   # nvidia-smi GPU0
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2   # nvidia-smi GPU1

log "=== Phase 1: recover missing MLVU videos from existing zips (no network) ==="
$PY - >>"$OL" 2>&1 <<'PY'
import glob, os, zipfile, shutil, pandas as pd
snap = glob.glob("/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/*/")[0]
mlvu = "/home/msj_team/.cache/huggingface/mlvu"; os.makedirs(mlvu, exist_ok=True)
df = pd.read_parquet("/home/msj_team/.cache/huggingface/mlvu_meta/test-00000-of-00001.parquet")
need = set(df['video_name'].unique())
missing = {v for v in need if not os.path.exists(os.path.join(mlvu, v))}
print("missing before:", len(missing), flush=True)
recovered = 0
for z in sorted(glob.glob(snap+"/video_part_*.zip")):
    try:
        zf = zipfile.ZipFile(z)
    except Exception as e:
        print("UNREADABLE", os.path.basename(z), e, flush=True); continue
    for info in zf.infolist():
        if not info.filename.endswith('.mp4'): continue
        base = os.path.basename(info.filename)
        if base not in missing: continue
        tp = os.path.join(mlvu, base); tmp = tp + ".part"
        try:
            with zf.open(info) as src, open(tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
            os.replace(tmp, tp); recovered += 1
            if recovered % 25 == 0: print("recovered", recovered, flush=True)
        except Exception as e:
            if os.path.exists(tmp): os.remove(tmp)
            print("FAIL", base, repr(e), flush=True)
    zf.close()
missing2 = {v for v in need if not os.path.exists(os.path.join(mlvu, v))}
print("recovered:", recovered, "| missing after:", len(missing2), flush=True)
print("STILL_MISSING", sorted(missing2)[:10], flush=True)
PY

NVID=$(find /home/msj_team/.cache/huggingface/mlvu -name '*.mp4' | wc -l)
log "videos on disk now: $NVID (need 1122 unique)"

has_results(){ local base; [ "$1" = ov ] && base=logs/repro/ov-7b || base=logs/repro/vid-7b
  find "${base}/mlvu_dev_${2}" -name '*results*.json' 2>/dev/null | grep -v submission | grep -q .; }
run_job(){ # gpu model ratio label name
  local gpu=$1 model=$2 ratio=$3 label=$4 name=$5
  if has_results "$model" "$label"; then log "SKIP done $model $label"; return 0; fi
  log "RUN $model mlvu R=$ratio ($label) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh "$model" mlvu_dev "$ratio" "$label" > "$LOGD/$name.log" 2>&1
  local rc=$?
  if has_results "$model" "$label"; then log "OK $model $label"; else log "FAIL(rc=$rc) $model $label (see $LOGD/$name.log)"; fi
}

log "=== Phase 2: run 4 MLVU evals on GPU0+GPU1 ==="
run_job "$G0" ov  0.25 0p25 ov_mlvu_0p25 &  A=$!
run_job "$G1" vid 0.25 0p25 vid_mlvu_0p25 & B=$!
wait $A $B
run_job "$G0" ov  0.15 0p15 ov_mlvu_0p15 &  C=$!
run_job "$G1" vid 0.15 0p15 vid_mlvu_0p15 & D=$!
wait $C $D

log "=== results ==="
$PY - >>"$OL" 2>&1 <<'PY'
import json, glob
for model,base in [("ov","logs/repro/ov-7b"),("vid","logs/repro/vid-7b")]:
    for label in ["0p25","0p15"]:
        js=[j for j in glob.glob(f"{base}/mlvu_dev_{label}/**/*result*.json",recursive=True) if 'submission' not in j]
        if js:
            d=json.load(open(sorted(js)[-1])); r=d["results"]; k=list(r.keys())[0]
            sc={kk:round(vv,2) for kk,vv in r[k].items() if isinstance(vv,(int,float))}
            print(f"  {model} mlvu {label}: {sc}")
        else:
            print(f"  {model} mlvu {label}: NO RESULT")
PY
touch "$LOGD/ALL_DONE"
log "=== MLVU RECOVER ALL DONE ==="
