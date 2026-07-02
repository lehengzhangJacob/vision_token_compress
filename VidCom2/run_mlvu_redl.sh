#!/usr/bin/env bash
# MLVU re-download (hole-fill) + extract + eval.
# The zips are sparse (interior zero-holes ~70GB) so 274 videos are unextractable.
# Phase 1: fill the holes via HTTP Range (proxy+mirror, resumable).
# Phase 2: extract the now-recoverable videos (python per-file) -> verify 1122.
# Phase 3: run the 4 MLVU evals on GPU0+GPU1, fully offline.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOGD=logs/mlvu_redl; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
PY=/home/msj_team/.conda/envs/VidCom2/bin/python3.10
export HF_HOME=/home/msj_team/.cache/huggingface
G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2

log "=== Phase 1: fill sparse holes in MLVU zips (proxy+mirror) ==="
$PY mlvu_holefill.py >>"$LOGD/holefill.log" 2>&1
rc=$?
log "holefill exit=$rc (residual holes should be 0; see $LOGD/holefill.log)"
if [ $rc -ne 0 ]; then log "WARN holefill non-zero; continuing to extract what we can"; fi

log "=== Phase 2: extract missing videos from now-complete zips ==="
$PY - >>"$OL" 2>&1 <<'PY'
import glob, os, zipfile, shutil, pandas as pd
snap = glob.glob("/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/*/")[0]
mlvu = "/home/msj_team/.cache/huggingface/mlvu"; os.makedirs(mlvu, exist_ok=True)
df = pd.read_parquet("/home/msj_team/.cache/huggingface/mlvu_meta/test-00000-of-00001.parquet")
need = set(df['video_name'].unique())
missing = {v for v in need if not os.path.exists(os.path.join(mlvu, v))}
print("missing before extract:", len(missing), flush=True)
rec = 0; fail = 0
for z in sorted(glob.glob(snap+"/video_part_*.zip")):
    try: zf = zipfile.ZipFile(z)
    except Exception as e: print("UNREADABLE", os.path.basename(z), e, flush=True); continue
    for info in zf.infolist():
        base = os.path.basename(info.filename)
        if not base.endswith('.mp4') or base not in missing: continue
        tp = os.path.join(mlvu, base); tmp = tp+".part"
        try:
            with zf.open(info) as s, open(tmp,'wb') as d: shutil.copyfileobj(s, d, 1<<20)
            os.replace(tmp, tp); rec += 1
        except Exception as e:
            fail += 1
            if os.path.exists(tmp): os.remove(tmp)
            print("FAIL", base, repr(e)[:80], flush=True)
    zf.close()
miss2 = sorted(v for v in need if not os.path.exists(os.path.join(mlvu, v)))
print(f"recovered={rec} failed={fail} | missing after={len(miss2)}", flush=True)
if miss2: print("STILL_MISSING sample:", miss2[:15], flush=True)
PY

NV=$(find /home/msj_team/.cache/huggingface/mlvu -name '*.mp4' | wc -l)
log "videos on disk: $NV (need 1122 unique)"
if [ "$NV" -lt 1122 ]; then log "WARN still <1122; evals may skip. Proceeding anyway."; fi

has_results(){ local base; [ "$1" = ov ] && base=logs/repro/ov-7b || base=logs/repro/vid-7b
  find "${base}/mlvu_dev_${2}" -name '*results*.json' 2>/dev/null | grep -v submission | grep -q .; }
run_job(){ local gpu=$1 model=$2 ratio=$3 label=$4 name=$5
  if has_results "$model" "$label"; then log "SKIP done $model $label"; return 0; fi
  log "RUN $model mlvu R=$ratio ($label) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh "$model" mlvu_dev "$ratio" "$label" > "$LOGD/$name.log" 2>&1
  local rc=$?
  if has_results "$model" "$label"; then log "OK $model $label"; else log "FAIL(rc=$rc) $model $label (see $LOGD/$name.log)"; fi
}

log "=== Phase 3: 4 MLVU evals on GPU0+GPU1 ==="
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
            r=json.load(open(sorted(js)[-1]))["results"]; k=list(r.keys())[0]
            print(f"  {model} mlvu {label}: "+str({kk:round(vv,2) for kk,vv in r[k].items() if isinstance(vv,(int,float))}))
        else: print(f"  {model} mlvu {label}: NO RESULT")
PY
touch "$LOGD/ALL_DONE" logs/mlvu_recover/ALL_DONE
log "=== MLVU REDL ALL DONE ==="
