#!/usr/bin/env bash
# Fresh repeatability re-run of the 4 VidCom2 MLVU evals (the ONLY evals that used
# the previously-corrupt MLVU dataset). Videos already re-fetched+ffprobe-verified
# (0 corrupt). New labels (rerun_*) keep the existing clean v1 results intact so we
# can compare v1 vs v2. One GPU per job -> all 4 run concurrently (~14h, bounded by
# the two slow LLaVA-Video jobs).
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOGD=logs/mlvu_rerun; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2
G2=GPU-02602593-650b-b58f-d349-418c53deb125
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751

# quick sanity: confirm 0 corrupt before spending GPU time
rem=$(/home/msj_team/.conda/envs/VidCom2/bin/python3.10 - <<'PY'
import glob,subprocess,concurrent.futures as cf
vids=sorted(glob.glob('/home/msj_team/.cache/huggingface/mlvu/*.mp4'))
def ck(v):
    try:
        r=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_type","-of","csv=p=0",v],capture_output=True,text=True,timeout=60)
        return r.returncode==0 and "video" in (r.stdout or "")
    except: return False
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    print(sum(1 for ok in ex.map(ck,vids) if not ok))
PY
)
log "pre-run corrupt scan: $rem corrupt (of $(ls /home/msj_team/.cache/huggingface/mlvu/*.mp4 | wc -l))"
if [ "$rem" != "0" ]; then log "ABORT: corrupt videos present"; exit 2; fi

one(){ # gpu model ratio label
  log "RUN $2 mlvu_dev R=$3 ($4) on ${1:0:20}"
  GPUS="$1" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh "$2" mlvu_dev "$3" "$4" > "$LOGD/$2_$4.log" 2>&1
  log "DONE $2 $4 rc=$?"
}

log "=== 4 fresh evals: GPU0=ov R25, GPU1=ov R15, GPU2=vid R25, GPU3=vid R15 ==="
one "$G0" ov  0.25 rerun_0p25 & P0=$!
one "$G1" ov  0.15 rerun_0p15 & P1=$!
one "$G2" vid 0.25 rerun_0p25 & P2=$!
one "$G3" vid 0.15 rerun_0p15 & P3=$!
wait $P0 $P1 $P2 $P3

log "=== v1 vs v2 comparison ==="
/home/msj_team/.conda/envs/VidCom2/bin/python3.10 - <<'PY'
import glob, json
def score(m,lbl):
    js=[j for j in glob.glob(f"logs/repro/{m}/mlvu_dev_{lbl}/**/*result*.json",recursive=True) if 'submission' not in j]
    if not js: return None
    r=json.load(open(sorted(js)[-1]))["results"]; k=list(r.keys())[0]
    return r[k].get("mlvu_percetion_score,none")
for m,tag in [("ov-7b","LLaVA-OV"),("vid-7b","LLaVA-Video")]:
    for v1lbl,v2lbl,r in [("mlvu_0p25","rerun_0p25","R=25%"),("mlvu_0p15","rerun_0p15","R=15%")]:
        a,b=score(m,v1lbl),score(m,v2lbl)
        d="" if (a is None or b is None) else f"  delta={round(b-a,2)}"
        print(f"  {tag:12s} {r}: v1={round(a,2) if a else a}  v2={round(b,2) if b else b}{d}")
PY
touch "$LOGD/ALL_DONE"
log "=== MLVU RERUN ALL DONE ==="
