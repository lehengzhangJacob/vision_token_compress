#!/usr/bin/env bash
# Clean MLVU re-run after the corrupt-video fix. Waits for mlvu_fix.py to finish
# (validated re-fetch), re-verifies corruption is cleared, then runs the 4 MLVU evals:
#   GPU0: ov R25, ov R15   |   GPU1: vid R25, vid R15   (parallel across GPUs).
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOGD=logs/mlvu_eval2; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751
FIXLOG=/home/msj_team/Jacob/nk/VidCom2/logs/mlvu_fix.log

log "=== phase 1: wait for mlvu_fix.py to finish ==="
while ! rg -q "^\[.*\] DONE fixed=" "$FIXLOG" 2>/dev/null; do sleep 60; done
log "fix done: $(rg -o 'DONE fixed=.*' "$FIXLOG" | tail -1)"

# re-verify corruption cleared (quick ffprobe recount)
rem=$(/home/msj_team/.conda/envs/VidCom2/bin/python3.10 - <<'PY'
import glob,subprocess,concurrent.futures as cf,os
bad=0
vids=[os.path.join('/home/msj_team/.cache/huggingface/mlvu',l.strip()) for l in open('/home/msj_team/Jacob/nk/VidCom2/logs/mlvu_corrupt_list.txt') if l.strip()]
def ck(v):
    try:
        r=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_type","-of","csv=p=0",v],capture_output=True,text=True,timeout=40)
        return r.returncode==0 and "video" in (r.stdout or "")
    except: return False
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    bad=sum(1 for ok in ex.map(ck,vids) if not ok)
print(bad)
PY
)
log "residual corrupt after fix: $rem"

one(){ # gpu model ratio label
  log "RUN $2 mlvu_dev R=$3 ($4) on ${1:0:20}"
  GPUS="$1" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh "$2" mlvu_dev "$3" "$4" > "$LOGD/$2_$4.log" 2>&1
  log "DONE $2 $4 rc=$?"
}
# GPU0 chains the two fast OV jobs (~3h each); the two slow LLaVA-Video jobs
# (~7-13h each) get their own GPU (G1, G3) so they run concurrently, not serially.
log "=== phase 2: 4 clean evals (GPU0=ov x2, GPU1=vid R25, GPU3=vid R15) ==="
( one "$G0" ov 0.25 mlvu_0p25; one "$G0" ov 0.15 mlvu_0p15 ) & P0=$!
one "$G1" vid 0.25 mlvu_0p25 & P1=$!
one "$G3" vid 0.15 mlvu_0p15 & P3=$!
wait $P0 $P1 $P3

log "=== MLVU results ==="
/home/msj_team/.conda/envs/VidCom2/bin/python3.10 - <<'PY'
import glob, json
for m,tag in [("ov-7b","LLaVA-OV"),("vid-7b","LLaVA-Video")]:
    for lbl,r in [("mlvu_0p25","R=25%"),("mlvu_0p15","R=15%")]:
        js=[j for j in glob.glob(f"logs/repro/{m}/mlvu_dev_{lbl}/**/*result*.json",recursive=True) if 'submission' not in j]
        sc=None
        if js:
            r0=json.load(open(sorted(js)[-1]))["results"]; k=list(r0.keys())[0]
            sc=r0[k].get("mlvu_percetion_score,none") or r0[k].get("mlvu_percep_score,none")
        print(f"  {tag} {r}: MLVU={sc}")
PY
touch "$LOGD/ALL_DONE"
log "=== MLVU EVAL2 ALL DONE ==="
