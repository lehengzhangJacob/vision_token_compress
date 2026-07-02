#!/usr/bin/env bash
# Option B: make AOT-OV long-video (VideoMME + LongVideoBench) match the paper's
# 25/20/15/10% "Retained" rows. Authors only published one long-video point (VTN=108,
# KR=0.1 -> ~7.2% retained), which is MORE aggressive than the paper's table. Here we
# keep the authors' long-video intra value VTN=108 and sweep the temporal KEEP_RATIO
# (KR) upward, calibrate KR->retained, then run full VideoMME+LVB at the calibrated KRs.
set -uo pipefail
cd /home/msj_team/Jacob/nk/AOT
source /home/msj_team/.conda/etc/profile.d/conda.sh 2>/dev/null || true
LOGD=logs/ov_optB; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
PY=/home/msj_team/.conda/envs/AOT/bin/python3.10

G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b   # GPU0
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2   # GPU1
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751   # GPU3
MLVU_DONE=/home/msj_team/Jacob/nk/VidCom2/logs/mlvu_recover/ALL_DONE

VTN=108
retained_of(){ rg -o "Retenion ratio : [0-9.]+" "$1" 2>/dev/null | awk -F': ' '{s+=$2;n++} END{if(n)printf "%.4f",s/n; else print "NA"}'; }

# ---------- Phase 1: calibrate KR -> retained on GPU3 (VideoMME, --limit) ----------
log "=== Phase 1: KR calibration (VTN=$VTN) on GPU3 ==="
CALIB="$LOGD/calib.csv"; : > "$CALIB"
for KR in 0.15 0.25 0.35 0.45; do
  lbl="calib_kr${KR/./}"
  rm -rf "logs/repro/ov-7b/videomme_${lbl}"
  GPUS="$G3" NPROC=1 LIMIT=96 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh ov videomme "$VTN" "$KR" "$lbl" > "$LOGD/${lbl}.log" 2>&1
  r=$(retained_of "$LOGD/${lbl}.log")
  log "  KR=$KR -> retained=$r"; echo "$KR,$r" >> "$CALIB"
done

# choose KR for target retained {0.10,0.15,0.20,0.25} by linear interpolation on calib points
mapfile -t KRS < <($PY - "$CALIB" <<'PY'
import sys, numpy as np
pts=[]
for line in open(sys.argv[1]):
    a=line.strip().split(",")
    if len(a)==2 and a[1] not in ("NA",""):
        pts.append((float(a[0]),float(a[1])))
pts.sort(key=lambda x:x[1])
kr=np.array([p[0] for p in pts]); rt=np.array([p[1] for p in pts])
for tgt in (0.10,0.15,0.20,0.25):
    k=float(np.interp(tgt, rt, kr))         # invert retained->KR
    print(f"{tgt:.2f} {max(0.05,min(0.9,round(k,3)))}")
PY
)
log "=== calibrated KR per target retained ==="; for row in "${KRS[@]}"; do log "  target=$row"; done

# ---------- Phase 2: full VideoMME + LVB runs at calibrated KRs ----------
gpu_free(){ local u=$1 m; m=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v U="$u" '$1==U{print $2}'); [ -n "$m" ] && [ "$m" -lt 2500 ]; }
log "=== Phase 2: wait for MLVU done + GPU0/GPU1 free ==="
while :; do [ -f "$MLVU_DONE" ] && gpu_free "$G0" && gpu_free "$G1" && break; sleep 60; done
log "GPUs free -> full runs"

run_full(){ # gpu task tgt kr
  local gpu=$1 task=$2 tgt=$3 kr=$4 lbl="optB_r${3/./}"
  log "RUN ov $task VTN=$VTN KR=$kr (target ${tgt}) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 LIMIT="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh ov "$task" "$VTN" "$kr" "$lbl" > "$LOGD/${task}_${lbl}.log" 2>&1
  log "DONE ov $task ${lbl} rc=$?"
}

# 4 targets x 2 benchmarks = 8 runs, 3 at a time across G0/G1/G3
declare -a JOBS=()
for row in "${KRS[@]}"; do set -- $row; tgt=$1; kr=$2
  JOBS+=("videomme $tgt $kr"); JOBS+=("longvideobench_val_v $tgt $kr")
done
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in "$G0" "$G1" "$G3"; do
    [ $i -ge ${#JOBS[@]} ] && break
    set -- ${JOBS[$i]}; run_full "$g" "$1" "$2" "$3" & i=$((i+1))
  done
  wait
done

log "=== Option B results (score + measured retained) ==="
$PY - >>"$OL" 2>&1 <<'PY'
import json, glob, re
def retained(task,lbl):
    f=f"logs/ov_optB/{task}_{lbl}.log"
    import os
    if not os.path.exists(f): return None
    v=[float(m.group(1)) for m in re.finditer(r"Retenion ratio : ([0-9.]+)", open(f,errors='ignore').read())]
    return round(100*sum(v)/len(v),2) if v else None
for tgt in ["010","015","020","025"]:
    lbl=f"optB_r{tgt}"
    for task,key in [("videomme","videomme_perception_score,none"),("longvideobench_val_v","lvb_acc,none")]:
        js=[j for j in glob.glob(f"logs/repro/ov-7b/{task}_{lbl}/**/*result*.json",recursive=True) if 'submission' not in j]
        sc=None
        if js:
            r=json.load(open(sorted(js)[-1]))["results"]; k=list(r.keys())[0]; sc=r[k].get(key)
        print(f"  target{tgt} {task}: score={sc} retained%={retained(task,lbl)}")
PY
touch "$LOGD/ALL_DONE"
log "=== OPTION B ALL DONE ==="
