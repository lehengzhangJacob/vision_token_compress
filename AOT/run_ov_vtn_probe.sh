#!/usr/bin/env bash
# DECISIVE probe: is the AOT-OV LongVideoBench gap caused by too-low VTN (so the
# "25%" rows are really ~13% retained), or is it algorithmic (plateau)?
# Phase 1: map VTN -> measured retained (KR=0.1, LVB --limit) on GPU3.
# Phase 2: run FULL LVB at the VTN that yields ~25% measured retained.
# If full LVB jumps toward the paper's 56.3 -> VTN was the issue -> build full sweep.
# If it still plateaus ~53 -> gap is inherent; report honestly.
set -uo pipefail
cd /home/msj_team/Jacob/nk/AOT
LOGD=logs/ov_vtn; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
PY=/home/msj_team/.conda/envs/AOT/bin/python3.10
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751
KR=0.1
retained_of(){ rg -o "Retenion ratio : [0-9.]+" "$1" 2>/dev/null | awk -F': ' '{s+=$2;n++} END{if(n)printf "%.4f",s/n; else print "NA"}'; }

log "=== Phase 1: VTN -> retained calibration (KR=$KR, LVB --limit 48) ==="
CAL="$LOGD/vtn_calib.csv"; : > "$CAL"
for VTN in 150 250 350 450; do
  lbl="vcal${VTN}"; rm -rf "logs/repro/ov-7b/longvideobench_val_v_${lbl}"
  GPUS="$G3" NPROC=1 LIMIT=48 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh ov longvideobench_val_v "$VTN" "$KR" "$lbl" > "$LOGD/${lbl}.log" 2>&1
  r=$(retained_of "$LOGD/${lbl}.log"); log "  VTN=$VTN -> retained=$r"; echo "$VTN,$r" >> "$CAL"
done

VTN25=$($PY - "$CAL" <<'PY'
import sys, numpy as np
pts=[]
for l in open(sys.argv[1]):
    a=l.strip().split(",")
    if len(a)==2 and a[1] not in("NA",""): pts.append((float(a[0]),float(a[1])))
pts.sort(key=lambda x:x[1])
vtn=np.array([p[0] for p in pts]); rt=np.array([p[1] for p in pts])
v=float(np.interp(0.25, rt, vtn)); print(int(max(108,min(729,round(v)))))
PY
)
log "=== interpolated VTN for 25% measured retained: $VTN25 ==="

log "=== Phase 2: FULL LongVideoBench at VTN=$VTN25, KR=$KR (decisive) ==="
lbl="probe25"; rm -rf "logs/repro/ov-7b/longvideobench_val_v_${lbl}"
GPUS="$G3" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  bash run_eval.sh ov longvideobench_val_v "$VTN25" "$KR" "$lbl" > "$LOGD/${lbl}.log" 2>&1
r=$(retained_of "$LOGD/${lbl}.log")
$PY - "$lbl" "$VTN25" "$r" <<'PY'
import json, glob, sys
lbl, vtn, r = sys.argv[1], sys.argv[2], sys.argv[3]
js=[j for j in glob.glob(f"logs/repro/ov-7b/longvideobench_val_v_{lbl}/**/*result*.json",recursive=True) if 'submission' not in j]
sc=None
if js:
    d=json.load(open(sorted(js)[-1]))["results"]; k=list(d.keys())[0]; sc=d[k].get("lvb_acc,none")
print(f"[PROBE RESULT] VTN={vtn} measured_retained={float(r)*100:.1f}%  LVB_acc={sc}  (paper 25% row = 56.3)")
PY
log "=== PROBE DONE ==="
touch "$LOGD/ALL_DONE"
