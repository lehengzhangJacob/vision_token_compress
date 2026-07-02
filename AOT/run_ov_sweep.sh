#!/usr/bin/env bash
# AOT-OV CORRECTED long-video sweep. The probe showed the original rows used too-low
# VTN (mislabeled retained). Calibration (VTN->measured retained, KR=0.1):
#   150->10.3%, 250->18.4%, 350->27.6%, 450->36.5%  =>  10/15/20/25% ~ VTN 146/208/267/322.
# Runs LVB + VideoMME at those VTNs. LVB@322 already exists (probe: 54.75%), reused.
# GPU pool: GPU3 always; GPU0/GPU1 only AFTER the ST-LLM fullset job signals done.
set -uo pipefail
cd /home/msj_team/Jacob/nk/AOT
source /home/msj_team/.conda/etc/profile.d/conda.sh 2>/dev/null || true
LOGD=logs/ov_sweep; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
PY=/home/msj_team/.conda/envs/AOT/bin/python3.10

G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751
STLLM_DONE=/home/msj_team/Jacob/nk/PruneVid/logs/stllm_fullset/ALL_DONE
KR=0.1

gpu_ok(){ local u=$1 m
  if [ "$u" != "$G3" ] && [ ! -f "$STLLM_DONE" ]; then return 1; fi
  m=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v U="$u" '$1==U{print $2}')
  [ -n "$m" ] && [ "$m" -lt 2500 ]
}
retained_of(){ rg -o "Retenion ratio : [0-9.]+" "$1" 2>/dev/null | awk -F': ' '{s+=$2;n++} END{if(n)printf "%.4f",s/n; else print "NA"}'; }
has_res(){ find "logs/repro/ov-7b/$1" -name '*results*.json' 2>/dev/null | grep -v submission | grep -q .; }

launch(){ # gpu task vtn label
  local gpu=$1 task=$2 vtn=$3 label=$4
  log "RUN ov $task VTN=$vtn KR=$KR ($label) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh ov "$task" "$vtn" "$KR" "$label" > "$LOGD/${task}_${label}.log" 2>&1
  log "DONE ov $task ($label) rc=$?"
}

# job list: "task vtn label"  (LVB@322 reused from probe -> skip)
JOBS=(
  "longvideobench_val_v 146 sw_r10"
  "longvideobench_val_v 208 sw_r15"
  "longvideobench_val_v 267 sw_r20"
  "videomme 146 sw_r10"
  "videomme 208 sw_r15"
  "videomme 267 sw_r20"
  "videomme 322 sw_r25"
)
declare -A BUSY_PID BUSY_JOB
i=0; N=${#JOBS[@]}
log "=== sweep start: $N jobs (GPU3 now; GPU0/GPU1 after ST-LLM done) ==="
while [ $i -lt $N ] || [ ${#BUSY_PID[@]} -gt 0 ]; do
  # reap finished
  for g in "${!BUSY_PID[@]}"; do
    if ! kill -0 "${BUSY_PID[$g]}" 2>/dev/null; then unset BUSY_PID[$g] BUSY_JOB[$g]; fi
  done
  # dispatch to free gpus (GPU3-only: GPU0/GPU1 reserved for ST-LLM to avoid collisions)
  for g in "$G3"; do
    [ $i -ge $N ] && break
    [ -n "${BUSY_PID[$g]:-}" ] && continue
    if gpu_ok "$g"; then
      set -- ${JOBS[$i]}; task=$1; vtn=$2; label=$3
      if has_res "${task}_${label}"; then log "SKIP done ${task}_${label}"; i=$((i+1)); continue; fi
      launch "$g" "$task" "$vtn" "$label" & BUSY_PID[$g]=$!; BUSY_JOB[$g]="${task}_${label}"
      i=$((i+1)); sleep 20
    fi
  done
  sleep 30
done

log "=== sweep results (score + measured retained) ==="
$PY - <<'PY'
import json, glob, re, os
def ret(task,label):
    f=f"logs/ov_sweep/{task}_{label}.log"
    if not os.path.exists(f): 
        f=f"logs/ov_vtn/probe25.log" if label=="sw_r25" and task=="longvideobench_val_v" else f
    if not os.path.exists(f): return None
    v=[float(m.group(1)) for m in re.finditer(r"Retenion ratio : ([0-9.]+)", open(f,errors='ignore').read())]
    return round(100*sum(v)/len(v),2) if v else None
rows=[("r10","146"),("r15","208"),("r20","267"),("r25","322")]
for pct,vtn in rows:
    for task,key,lbldir in [("longvideobench_val_v","lvb_acc,none",None),("videomme","videomme_perception_score,none",None)]:
        label=f"sw_{pct}"
        d=f"logs/repro/ov-7b/{task}_{label}"
        if pct=="r25" and task=="longvideobench_val_v": d="logs/repro/ov-7b/longvideobench_val_v_probe25"; label="probe25"
        js=[j for j in glob.glob(f"{d}/**/*result*.json",recursive=True) if 'submission' not in j]
        sc=None
        if js:
            r=json.load(open(sorted(js)[-1]))["results"]; k=list(r.keys())[0]; sc=r[k].get(key)
        print(f"  {pct}(VTN{vtn}) {task}: score={sc} retained%={ret(task,label)}")
PY
touch "$LOGD/ALL_DONE"
log "=== OV SWEEP ALL DONE ==="
