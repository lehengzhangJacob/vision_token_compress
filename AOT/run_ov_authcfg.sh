#!/usr/bin/env bash
# "完全一致" re-run: OV VideoMME + LongVideoBench with the AUTHORS' documented
# long-video config (eval_ov-7b.sh): VISUAL_TOKEN_NUM=108, KEEP_RATIO=0.1,
# INTER_COMPRESS=True, GLOBAL_RATIO=0.5, INTRA_SCALE=INTER_SCALE=1.0, 32 frames.
# Waits for the MLVU evals to release GPU1/GPU3, then runs both benchmarks in
# parallel FULLY OFFLINE (model + datasets already cached; broken proxy irrelevant).
set -uo pipefail
cd /home/msj_team/Jacob/nk/AOT
source /home/msj_team/.conda/etc/profile.d/conda.sh 2>/dev/null || true

LOGD=logs/ov_authcfg; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2   # nvidia-smi GPU1
G3=GPU-fd2e342a-3610-76c0-ef30-d087968b4751   # nvidia-smi GPU3
MLVU_DONE=/home/msj_team/Jacob/nk/VidCom2/logs/mlvu_final/ALL_DONE

gpu_free(){ # uuid -> 0 if mem < 1500 MiB
  local u=$1 m
  m=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v U="$u" '$1==U{print $2}')
  [ -n "$m" ] && [ "$m" -lt 1500 ]
}

log "=== waiting for MLVU to finish + GPU1/GPU3 free ==="
while :; do
  if [ -f "$MLVU_DONE" ] && gpu_free "$G1" && gpu_free "$G3"; then
    log "MLVU done and GPU1/GPU3 free -> proceeding"; break
  fi
  sleep 60
done

run_one(){ # gpu task label
  local gpu=$1 task=$2 label=$3
  log "RUN ov $task VTN=108 KR=0.1 ($label) on ${gpu:0:20}"
  GPUS="$gpu" NPROC=1 LIMIT="" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh ov "$task" 108 0.1 "$label" > "$LOGD/${task}_${label}.log" 2>&1
  local rc=$?
  log "DONE ov $task ($label) rc=$rc"
}

log "=== launch VideoMME (GPU1) + LongVideoBench (GPU3) in parallel ==="
run_one "$G1" videomme            auth108 &  A=$!
run_one "$G3" longvideobench_val_v auth108 & B=$!
wait $A $B

# --- summarize scores + measured retained% (authors' formula) ---
log "=== results ==="
python3 - >> "$OL" 2>&1 <<'PY'
import json, glob, re, statistics
def score(task):
    js = sorted(glob.glob(f"logs/repro/ov-7b/{task}_auth108/**/*result*.json", recursive=True))
    if not js: return None
    d = json.load(open(js[-1])); r = d["results"]; k = list(r.keys())[0]
    for kk,vv in r[k].items():
        if kk.endswith("score,none") or kk.endswith("acc,none"): return kk, vv
    return list(r[k].items())[:3]
def retained(task, label):
    vals=[]
    for f in glob.glob(f"logs/ov_authcfg/{task}_{label}.log"):
        for line in open(f, errors="ignore"):
            m = re.search(r"Retenion ratio : ([0-9.]+)", line)
            if m: vals.append(float(m.group(1)))
    return (round(100*statistics.mean(vals),2), len(vals)) if vals else (None,0)
for t in ["videomme","longvideobench_val_v"]:
    s = score(t); rr = retained(t,"auth108")
    print(f"  {t}: score={s}  retained%~{rr[0]} (n={rr[1]})")
PY

touch "$LOGD/ALL_DONE"
log "=== ALL DONE ==="
