#!/usr/bin/env bash
# Self-launching MLVU eval. Waits for (1) the refetch to finish (1122 videos on disk)
# and (2) GPU0/GPU1 to be free (ST-LLM fullset done), then runs the 4 MLVU evals
# (ov/vid x R=25%/15%) two-per-GPU-sequential. Fills VidCom2 Table 2 MLVU + Average.
set -uo pipefail
cd /home/msj_team/Jacob/nk/VidCom2
LOGD=logs/mlvu_eval; mkdir -p "$LOGD"; OL="$LOGD/orchestrator.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }
MLVU=/home/msj_team/.cache/huggingface/mlvu
G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2
STLLM_DONE=/home/msj_team/Jacob/nk/PruneVid/logs/stllm_fullset/ALL_DONE

# ---- Phase 1: wait for the refetch to deliver all 1122 videos ----
log "=== phase 1: waiting for MLVU refetch (need 1122 videos) ==="
while :; do
  n=$(ls "$MLVU"/*.mp4 2>/dev/null | wc -l)
  [ "$n" -ge 1122 ] && { log "videos ready: $n"; break; }
  sleep 180
done

# ---- Phase 2: wait for GPU0/GPU1 free AND ST-LLM finished (no collisions) ----
gpu_free(){ local u=$1 m
  [ -f "$STLLM_DONE" ] || return 1
  m=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v U="$u" '$1==U{print $2}')
  [ -n "$m" ] && [ "$m" -lt 2500 ]
}
log "=== phase 2: waiting for GPU0+GPU1 (ST-LLM fullset done) ==="
while ! { gpu_free "$G0" && gpu_free "$G1"; }; do sleep 120; done
log "GPU0+GPU1 free"

# ---- Phase 3: 4 evals, GPU0 does ov{25,15}, GPU1 does vid{25,15} (parallel across GPUs) ----
one(){ # gpu model ratio label
  log "RUN $2 mlvu_dev R=$3 ($4) on ${1:0:20}"
  GPUS="$1" NPROC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    bash run_eval.sh "$2" mlvu_dev "$3" "$4" > "$LOGD/$2_$4.log" 2>&1
  log "DONE $2 $4 rc=$?"
}
( one "$G0" ov 0.25 mlvu_0p25; one "$G0" ov 0.15 mlvu_0p15 ) & P0=$!
( one "$G1" vid 0.25 mlvu_0p25; one "$G1" vid 0.15 mlvu_0p15 ) & P1=$!
wait $P0 $P1

# ---- results ----
log "=== MLVU results (mlvu_dev accuracy) ==="
/home/msj_team/.conda/envs/VidCom2/bin/python3.10 - <<'PY'
import glob, json, os
for m,tag in [("ov-7b","LLaVA-OV"),("vid-7b","LLaVA-Video")]:
    for lbl,r in [("mlvu_0p25","R=25%"),("mlvu_0p15","R=15%")]:
        d=f"logs/repro/{m}/mlvu_dev_{lbl}"
        js=[j for j in glob.glob(f"{d}/**/*result*.json",recursive=True) if 'submission' not in j]
        sc=None
        if js:
            r0=json.load(open(sorted(js)[-1]))["results"]; k=list(r0.keys())[0]
            sc=r0[k].get("mlvu_percep_score,none") or r0[k].get("accuracy,none") or list(r0[k].values())[0]
        print(f"  {tag} {r}: MLVU={sc}")
PY
touch "$LOGD/ALL_DONE"
log "=== MLVU EVAL ALL DONE ==="
