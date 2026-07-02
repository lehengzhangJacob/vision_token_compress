#!/usr/bin/env bash
# ST-LLM + PruneVid on EgoSchema FULLSET (5031 items), sharded across GPU0 + GPU1.
# Fills report line 48 ("59.80 / --"). Predictions only; fullset labels are private,
# so accuracy comes from a subsequent validation-server submission.
# ~4 s/item => ~2.8 h per 2516-item shard (parallel).
set -uo pipefail
cd /home/msj_team/Jacob/nk/PruneVid/third_party/ST-LLM
export PYTHONPATH="./:${PYTHONPATH:-}"
export PRUNEVID=True
export PRUNEVID_ALPHA=0.4 PRUNEVID_TAU=0.8 PRUNEVID_TEMPORAL_SEGMENT_RATIO=0.25 PRUNEVID_CLUSTER_RATIO=0.5

PY=/home/msj_team/.conda/envs/stllm/bin/python
CKPT=/home/msj_team/Jacob/nk/PruneVid/MODELS/st-llm/instructblipbase_stllm_qa
VROOT=/home/msj_team/Jacob/nk/PruneVid/DATAS/ego_schema/videos
SHARDS=/home/msj_team/Jacob/nk/PruneVid/DATAS/ego_schema/json/_shards
OUT=/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-egoschema-fullset
LOGD=/home/msj_team/Jacob/nk/PruneVid/logs/stllm_fullset; mkdir -p "$OUT" "$LOGD"
OL="$LOGD/orchestrator.log"; log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OL"; }

G0=GPU-eed39de3-0f59-48a3-28a4-82d0ca5dbf0b
G1=GPU-a373a2c2-921b-3802-4d11-7bbec9effcf2
CFG=config/instructblipbase_stllm_qa.yaml

log "=== launch fullset shards: a->GPU0  b->GPU1 ==="
CUDA_VISIBLE_DEVICES="$G0" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" stllm/test/prunevid_ext/video_mcq_infer.py --cfg-path "$CFG" --ckpt-path "$CKPT" \
  --anno-path "$SHARDS/fullset_a.json" --video-root "$VROOT" \
  --output-dir "$OUT" --output-name stllm_fullset_a --num-frames 16 --gpu-id 0 \
  > "$LOGD/shard_a.log" 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES="$G1" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" stllm/test/prunevid_ext/video_mcq_infer.py --cfg-path "$CFG" --ckpt-path "$CKPT" \
  --anno-path "$SHARDS/fullset_b.json" --video-root "$VROOT" \
  --output-dir "$OUT" --output-name stllm_fullset_b --num-frames 16 --gpu-id 0 \
  > "$LOGD/shard_b.log" 2>&1 &
PB=$!
log "launched shard a pid=$PA (GPU0), shard b pid=$PB (GPU1)"
wait $PA; RA=$?
wait $PB; RB=$?
log "shards finished: a_rc=$RA b_rc=$RB"

log "=== merge shards ==="
"$PY" - "$OUT" <<'PY'
import json, sys, os
out=sys.argv[1]; res=[]
for s in ("a","b"):
    f=os.path.join(out, f"stllm_fullset_{s}.json")
    if os.path.exists(f):
        n=len(json.load(open(f)).get("res_list", []))
        res += json.load(open(f)).get("res_list", [])
        print(f"shard {s}: {n} items")
    else:
        print("MISSING", f)
json.dump({"acc_dict":{"note":"fullset labels private; submit to server"},"res_list":res},
          open(os.path.join(out,"stllm_prunevid_egoschema_fullset.json"),"w"), indent=2)
print("merged res_list:", len(res), "(expect 5031)")
PY
touch "$LOGD/ALL_DONE"
log "=== ST-LLM FULLSET PREDICTIONS DONE (next: submit to validation server) ==="
