#!/usr/bin/env bash
# Build PruneVid DATAS/ layout from existing HF caches (symlinks only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF="${HF_HOME:-/home/msj_team/.cache/huggingface}"
MV_SNAP="$HF/hub/datasets--OpenGVLab--MVBench/snapshots/a776e554280b99b70f00cc3eacd69a65e0727efc"
VM_DATA="$HF/videomme/data"
ES_VIDEOS="$HF/egoschema/videos"
ES_SNAP="$(ls -d "$HF/hub/datasets--lmms-lab--egoschema/snapshots"/*/ 2>/dev/null | head -1)"

mkdir -p DATAS/MVBench DATAS/Video-MME DATAS/ego_schema/json DATAS/VCGBench

# --- MVBench ---
ln -sfn "$MV_SNAP/json" DATAS/MVBench/json
mkdir -p DATAS/MVBench/video/star DATAS/MVBench/video/clevrer DATAS/MVBench/video \
  DATAS/MVBench/video/Moments_in_Time_Raw DATAS/MVBench/video/FunQA_test \
  DATAS/MVBench/video/perception DATAS/MVBench/video/sta DATAS/MVBench/video/scene_qa \
  DATAS/MVBench/video/tvqa
ln -sfn "$MV_SNAP/star/Charades_segment" DATAS/MVBench/video/star/Charades_v1_480
ln -sfn "$MV_SNAP/clevrer/video_validation" DATAS/MVBench/video/clevrer/video_validation
ln -sfn "$MV_SNAP/ssv2_video_mp4" DATAS/MVBench/video/ssv2_video
ln -sfn "$MV_SNAP/Moments_in_Time_Raw/videos" DATAS/MVBench/video/Moments_in_Time_Raw/videos
ln -sfn "$MV_SNAP/FunQA_test/test" DATAS/MVBench/video/FunQA_test/test
ln -sfn "$MV_SNAP/perception/videos" DATAS/MVBench/video/perception/videos
ln -sfn "$MV_SNAP/sta/sta_video_segment" DATAS/MVBench/video/sta/sta_video
ln -sfn "$MV_SNAP/scene_qa/video" DATAS/MVBench/video/scene_qa/video
ln -sfn "$MV_SNAP/nturgbd_convert" DATAS/MVBench/video/nturgbd
ln -sfn "$MV_SNAP/vlnqa" DATAS/MVBench/video/vlnqa
mkdir -p DATAS/MVBench/video/tvqa/frames_fps3_hq
# Episodic Reasoning frames: run scripts/prep_tvqa_frames.py if empty

# --- Video-MME ---
ln -sfn "$VM_DATA" DATAS/Video-MME/data
mkdir -p DATAS/Video-MME/json
python3 "$ROOT/scripts/prep_json.py" videomme

# --- EgoSchema ---
ln -sfn "$ES_VIDEOS" DATAS/ego_schema/videos
python3 "$ROOT/scripts/prep_json.py" egoschema --es-snap "$ES_SNAP"

echo "DATAS layout ready under $ROOT/DATAS"
ls -la DATAS/MVBench DATAS/Video-MME DATAS/ego_schema
