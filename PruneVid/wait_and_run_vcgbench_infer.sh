#!/usr/bin/env bash
# Wait for VCGBench videos, then run inference (nohup backup if prefetch dies before infer).
set -uo pipefail
cd "$(dirname "$0")"
source prunevid_env.sh

ZIP=DATAS/VCGBench/Test_Videos.zip
VIDEOS=DATAS/VCGBench/Videos
MARKER="${VIDEOS}/.unzipped"
SAVE=test_results/pllava-7b-prunevid-vcgbench/vcgbench/inference_results.json
EXPECTED_MIN=11000000000

echo "===== wait_and_run_vcgbench_infer start $(date) ====="

while pgrep -f "prefetch_vcgbench.sh" >/dev/null 2>&1; do
  sz=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  echo "$(date) prefetch running, zip=${sz} bytes"
  sleep 60
done

if [ -f "$SAVE" ]; then
  echo "$(date) inference_results.json exists — skip"
  exit 0
fi

if pgrep -f "pllava_eval_vcgbench" >/dev/null 2>&1; then
  echo "$(date) pllava_eval_vcgbench already running — skip"
  exit 0
fi

if [ ! -f "$MARKER" ]; then
  sz=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  if [ "$sz" -lt "$EXPECTED_MIN" ]; then
    echo "$(date) ERROR: zip incomplete (${sz} bytes) and no .unzipped marker"
    exit 1
  fi
  echo "$(date) unzipping $ZIP"
  mkdir -p "$VIDEOS"
  unzip -o "$ZIP" -d "$VIDEOS"
  bench="${VIDEOS}/Benchmarking"
  if [ ! -d "$bench" ]; then
    mkdir -p "$bench"
    for f in "$VIDEOS"/*.mp4 "$VIDEOS"/*.mkv; do
      [ -f "$f" ] && mv "$f" "$bench/"
    done
  fi
  echo ok >"$MARKER"
fi

echo "===== inference start $(date) ====="
CUDA_VISIBLE_DEVICES=2 bash run_vcgbench_infer.sh
echo "===== done $(date) ====="
