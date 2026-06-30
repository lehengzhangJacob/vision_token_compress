#!/usr/bin/env bash
# Quick status check for PruneVid reproduction jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo ""
echo "=== Running eval processes ==="
ps aux | grep -E 'pllava_eval_(mvbench|videomme|egoschema|vcgbench)' | grep -v grep || echo "(none)"

echo ""
echo "=== Downloads ==="
for name in download_vcgbench prep_tvqa_frames; do
  if [[ -f "logs/${name}.log" ]]; then
  echo "-- ${name} (last line) --"
  tail -1 "logs/${name}.log" 2>/dev/null | tr '\r' '\n' | tail -1
  fi
done

echo ""
echo "=== Benchmark progress (last tqdm line) ==="
for bench in mvbench videomme egoschema vcgbench; do
  if [[ -f "logs/${bench}.log" ]]; then
    prog=$(tail -1 "logs/${bench}.log" 2>/dev/null | tr '\r' '\n' | grep -E '%|/' | tail -1 || true)
    echo "${bench}: ${prog:-starting...}"
  fi
done

echo ""
echo "=== Saved results ==="
for d in test_results/pllava-7b-prunevid-*/**/all_results.json; do
  [[ -f "$d" ]] || continue
  n=$(python3 -c "import json; print(len(json.load(open('$d'))['result_list']))" 2>/dev/null || echo "?")
  echo "$d -> $n samples"
done
