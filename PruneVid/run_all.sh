#!/usr/bin/env bash
# Launch all PruneVid PLLaVA benchmarks in background.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

for bench in mvbench videomme egoschema; do
  nohup bash "run_${bench}.sh" > "logs/${bench}.log" 2>&1 &
  echo "started ${bench} pid $!"
done

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  nohup bash run_vcgbench.sh > logs/vcgbench.log 2>&1 &
  echo "started vcgbench pid $!"
else
  echo "skip vcgbench (set OPENAI_API_KEY to enable GPT eval)"
fi
