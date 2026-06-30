#!/bin/bash
# Table 1: 2 models x (baseline + R=25% + R=15%) x 4 benchmarks
set -uo pipefail
cd "$(dirname "$0")"
source vidcom2_env.sh

TASKS=(mvbench longvideobench_val_v mlvu_dev videomme)
RATIOS=(baseline 0.25 0.15)

for MODEL in ov vid; do
  for R in "${RATIOS[@]}"; do
    for TASK in "${TASKS[@]}"; do
      if [ "$TASK" = "mlvu_dev" ] && [ ! -f logs/.mlvu_ready ]; then
        echo "##### SKIP mlvu_dev (MLVU not ready) $(date '+%F %T') #####"
        continue
      fi
      LABEL="${R//./p}"
      echo "##### $(date '+%F %T') Table1 $MODEL $TASK R=$R #####"
      bash run_eval.sh "$MODEL" "$TASK" "$R" "$LABEL" || echo "!!!! FAILED $MODEL $TASK $R !!!!"
    done
  done
done
echo "##### TABLE1 ALL DONE $(date '+%F %T') #####"
