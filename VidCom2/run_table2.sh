#!/bin/bash
# Table 2: EgoSchema + PerceptionTest @ R=15% and R=25%
set -uo pipefail
cd "$(dirname "$0")"
source vidcom2_env.sh

TASKS=(egoschema perceptiontest_val_mc)
RATIOS=(0.25 0.15)

for R in "${RATIOS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    if [ "$TASK" = "perceptiontest_val_mc" ] && [ ! -f logs/.table2_data_ready ]; then
      echo "##### SKIP perceptiontest_val_mc (Table2 data not ready) $(date '+%F %T') #####"
      continue
    fi
    LABEL="${R//./p}"
    echo "##### $(date '+%F %T') Table2 ov $TASK R=$R #####"
    bash run_eval.sh ov "$TASK" "$R" "$LABEL" || echo "!!!! FAILED $TASK $R !!!!"
    if [ "$TASK" = "egoschema" ]; then
      python3 evaluate_egoschema_result.py --log_dir "./logs/repro/ov-7b/egoschema_${LABEL}" || true
    fi
  done
done
echo "##### TABLE2 ALL DONE $(date '+%F %T') #####"
