#!/bin/bash
# Resume VidCom2 reproduction: skip jobs that already have results.json.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee -a logs/run_missing.log) 2>&1

echo "===== VidCom2 run_missing start $(date) ====="
source vidcom2_env.sh

has_results() {
  local model=$1 task=$2 label=$3
  local base
  if [ "$model" = "ov" ]; then base=./logs/repro/ov-7b; else base=./logs/repro/vid-7b; fi
  find "${base}/${task}_${label}" -name "*results.json" 2>/dev/null | grep -q .
}

run_if_missing() {
  local model=$1 task=$2 ratio=$3 label=$4
  if has_results "$model" "$task" "$label"; then
    echo "##### SKIP (done) $model $task R=$ratio #####"
    return 0
  fi
  if [ "$task" = "mlvu_dev" ] && [ ! -f logs/.mlvu_ready ]; then
    echo "##### SKIP mlvu_dev (not ready) #####"
    return 0
  fi
  echo "##### RUN $model $task R=$ratio $(date '+%F %T') #####"
  bash run_eval.sh "$model" "$task" "$ratio" "$label" || echo "!!!! FAILED $model $task $ratio !!!!"
}

bash prefetch_data.sh || echo "!!!! prefetch_data failed !!!!"
bash prefetch_lvb_meta.sh || echo "!!!! prefetch_lvb_meta failed !!!!"

TASKS=(mvbench longvideobench_val_v mlvu_dev videomme)
RATIOS=(baseline 0.25 0.15)
for MODEL in ov vid; do
  for R in "${RATIOS[@]}"; do
    LABEL="${R//./p}"
    for TASK in "${TASKS[@]}"; do
      run_if_missing "$MODEL" "$TASK" "$R" "$LABEL"
    done
  done
done
echo "##### TABLE1 MISSING DONE $(date '+%F %T') #####"

bash run_table2.sh
bash run_efficiency.sh

if [ -d /home/msj_team/Jacob/nk/VidCom2-qwen ]; then
  if python -c "from transformers import Qwen2VLForConditionalGeneration" 2>/dev/null; then
    bash run_qwen_videomme.sh || echo "!!!! qwen failed !!!!"
  else
    echo "##### SKIP qwen (upgrade transformers: pip install 'transformers>=4.45') #####"
  fi
fi

python3 parse_results.py
echo "===== VidCom2 run_missing end $(date) ====="
