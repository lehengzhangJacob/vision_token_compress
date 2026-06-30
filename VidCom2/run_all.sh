#!/bin/bash
# Full reproduction pipeline (sequential, nohup-friendly)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee -a logs/run_all.log) 2>&1

echo "===== VidCom2 reproduction start $(date) ====="

bash prefetch_data.sh || { echo "!!!!! TABLE1 PREFETCH FAILED — aborting $(date) !!!!!"; exit 1; }

if [ ! -f logs/.mlvu_ready ]; then
  if ! pgrep -f "prefetch_mlvu.sh" >/dev/null; then
    echo "===== Starting MLVU background download $(date) ====="
    nohup bash prefetch_mlvu.sh >> logs/prefetch_mlvu_nohup.log 2>&1 &
    sleep 15
  else
    echo "===== MLVU prefetch already running $(date) ====="
  fi
fi

if [ ! -f logs/.table2_data_ready ]; then
  if ! pgrep -f "prefetch_table2.sh" >/dev/null; then
    echo "===== Starting Table2 data background download $(date) ====="
    nohup bash prefetch_table2.sh >> logs/prefetch_table2_nohup.log 2>&1 &
  else
    echo "===== Table2 prefetch already running $(date) ====="
  fi
fi

bash run_table1_batch.sh
bash run_table2.sh
bash run_efficiency.sh

if [ -d /home/msj_team/Jacob/nk/VidCom2-qwen ]; then
  bash run_qwen_videomme.sh
fi

python3 parse_results.py
echo "===== VidCom2 reproduction end $(date) ====="
