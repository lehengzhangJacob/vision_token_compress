#!/bin/bash
# Efficiency: LLaVA-OV MVBench baseline vs VidCom2 R=25%
set -euo pipefail
cd "$(dirname "$0")"
source vidcom2_env.sh
export EFFICIENCY=1
if python -c "import flash_attn" 2>/dev/null; then
  export FLASH_ATTN=1
else
  echo "flash_attn not installed; efficiency run without FA2"
  export FLASH_ATTN=0
fi

echo "##### Efficiency baseline #####"
unset COMPRESSOR
bash run_eval.sh ov mvbench baseline eff_baseline

echo "##### Efficiency VidCom2 R=25% #####"
bash run_eval.sh ov mvbench 0.25 eff_r25

echo "##### EFFICIENCY DONE #####"
