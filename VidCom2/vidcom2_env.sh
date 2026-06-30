# Source before any VidCom2 run:  source vidcom2_env.sh
if [ -f /home/msj_team/Jacob/0/env.sh ]; then
  # Required for nohup / non-interactive shells (HF xethub CDN).
  source /home/msj_team/Jacob/0/env.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate VidCom2

# With local proxy, direct huggingface.co is more reliable than hf-mirror (SSL drops).
if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
  unset HF_ENDPOINT
else
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
fi
export HF_HOME=/home/msj_team/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

export SIGLIP_CACHE_DIR=/home/huggingface/siglip-so400m-patch14-384
export TOKENIZERS_PARALLELISM=false
export VIDCOM2_ROOT=/home/msj_team/Jacob/nk/VidCom2
export PYTHONPATH=$VIDCOM2_ROOT:$VIDCOM2_ROOT/lmms-eval:${PYTHONPATH:-}
cd "$VIDCOM2_ROOT"

# PruneVid EgoSchema often on GPU 0; use GPU 1 until free.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
