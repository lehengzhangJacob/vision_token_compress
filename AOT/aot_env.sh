# Source this before any AOT run:  source aot_env.sh
if [ -f /home/msj_team/Jacob/0/env.sh ]; then
  source /home/msj_team/Jacob/0/env.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate AOT

# Prefer direct huggingface.co through proxy; fall back to mirror when no proxy is set.
if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
  unset HF_ENDPOINT
else
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
fi
export HF_HOME=/home/msj_team/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# Local siglip checkpoint (already on server)
export SIGLIP_CACHE_DIR=/home/huggingface/siglip-so400m-patch14-384

export TOKENIZERS_PARALLELISM=false
export AOT_ROOT=/home/msj_team/Jacob/nk/AOT
cd "$AOT_ROOT"
