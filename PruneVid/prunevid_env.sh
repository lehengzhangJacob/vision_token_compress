# Source before any PruneVid run:  source prunevid_env.sh
if [ -f /home/msj_team/Jacob/0/env.sh ]; then
  source /home/msj_team/Jacob/0/env.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PruneVid

if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
  unset HF_ENDPOINT
else
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
fi
export HF_HOME=/home/msj_team/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# Idle card in nvtop/nvidia-smi is "GPU 3" (PCI CA:00.0).
# CUDA skips broken nvidia-smi GPU 2, so use CUDA_VISIBLE_DEVICES=2 (NOT 3).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TOKENIZERS_PARALLELISM=false
export PRUNEVID_ROOT=/home/msj_team/Jacob/nk/PruneVid
export PYTHONPATH=$PRUNEVID_ROOT:${PYTHONPATH:-}
cd "$PRUNEVID_ROOT"
