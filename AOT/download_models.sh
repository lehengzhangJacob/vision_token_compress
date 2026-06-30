#!/bin/bash
source /home/msj_team/Jacob/nk/AOT/aot_env.sh
echo "==== downloading llava-onevision-qwen2-7b-ov ===="
huggingface-cli download lmms-lab/llava-onevision-qwen2-7b-ov --resume-download
echo "==== downloading LLaVA-Video-7B-Qwen2 ===="
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --resume-download
echo "==== ALL MODELS DONE ===="
