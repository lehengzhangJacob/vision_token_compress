#!/usr/bin/env bash
# Quick sanity check: load PLLaVA+PruneVid and run 3 MVBench samples.
set -euo pipefail
cd "$(dirname "$0")"
source prunevid_env.sh

export PRUNEVID_SMOKE_LIMIT=3

python - <<'PY'
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates
from tasks.eval.mvbench import MVBenchDataset, check_ans

limit = int(os.environ.get("PRUNEVID_SMOKE_LIMIT", "3"))
model_dir = "MODELS/pllava-7b"

model, processor = load_pllava(
    model_dir, num_frames=16, use_lora=True, weight_dir=model_dir,
    lora_alpha=14, pooling_shape=(16, 12, 12),
    selected_layer=10, alpha=0.4, tau=0.8,
    temporal_segment_ratio=0.25, cluster_ratio=0.5,
)
model = model.to("cuda").eval()

ds = MVBenchDataset(num_segments=16)
ds.data_list = ds.data_list[:limit]
print(f"smoke on {len(ds.data_list)} samples")

correct = 0
for i, ex in enumerate(ds):
    conv = conv_templates["eval_mvbench"].copy()
    conv.user_query(ex["question"] + "\nOnly give the best option.", is_mm=True)
    pred, _ = pllava_answer(
        conv=conv, model=model, processor=processor,
        img_list=ex["video_pils"], do_sample=False, max_new_tokens=32,
    )
    ok = check_ans(pred=pred, gt=ex["answer"])
    correct += int(ok)
    print(f"[{i}] pred={pred[:40]!r} gt={ex['answer'][:20]!r} ok={ok}")

print(f"smoke acc {correct}/{len(ds.data_list)}")
PY
