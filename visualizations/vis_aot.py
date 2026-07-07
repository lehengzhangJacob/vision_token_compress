#!/usr/bin/env python
"""AOT qualitative visualization (paper Fig. "vis_cases" style).

Top row: original sampled frames. Bottom row: token grid where
green = Local anchors (shallow-layer window CLS attention) and
orange = Global anchors (deep-layer CLS attention); unselected patches dimmed.

Runs the real LLaVA-OneVision-7B AOT path (same env config as eval_ov-7b.sh);
anchor indices are captured via llava_arch.VIS_CAPTURE (AOT_VIS_CAPTURE=1).
"""
import argparse
import os
import sys

os.environ.setdefault("VISUAL_TOKEN_NUM", "126")
os.environ.setdefault("KEEP_RATIO", "0.3")
os.environ.setdefault("INTER_COMPRESS", "True")
os.environ.setdefault("GLOBAL_RATIO", "0.5")
os.environ.setdefault("INTRA_SCALE", "1.0")
os.environ.setdefault("INTER_SCALE", "1.0")
os.environ["AOT_VIS_CAPTURE"] = "1"

sys.path.insert(0, "/home/msj_team/Jacob/nk/AOT/LLaVA-NeXT")
sys.path.insert(0, "/home/msj_team/Jacob/nk/visualizations")

import numpy as np
import torch

from vis_utils import (assemble_rows, dim_frame, draw_grid_lines, overlay_cells,
                       sample_frames, save_fig)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
import llava.model.llava_arch as llava_arch

MODEL_PATH = "/home/msj_team/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"
GRID = 27  # siglip-so400m-patch14-384 -> 27x27 = 729 tokens/frame
GREEN = (46, 204, 113)
ORANGE = (255, 140, 0)


def render(frames, local_idx, global_idx, sel, title, footer):
    top, bottom = [], []
    for t in sel:
        f = frames[t]
        lm = np.zeros(GRID * GRID, bool)
        lm[local_idx[t]] = True
        gm = np.zeros(GRID * GRID, bool)
        gm[global_idx[t]] = True
        lm, gm = lm.reshape(GRID, GRID), gm.reshape(GRID, GRID)
        b = overlay_cells(f, lm | gm, (0, 0, 0), alpha=0.0, dim_bg=0.22)
        b = overlay_cells(b, lm, GREEN, alpha=0.45)
        b = overlay_cells(b, gm, ORANGE, alpha=0.45)
        b = draw_grid_lines(b, GRID, GRID)
        top.append(f)
        bottom.append(b)
    return assemble_rows([top, bottom], row_labels=["frames", "anchors"],
                         title=title, footer=footer, label_w=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--questions", nargs="*", default=[])
    ap.add_argument("--out-dir", default="/home/msj_team/Jacob/nk/visualizations/out")
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--show-frames", type=int, default=8)
    a = ap.parse_args()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava_qwen", device_map="cuda:0", multimodal=True,
        overwrite_config={"mm_spatial_pool_stride": 2, "mm_spatial_pool_mode": "bilinear"},
        attn_implementation="sdpa")
    model.eval()

    vtn = os.environ["VISUAL_TOKEN_NUM"]
    for vi, vp in enumerate(a.videos):
        q = a.questions[vi] if vi < len(a.questions) else "Describe what happens in this video."
        frames = sample_frames(vp, a.num_frames)
        pix = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()

        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + q)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).cuda()

        llava_arch.VIS_CAPTURE.clear()
        with torch.inference_mode():
            out = model.generate(input_ids, images=[pix], modalities=["video"],
                                 do_sample=False, temperature=0, max_new_tokens=64)
        answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

        cap = llava_arch.VIS_CAPTURE
        local_idx = cap["local_indices"].numpy()
        global_idx = cap["global_indices"].numpy()
        sel = np.linspace(0, local_idx.shape[0] - 1, a.show_frames).astype(int)

        name = os.path.splitext(os.path.basename(vp))[0]
        title = (f"AOT token anchors  |  green = Local (shallow window CLS attn), "
                 f"orange = Global (deep CLS attn)  |  VTN={vtn}/729 per frame")
        footer = f"Q: {q}   A: {answer[:150]}"
        fig = render(frames, local_idx, global_idx, sel, title, footer)
        save_fig(fig, os.path.join(a.out_dir, f"aot_anchors_{name}"))
        print(f"[{name}] answer: {answer[:200]}")


if __name__ == "__main__":
    main()
