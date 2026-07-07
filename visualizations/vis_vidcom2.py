#!/usr/bin/env python
"""VidCom2 qualitative visualization (paper Fig. "frame_score_vis" style).

Per video: a strip of sampled frames, per-frame uniqueness bars (taller/darker =
more unique -> more tokens kept), the dynamic retention rate r_t, and an extra
row showing which 14x14 patches were actually kept per frame.

Runs LLaVA-OneVision-7B with the VidCom2 compressor (R_RATIO=0.25, same as the
Table-2 repro config); scores/indices captured via VIDCOM2_VIS_CAPTURE=1.
"""
import argparse
import os
import sys
import types

os.environ["VIDCOM2_VIS_CAPTURE"] = "1"
os.environ["COMPRESSOR"] = "vidcom2"
os.environ.setdefault("R_RATIO", "0.25")

sys.path.insert(0, "/home/msj_team/Jacob/nk/VidCom2")
sys.path.insert(0, "/home/msj_team/Jacob/nk/visualizations")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vis_utils import draw_grid_lines, overlay_cells, sample_frames

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from token_compressor.vidcom2.models.llava import cus_prepare_inputs_labels_for_multimodal
from token_compressor.vidcom2.vidcom2 import VIS_CAPTURE

MODEL_PATH = "/home/msj_team/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"
GRID = 14  # llava_ov: 196 pooled tokens per frame
GREEN = (46, 204, 113)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out-dir", default="/home/msj_team/Jacob/nk/visualizations/out")
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--show-frames", type=int, default=16)
    a = ap.parse_args()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava_qwen", device_map="cuda:0", multimodal=True,
        overwrite_config={"mm_spatial_pool_stride": 2, "mm_spatial_pool_mode": "bilinear"},
        attn_implementation="sdpa")
    model.eval()
    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        cus_prepare_inputs_labels_for_multimodal, model)

    rr = float(os.environ["R_RATIO"])
    for vp in a.videos:
        frames = sample_frames(vp, a.num_frames)
        pix = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nDescribe this video.")
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).cuda()
        VIS_CAPTURE.clear()
        with torch.inference_mode():
            out = model.generate(input_ids, images=[pix], modalities=["video"],
                                 do_sample=False, temperature=0, max_new_tokens=48)
        answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

        uniq = VIS_CAPTURE["uniqueness"].numpy()
        scales = VIS_CAPTURE["scales"].numpy()
        indices = [i.numpy() for i in VIS_CAPTURE["indices"]]
        T = len(uniq)
        u = (uniq - uniq.min()) / (uniq.max() - uniq.min() + 1e-9)

        sel = np.linspace(0, T - 1, a.show_frames).astype(int)
        n = len(sel)
        cmap = plt.get_cmap("plasma")

        fig = plt.figure(figsize=(1.55 * n, 6.2))
        gs = fig.add_gridspec(3, n, height_ratios=[1.5, 1.0, 1.5], hspace=0.25, wspace=0.04)

        for c, t in enumerate(sel):
            ax = fig.add_subplot(gs[0, c])
            ax.imshow(frames[t])
            ax.set_title(f"f{t}", fontsize=8)
            ax.axis("off")

        axb = fig.add_subplot(gs[1, :])
        colors = [cmap(0.15 + 0.8 * u[t]) for t in range(T)]
        axb.bar(np.arange(T), u, color=colors, width=0.82)
        for t in sel:
            axb.text(t, u[t] + 0.03, f"{scales[t] * 100:.0f}%", ha="center", fontsize=7, rotation=90)
        axb.set_xlim(-0.6, T - 0.4)
        axb.set_ylim(0, 1.25)
        axb.set_ylabel("frame uniqueness", fontsize=9)
        axb.set_xlabel("frame index (labels above bars = retained token ratio $r_t$)", fontsize=9)
        axb.spines[["top", "right"]].set_visible(False)

        for c, t in enumerate(sel):
            ax = fig.add_subplot(gs[2, c])
            m = np.zeros(GRID * GRID, bool)
            m[indices[t]] = True
            ov = overlay_cells(frames[t], m.reshape(GRID, GRID), GREEN, alpha=0.35, dim_bg=0.25)
            ov = draw_grid_lines(ov, GRID, GRID)
            ax.imshow(ov)
            ax.set_title(f"kept {len(indices[t])}/196", fontsize=7)
            ax.axis("off")

        name = os.path.splitext(os.path.basename(vp))[0]
        fig.suptitle(f"VidCom$^2$ frame uniqueness & token allocation  (LLaVA-OV-7B, R={rr:.0%}, "
                     f"mean kept {np.mean([len(i) for i in indices]) / 1.96:.1f}%)", fontsize=12)
        base = os.path.join(a.out_dir, f"vidcom2_uniqueness_{name}")
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"saved {base}.png  | answer: {answer[:150]}")


if __name__ == "__main__":
    main()
