#!/usr/bin/env python
"""VidCom2 qualitative visualization (paper Fig. "frame_score_vis" style).

Per video: ALL 32 sampled frames in two staggered rows (even frames on top,
odd frames offset by half a cell below), per-frame uniqueness bars (taller/
darker = more unique -> more tokens kept) with the dynamic retention rate r_t,
and the kept 14x14 patches per frame in the same staggered two-row layout.

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
        cmap = plt.get_cmap("plasma")

        # All T frames, two staggered rows: even frames on top (cols t..t+2),
        # odd frames below shifted by half a cell -> brick-like interleaving.
        ncols = T + 1
        fig = plt.figure(figsize=(0.82 * T, 10.0))
        gs = fig.add_gridspec(5, ncols, height_ratios=[1.25, 1.25, 0.9, 1.25, 1.25],
                              hspace=0.35, wspace=0.05)

        def staggered(row0, imgs, titles):
            for t in range(T):
                r = row0 if t % 2 == 0 else row0 + 1
                ax = fig.add_subplot(gs[r, t:t + 2])
                ax.imshow(imgs[t])
                ax.set_title(titles[t], fontsize=6.5, pad=1.5)
                ax.axis("off")

        staggered(0, frames, [f"f{t}" for t in range(T)])

        axb = fig.add_subplot(gs[2, :])
        colors = [cmap(0.15 + 0.8 * u[t]) for t in range(T)]
        axb.bar(np.arange(T), u, color=colors, width=0.82)
        for t in range(T):
            axb.text(t, u[t] + 0.03, f"{scales[t] * 100:.0f}%", ha="center", fontsize=6.5, rotation=90)
        axb.set_xlim(-0.6, T - 0.4)
        axb.set_ylim(0, 1.3)
        axb.set_xticks(np.arange(0, T, 2))
        axb.set_ylabel("frame uniqueness", fontsize=9)
        axb.set_xlabel("frame index (labels above bars = retained token ratio $r_t$)", fontsize=9)
        axb.spines[["top", "right"]].set_visible(False)

        kept_imgs, kept_titles = [], []
        for t in range(T):
            m = np.zeros(GRID * GRID, bool)
            m[indices[t]] = True
            ov = overlay_cells(frames[t], m.reshape(GRID, GRID), GREEN, alpha=0.35, dim_bg=0.25)
            kept_imgs.append(draw_grid_lines(ov, GRID, GRID))
            kept_titles.append(f"{len(indices[t])}/196")
        staggered(3, kept_imgs, kept_titles)

        name = os.path.splitext(os.path.basename(vp))[0]
        fig.suptitle(f"VidCom$^2$ frame uniqueness & token allocation  (LLaVA-OV-7B, R={rr:.0%}, "
                     f"all {T} sampled frames, mean kept {np.mean([len(i) for i in indices]) / 1.96:.1f}%)",
                     fontsize=13)
        base = os.path.join(a.out_dir, f"vidcom2_uniqueness_{name}")
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"saved {base}.png  | answer: {answer[:150]}")


if __name__ == "__main__":
    main()
