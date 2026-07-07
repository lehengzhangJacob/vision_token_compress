#!/usr/bin/env python
"""PruneVid qualitative visualization (paper teaser(b)/(c) + "selected tokens" style).

Per video, two figures:
  prunevid_<name>.png        4 rows: frames | static(blue)/dynamic(orange) | smooth
                             layer-10 question-to-visual attention | kept tokens
  prunevid_attn_layers_<name>.png   paper-teaser(c)-style: smooth attention maps
                             at layers 5/10/20/30 (post-prune layers cover only
                             the kept tokens)

Runs the released PLLaVA-7B PruneVid path with the repro eval config
(16 frames, 12x12 grid, tau=0.8, alpha=0.4, selected_layer=10).
Capture hooks are enabled via PRUNEVID_VIS_CAPTURE / PRUNEVID_VIS_ATTN_LAYERS.
"""
import argparse
import os
import sys

os.environ["PRUNEVID_VIS_CAPTURE"] = "1"
os.environ["PRUNEVID_VIS_ATTN_LAYERS"] = "5,10,20,30"

REPO = "/home/msj_team/Jacob/nk/PruneVid"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/msj_team/Jacob/nk/visualizations")
os.chdir(REPO)

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
import torchvision

from vis_utils import assemble_rows, draw_grid_lines, overlay_cells, overlay_heatmap, save_fig

from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates
from prunevid_core.vision_merge import VIS_CAPTURE

RESOLUTION = 672
NUM_FRAMES = 16
GRID = 12  # pooling_shape (16,12,12) -> 144 tokens/frame
TPF = GRID * GRID
BLUE = (70, 130, 220)
ORANGE = (255, 150, 40)
GREEN = (46, 204, 113)
ATTN_LAYERS = [5, 10, 20, 30]


def get_index(total, num_segments):
    seg_size = float(total - 1) / num_segments
    start = int(seg_size / 2)
    return np.array([start + int(np.round(seg_size * i)) for i in range(num_segments)])


def load_video(path):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idx = get_index(len(vr), NUM_FRAMES)
    raw = [vr[i].asnumpy() for i in idx]
    tf = torchvision.transforms.Resize(size=RESOLUTION)
    pils = [tf(Image.fromarray(f)) for f in raw]
    return raw, pils


def build_token_map(windows):
    """merged-token id -> (frame_lo, frame_hi_exclusive, patch_indices array)."""
    token_map = []
    static_map = np.zeros((sum(w["window_size"] for w in windows), TPF), bool)
    f0 = 0
    for win in windows:
        ws = win["window_size"]
        mask = win["mask"].numpy().astype(bool)
        static_pos = np.nonzero(mask)[0]
        dyn_pos = np.nonzero(~mask)[0]
        static_map[f0:f0 + ws, mask] = True

        sci = win["static_cluster_idx"]
        sci = sci.numpy() if sci is not None else None
        for j in range(win["static_size"]):
            patches = static_pos[sci == j] if sci is not None else static_pos[j:j + 1]
            token_map.append((f0, f0 + ws, patches))

        per_frame = win["dynamic_size"] // ws
        for fi in range(ws):
            dci = win["dynamic_cluster_idx"][fi]
            dci = dci.numpy() if dci is not None else None
            for j in range(per_frame):
                patches = dyn_pos[dci == j] if dci is not None else dyn_pos[j:j + 1]
                token_map.append((f0 + fi, f0 + fi + 1, patches))
        f0 += ws
    return token_map, static_map, f0


def broadcast_scores(token_map, T, scores, token_ids=None):
    """Broadcast per-merged-token scores back to a (T, TPF) patch map."""
    out = np.zeros((T, TPF), np.float32)
    ids = range(len(scores)) if token_ids is None else token_ids
    for s, tid in zip(scores, ids):
        lo, hi, patches = token_map[tid]
        out[lo:hi, patches] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--questions", nargs="*", default=[])
    ap.add_argument("--out-dir", default="/home/msj_team/Jacob/nk/visualizations/out")
    ap.add_argument("--show-frames", type=int, default=8)
    a = ap.parse_args()

    model, processor = load_pllava(
        "MODELS/pllava-7b", num_frames=NUM_FRAMES, use_lora=True, weight_dir="MODELS/pllava-7b",
        lora_alpha=14, pooling_shape=(16, 12, 12), selected_layer=10, alpha=0.4,
        tau=0.8, cluster_ratio=0.5, temporal_segment_ratio=0.25)
    model = model.to("cuda:0").eval()

    for vi, vp in enumerate(a.videos):
        q = a.questions[vi] if vi < len(a.questions) else "Describe what happens in this video."
        raw, pils = load_video(vp)
        conv = conv_templates["plain"].copy()
        conv.user_query(q, is_mm=True)
        VIS_CAPTURE.clear()
        answer, _ = pllava_answer(conv=conv, model=model, processor=processor, img_list=pils,
                                  do_sample=False, max_new_tokens=64, print_res=False,
                                  top_p=1.0, temperature=1.0)
        cap = dict(VIS_CAPTURE)
        token_map, static_map, T = build_token_map(cap["windows"])
        t2i = cap["t2i_scores"].numpy()
        keep_ids = sorted(cap["topk_indices"].numpy().tolist())
        score_map = broadcast_scores(token_map, T, t2i)
        keep_map = broadcast_scores(token_map, T, np.ones(len(keep_ids)), keep_ids) > 0

        sel = np.linspace(0, NUM_FRAMES - 1, a.show_frames).astype(int)
        name = os.path.splitext(os.path.basename(vp))[0]

        # ---------- figure 1: stages (frames / static-dyn / smooth attn / kept) ----------
        r_orig, r_sd, r_attn, r_keep = [], [], [], []
        for t in sel:
            f = raw[t]
            sm = static_map[t].reshape(GRID, GRID)
            b = overlay_cells(f, sm, BLUE, alpha=0.45)
            b = overlay_cells(b, ~sm, ORANGE, alpha=0.30)
            r_orig.append(f)
            r_sd.append(draw_grid_lines(b, GRID, GRID))
            r_attn.append(overlay_heatmap(f, score_map[t].reshape(GRID, GRID), alpha=0.5, smooth=True))
            k = keep_map[t].reshape(GRID, GRID)
            kb = overlay_cells(f, k, GREEN, alpha=0.35, dim_bg=0.25)
            r_keep.append(draw_grid_lines(kb, GRID, GRID))

        title = (f"PruneVid (PLLaVA-7B)  |  blue=static, orange=dynamic (tau=0.8)  |  "
                 f"layer-10 question-to-visual attention  |  kept {len(keep_ids)}/{len(t2i)} merged tokens "
                 f"(segments: {cap['window_list']})")
        footer = f"Q: {q}   A: {answer[:140]}"
        fig = assemble_rows([r_orig, r_sd, r_attn, r_keep],
                            row_labels=["frames", "static/dyn", "attention", "kept"],
                            title=title, footer=footer, label_w=78)
        save_fig(fig, os.path.join(a.out_dir, f"prunevid_{name}"))

        # ---------- figure 2: attention across layers (paper teaser (c) style) ----------
        layer_attn = cap.get("layer_attn", {})
        rows, labels = [], []
        sel4 = np.linspace(0, NUM_FRAMES - 1, min(6, a.show_frames)).astype(int)
        rows.append([raw[t] for t in sel4])
        labels.append("frames")
        for L in ATTN_LAYERS:
            if L not in layer_attn:
                continue
            v = layer_attn[L].numpy()
            if L <= 10:  # pre-prune: scores over all merged tokens
                m = broadcast_scores(token_map, T, v)
            else:        # post-prune: scores over kept tokens only
                m = broadcast_scores(token_map, T, v, keep_ids)
            rows.append([overlay_heatmap(raw[t], m[t].reshape(GRID, GRID), alpha=0.5, smooth=True)
                         for t in sel4])
            labels.append(f"Layer {L}")
        title2 = (f"PruneVid question-to-visual attention across LLM layers (PLLaVA-7B). "
                  f"Tokens are pruned after layer 10 (alpha=0.4), so layers 20/30 attend only to kept tokens.")
        fig2 = assemble_rows(rows, row_labels=labels, title=title2, footer=footer, label_w=78)
        save_fig(fig2, os.path.join(a.out_dir, f"prunevid_attn_layers_{name}"))
        print(f"[{name}] layers={sorted(layer_attn)} kept={len(keep_ids)}/{len(t2i)} answer: {answer[:150]}")


if __name__ == "__main__":
    main()
