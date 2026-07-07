"""Shared helpers for token-selection/pruning visualizations.

All three methods (AOT / PruneVid / VidCom2) sample frames uniformly with
np.linspace over the full video, so we reuse one sampler and map per-frame
patch-grid masks back onto the frames with nearest-neighbor upsampling.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import decord
    decord.bridge.set_bridge("native")
except Exception:  # decord only needed for sampling, not rendering
    decord = None


# ---------------------------------------------------------------- sampling

def sample_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Uniformly sample num_frames RGB frames (same convention as the evals:
    np.linspace(0, total-1, num_frames) rounded to int)."""
    vr = decord.VideoReader(video_path, num_threads=2)
    idx = np.linspace(0, len(vr) - 1, num_frames).astype(int)
    return vr.get_batch(list(idx)).asnumpy()  # (T, H, W, 3) uint8


# ---------------------------------------------------------------- overlays

def _to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def dim_frame(frame: np.ndarray, factor: float = 0.35) -> np.ndarray:
    return (frame.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def upsample_grid(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbor upsample a (gh, gw) grid to (h, w)."""
    gh, gw = grid.shape
    ri = (np.arange(h) * gh // h).clip(0, gh - 1)
    ci = (np.arange(w) * gw // w).clip(0, gw - 1)
    return grid[np.ix_(ri, ci)]


def overlay_cells(frame: np.ndarray, grid_mask: np.ndarray, color, alpha: float = 0.65,
                  dim_bg: float | None = None) -> np.ndarray:
    """Blend `color` over cells where grid_mask is True. grid_mask is (gh, gw) bool.
    If dim_bg is set, non-masked pixels are dimmed by that factor first."""
    h, w = frame.shape[:2]
    m = upsample_grid(grid_mask.astype(bool), h, w)
    out = frame.astype(np.float32).copy()
    if dim_bg is not None:
        out[~m] *= dim_bg
    col = np.asarray(color, dtype=np.float32)
    out[m] = out[m] * (1 - alpha) + col * alpha
    return out.clip(0, 255).astype(np.uint8)


def overlay_heatmap(frame: np.ndarray, grid_scores: np.ndarray, alpha: float = 0.55,
                    cmap_name: str = "jet", smooth: bool = False) -> np.ndarray:
    """Overlay a (gh, gw) float score grid as a colormap heatmap.

    smooth=True upsamples with bicubic interpolation + Gaussian blur (paper-style
    smooth attention maps) instead of blocky nearest-neighbor cells."""
    import matplotlib.cm as cm
    h, w = frame.shape[:2]
    g = grid_scores.astype(np.float32)
    if g.max() > g.min():
        g = (g - g.min()) / (g.max() - g.min())
    else:
        g = np.zeros_like(g)
    if smooth:
        from PIL import ImageFilter
        im = Image.fromarray((g * 255).astype(np.uint8), mode="L")
        im = im.resize((w, h), Image.BICUBIC)
        im = im.filter(ImageFilter.GaussianBlur(radius=max(2, min(h, w) // 60)))
        up = np.asarray(im).astype(np.float32) / 255.0
    else:
        up = upsample_grid(g, h, w)
    heat = (cm.get_cmap(cmap_name)(up)[..., :3] * 255).astype(np.float32)
    out = frame.astype(np.float32) * (1 - alpha) + heat * alpha
    return out.clip(0, 255).astype(np.uint8)


def draw_grid_lines(frame: np.ndarray, gh: int, gw: int, color=(255, 255, 255), width: int = 1) -> np.ndarray:
    """Draw faint patch-grid lines so token cells are visible."""
    im = _to_pil(frame)
    d = ImageDraw.Draw(im, "RGBA")
    h, w = frame.shape[:2]
    c = (*color, 60)
    for r in range(1, gh):
        y = round(r * h / gh)
        d.line([(0, y), (w, y)], fill=c, width=width)
    for cidx in range(1, gw):
        x = round(cidx * w / gw)
        d.line([(x, 0), (x, h)], fill=c, width=width)
    return np.asarray(im)


# ---------------------------------------------------------------- figure assembly

def _font(size: int):
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def assemble_rows(rows, row_labels=None, title: str = "", cell_h: int = 168, pad: int = 6,
                  label_w: int = 0, footer: str = "") -> Image.Image:
    """rows: list of lists of np.ndarray frames (all rows same length).
    Each frame is resized to height cell_h keeping aspect. Returns PIL image."""
    n_cols = len(rows[0])
    resized = []
    for row in rows:
        rr = []
        for f in row:
            im = _to_pil(f)
            wnew = round(im.width * cell_h / im.height)
            rr.append(im.resize((wnew, cell_h), Image.BILINEAR))
        resized.append(rr)
    col_ws = [max(resized[r][c].width for r in range(len(rows))) for c in range(n_cols)]
    W = label_w + sum(col_ws) + pad * (n_cols + 1)
    title_h = 34 if title else 8
    footer_h = 30 if footer else 0
    H = title_h + len(rows) * (cell_h + pad) + pad + footer_h
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((pad + label_w, 8), title, fill="black", font=_font(16))
    y = title_h + pad
    for ri, row in enumerate(resized):
        if row_labels and label_w:
            d.text((6, y + cell_h // 2 - 8), row_labels[ri], fill="black", font=_font(13))
        x = label_w + pad
        for ci, im in enumerate(row):
            canvas.paste(im, (x + (col_ws[ci] - im.width) // 2, y))
            x += col_ws[ci] + pad
        y += cell_h + pad
    if footer:
        d.text((pad + label_w, H - footer_h + 4), footer, fill=(60, 60, 60), font=_font(13))
    return canvas


def save_fig(img: Image.Image, out_base: str):
    """Save PNG (+PDF) for a PIL image."""
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    img.save(out_base + ".png")
    img.save(out_base + ".pdf", "PDF", resolution=150)
    print("saved", out_base + ".png")
