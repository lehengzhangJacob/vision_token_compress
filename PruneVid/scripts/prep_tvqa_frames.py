#!/usr/bin/env python3
"""Extract MVBench Episodic Reasoning frames from tvqa mp4 segments."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "DATAS/MVBench/json/episodic_reasoning.json"
SRC_DIR = Path(
    os.environ.get(
        "TVQA_SRC",
        "/home/msj_team/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots"
        "/a776e554280b99b70f00cc3eacd69a65e0727efc/tvqa/video_fps3_hq_segment",
    )
)
OUT_DIR = ROOT / "DATAS/MVBench/video/tvqa/frames_fps3_hq"


def extract_clip(mp4_path: Path, out_dir: Path, fps: float = 3.0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(src_fps / fps, 1.0)
    frame_id = 0
    saved = 0
    next_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id >= next_save - 1e-6:
            saved += 1
            cv2.imwrite(str(out_dir / f"{saved:05d}.jpg"), frame)
            next_save += step
        frame_id += 1

    cap.release()
    return saved


def main() -> None:
    items = json.loads(JSON_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0

    for item in tqdm(items, desc="tvqa frames"):
        name = item["video"]
        out = OUT_DIR / name
        if out.is_dir() and any(out.glob("*.jpg")):
            skipped += 1
            continue
        src = SRC_DIR / name
        if not src.exists():
            failed += 1
            tqdm.write(f"missing source: {src}")
            continue
        try:
            extract_clip(src, out, fps=float(item.get("fps", 3)))
            done += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"failed {name}: {exc}")

    print(f"done={done} skipped={skipped} failed={failed} out={OUT_DIR}")


if __name__ == "__main__":
    main()
