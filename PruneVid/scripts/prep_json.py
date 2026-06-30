#!/usr/bin/env python3
"""Convert HF parquet annotations to PruneVid JSON format."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HF = Path("/home/msj_team/.cache/huggingface")
ROOT = Path(__file__).resolve().parents[1]


def _strip_option_prefix(opt: str) -> str:
    return re.sub(r"^[A-G]\.\s*", "", opt.strip())


def prep_videomme() -> None:
    snap = sorted((HF / "hub/datasets--lmms-lab--Video-MME/snapshots").glob("*/"))[0]
    parquet = snap / "videomme/test-00000-of-00001.parquet"
    df = pd.read_parquet(parquet)
    out_dir = ROOT / "DATAS/Video-MME/json"
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list] = {"short": [], "medium": [], "long": []}
    for row in df.itertuples(index=False):
        duration = str(row.duration).lower()
        if duration not in buckets:
            continue
        opts = list(row.options) if hasattr(row.options, "__iter__") else row.options
        candidates = [_strip_option_prefix(o) for o in opts]
        answer_letter = str(row.answer).strip().upper()
        answer_idx = ord(answer_letter[0]) - ord("A") if answer_letter else 0
        answer_text = candidates[answer_idx] if 0 <= answer_idx < len(candidates) else answer_letter
        buckets[duration].append(
            {
                "video": f"{row.videoID}.mp4",
                "question": row.question,
                "candidates": candidates,
                "answer": answer_text,
            }
        )

    for name, items in buckets.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
        print(f"wrote {path} ({len(items)} items)")


def prep_egoschema(es_snap: Path) -> None:
    mc = pd.read_parquet(es_snap / "MC/test-00000-of-00001.parquet")
    subset = pd.read_parquet(es_snap / "Subset/test-00000-of-00001.parquet")
    subset_ids = set(subset["question_idx"].astype(str))
    # MC test answers are withheld (all null); only Subset parquet has labels.
    subset_answers = {
        str(row.question_idx): int(row.answer)
        for row in subset.itertuples(index=False)
        if row.answer is not None
    }

    def _row_to_item(row, qid: str, is_subset: bool, ans_idx: int | None) -> dict:
        opts = list(row.option)
        candidates = [_strip_option_prefix(o) for o in opts]
        if ans_idx is not None and 0 <= ans_idx < len(candidates):
            answer_text = candidates[ans_idx]
        else:
            answer_text = None
        return {
            "video": f"{row.video_idx}.mp4",
            "question": row.question,
            "candidates": candidates,
            "answer": answer_text,
            "answer_idx": ans_idx,
            "question_idx": qid,
            "subset": is_subset,
        }

    fullset = []
    subset_only = []
    for row in mc.itertuples(index=False):
        qid = str(row.question_idx)
        if row.answer is not None:
            ans_idx = int(row.answer)
        elif qid in subset_answers:
            ans_idx = subset_answers[qid]
        else:
            ans_idx = None
        item = _row_to_item(row, qid, qid in subset_ids, ans_idx)
        fullset.append(item)
        if item["subset"]:
            subset_only.append(item)

    out_dir = ROOT / "DATAS/ego_schema/json"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "egoschema_fullset.json"
    full_path.write_text(json.dumps(fullset, ensure_ascii=False, indent=2))
    subset_path = out_dir / "egoschema_subset.json"
    subset_path.write_text(json.dumps(subset_only, ensure_ascii=False, indent=2))
    print(f"wrote {full_path} ({len(fullset)} items)")
    print(f"wrote {subset_path} ({len(subset_only)} items)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["videomme", "egoschema"])
    parser.add_argument("--es-snap", type=Path, default=None)
    args = parser.parse_args()
    if args.task == "videomme":
        prep_videomme()
    else:
        snap = args.es_snap or sorted((HF / "hub/datasets--lmms-lab--egoschema/snapshots").glob("*/"))[0]
        prep_egoschema(snap)


if __name__ == "__main__":
    main()
