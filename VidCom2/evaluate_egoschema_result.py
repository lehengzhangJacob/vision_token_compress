#!/usr/bin/env python3
"""Upload EgoSchema predictions from lmms-eval log dir to official validation server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

SERVER_URL = "https://validation-server.onrender.com/api/upload/"
PAPER = {"subset": 59.7, "fullset": 42.6}


def find_submission(log_dir: Path) -> dict:
    for p in sorted(log_dir.rglob("inference_results_egoschema_*.json"), reverse=True):
        data = json.loads(p.read_text())
        if isinstance(data, dict) and data:
            print(f"using submission file: {p} ({len(data)} entries)")
            return data
    samples = sorted(log_dir.rglob("*samples_egoschema*.jsonl"), reverse=True)
    if not samples:
        raise FileNotFoundError(f"No egoschema submission in {log_dir}")
    combined: dict[str, int] = {}
    for line in samples[0].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sub = row.get("egoschema_submission") or row.get("submission")
        if isinstance(sub, dict):
            combined.update(sub)
        elif "doc" in row and "filtered_resps" in row:
            vid = row["doc"].get("video_idx", "")
            pred = row["filtered_resps"][0] if row["filtered_resps"] else "A"
            idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(str(pred).strip()[:1].upper(), 0)
            combined[vid] = idx
    if not combined:
        raise ValueError(f"Could not build submission from {log_dir}")
    print(f"built submission from jsonl: {len(combined)} entries")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, required=True)
    args = parser.parse_args()
    log_dir = args.log_dir.resolve()
    submission = find_submission(log_dir)
    # Server expects int values 0-4 keyed by video id
    payload = {k: int(v) for k, v in submission.items()}
    print(f"uploading {len(payload)} predictions...")
    resp = requests.post(SERVER_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=180)
    print(resp.status_code, resp.text[:500])
    out = log_dir / "validation_server_response.json"
    try:
        body = resp.json()
    except Exception:
        body = {"status_code": resp.status_code, "text": resp.text}
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    if resp.status_code != 200:
        sys.exit(1)


if __name__ == "__main__":
    main()
