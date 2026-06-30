#!/usr/bin/env python3
"""Upload EgoSchema fullset predictions to the official validation server."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
PAPER = {"subset": 49.0, "fullset": 42.6}
SERVER_URL = "https://validation-server.onrender.com/api/upload/"


def parse_option_index(pred: str) -> int:
    pred = pred.strip()
    m = re.search(r"\(([A-Ea-e])\)", pred)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    m = re.search(r"\b([A-Ea-e])\b", pred)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    if len(pred) >= 2 and pred[0] == "(" and pred[1] in "ABCDEabcde":
        return ord(pred[1].upper()) - ord("A")
    raise ValueError(f"Cannot parse option from prediction: {pred!r}")


def build_submission(result_dir: Path) -> dict[str, int]:
    results_path = result_dir / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}; run pllava_eval_egoschema first.")

    data_list = json.loads(results_path.read_text())["result_list"]
    submission: dict[str, int] = {}
    failed = 0
    for row in data_list:
        vid = Path(row["video_path"]).name.rsplit(".", 1)[0]
        try:
            submission[vid] = parse_option_index(row["pred"])
        except ValueError:
            failed += 1
            submission[vid] = 0
    if failed:
        print(f"warning: failed to parse {failed} predictions, defaulted to A", file=sys.stderr)
    print(f"submission entries: {len(submission)}")
    return submission


def upload(submission: dict[str, int]) -> requests.Response:
    return requests.post(
        SERVER_URL,
        headers={"Content-Type": "application/json"},
        json=submission,
        timeout=120,
    )


def save_server_response(result_dir: Path, response: requests.Response) -> Path:
    out = result_dir / "validation_server_response.json"
    try:
        payload = response.json()
    except Exception:
        payload = {"status_code": response.status_code, "text": response.text}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


def write_report(result_dir: Path, response: requests.Response) -> Path:
    report = result_dir / "egoschema_server_scores.md"
    lines = [
        "# EgoSchema Official Server Scores",
        "",
        f"- Server: `{SERVER_URL}`",
        f"- HTTP status: {response.status_code}",
        "",
        "## Paper targets (PLLaVA w/ PruneVid)",
        f"- Subset: {PAPER['subset']}",
        f"- Fullset: {PAPER['fullset']}",
        "",
        "## Server response",
        "",
        "```",
        response.text.strip(),
        "```",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=ROOT / "test_results/pllava-7b-prunevid-egoschema/egoschema",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()

    submission = build_submission(result_dir)
    print(f"uploading {len(submission)} predictions from {result_dir} ...")
    response = upload(submission)
    print(f"status: {response.status_code}")
    print(response.text)

    save_server_response(result_dir, response)
    write_report(result_dir, response)

    if response.status_code != 200:
        sys.exit(1)


if __name__ == "__main__":
    main()
