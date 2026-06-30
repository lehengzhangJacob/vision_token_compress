#!/usr/bin/env python3
"""Parse PruneVid eval outputs and compare to paper Table 1 (PLLaVA w/ PruneVid)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = {
    "mvbench": 47.6,
    "videomme": 45.3,
    "egoschema_subset": 49.0,
    "egoschema_fullset": 42.6,
    "vcgbench_avg": 2.98,
}

RESULT_DIRS = {
    "mvbench": ROOT / "test_results/pllava-7b-prunevid-mvbench/mvbench",
    "videomme": ROOT / "test_results/pllava-7b-prunevid-videomme/videomme",
    "egoschema": ROOT / "test_results/pllava-7b-prunevid-egoschema/egoschema",
    "vcgbench": ROOT / "test_results/pllava-7b-prunevid-vcgbench/vcgbench",
}


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_mc_benchmark(save_dir: Path) -> float | None:
    data = load_json(save_dir / "upload_leaderboard.json")
    if not data:
        return None
    return float(data.get("Avg", data.get("avg")))


def parse_egoschema_server(save_dir: Path) -> tuple[float | None, float | None]:
    """Read subset/fullset scores from official validation server response."""
    resp_path = save_dir / "validation_server_response.json"
    if not resp_path.exists():
        return None, None
    data = load_json(resp_path)
    if not isinstance(data, dict):
        return None, None

    def pick(d: dict, *keys: str) -> float | None:
        for k in keys:
            if k in d and d[k] is not None:
                return float(d[k])
        return None

    subset = pick(data, "subset_accuracy", "subset", "subset_score", "Subset")
    fullset = pick(data, "fullset_accuracy", "fullset", "fullset_score", "Fullset", "accuracy")
    return subset, fullset


def parse_vcgbench(save_dir: Path) -> float | None:
    for name in ("final_scores.json", "scores.json", "upload_leaderboard.json"):
        data = load_json(save_dir / name)
        if isinstance(data, dict):
            if "Avg" in data:
                return float(data["Avg"])
            if "avg" in data:
                return float(data["avg"])
    return None


def main() -> None:
    mv = parse_mc_benchmark(RESULT_DIRS["mvbench"])
    vm = parse_mc_benchmark(RESULT_DIRS["videomme"])
    es_subset, es_fullset = parse_egoschema_server(RESULT_DIRS["egoschema"])
    vcg = parse_vcgbench(RESULT_DIRS["vcgbench"])

    metrics = [
        ("MVBench", mv, PAPER["mvbench"]),
        ("VideoMME", vm, PAPER["videomme"]),
        ("EgoSchema subset (server)", es_subset, PAPER["egoschema_subset"]),
        ("EgoSchema fullset (server)", es_fullset, PAPER["egoschema_fullset"]),
        ("VCGBench Avg", vcg, PAPER["vcgbench_avg"]),
    ]

    lines = [
        "# PruneVid Reproduction Report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "| Benchmark | Reproduced | Paper | Delta |",
        "|---|---|---|---|",
    ]
    for name, got, paper in metrics:
        if got is None:
            lines.append(f"| {name} | pending | {paper} | |")
        else:
            delta = got - paper
            lines.append(f"| {name} | {got:.2f} | {paper} | {delta:+.2f} |")

    report = "\n".join(lines) + "\n"
    out = ROOT / "REPRODUCTION_REPORT.md"
    out.write_text(report)
    print(report)
    sys.exit(0)


if __name__ == "__main__":
    main()
