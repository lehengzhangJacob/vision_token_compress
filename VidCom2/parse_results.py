#!/usr/bin/env python3
"""Parse VidCom2 reproduction logs and compare to paper (arXiv-2505.14454)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "repro"

PAPER_OV = {
    "mvbench": 57.2,
    "longvideobench_val_v": 54.9,
    "mlvu_dev": 62.5,
    "videomme": 58.6,
    "egoschema": 59.7,
    "perceptiontest_val_mc": 56.7,
    "avg_pct_r25": 99.6,
}
PAPER_VID = {
    "mvbench": 57.0,
    "longvideobench_val_v": 55.5,
    "mlvu_dev": 59.0,
    "videomme": 61.7,
    "avg_pct_r25": 93.6,
}
PAPER_OV_R15_AVG = 95.1
PAPER_VID_R15_AVG = 88.5

TABLE1_TASKS = ["mvbench", "longvideobench_val_v", "mlvu_dev", "videomme"]


def latest_results(dirpath: Path) -> dict | None:
    if not dirpath.exists():
        return None
    cands = sorted(dirpath.rglob("*results*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if "samples" in p.name:
            continue
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    return None


def score_from_results(data: dict, task: str) -> float | None:
    if task in data.get("results", {}):
        r = data["results"][task]
        for key in r:
            if "acc" in key.lower() or "none" in key:
                try:
                    return float(r[key])
                except Exception:
                    pass
    # flat keys
    for k, v in data.items():
        if task in k and isinstance(v, (int, float)):
            return float(v)
    return None


def egoschema_server(log_dir: Path) -> float | None:
    p = log_dir / "validation_server_response.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    for k in ("subset_accuracy", "subset", "accuracy", "fullset_accuracy"):
        if k in d and d[k] is not None:
            return float(d[k])
    return None


def videomme_long(data: dict) -> float | None:
    r = data.get("results", {}).get("videomme", {})
    for k in r:
        if "long" in k.lower():
            return float(r[k])
    return None


def parse_efficiency_log(text: str) -> dict[str, float]:
    out = {}
    for key in ("LLM_time_s", "Total_time_s", "Peak_mem_MB"):
        m = re.search(rf"{key}[^\d]*([\d.]+)", text)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"mvbench[^\d]*([\d.]+)", text, re.I)
    if m:
        out["mvbench_acc"] = float(m.group(1))
    return out


def main() -> None:
    lines = [
        "# VidCom² Reproduction Report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "## Table 1 (LLaVA-OV-7B, R=25%)",
        "",
        "| Benchmark | Reproduced | Paper | Delta |",
        "|---|---|---|---|",
    ]

    ov_base = {}
    for task in TABLE1_TASKS:
        got = score_from_results(latest_results(LOG / "ov-7b" / f"{task}_baseline") or {}, task)
        if got is not None:
            ov_base[task] = got
        r25 = score_from_results(latest_results(LOG / "ov-7b" / f"{task}_0p25") or {}, task)
        paper = PAPER_OV.get(task, 0)
        if r25 is None:
            lines.append(f"| {task} | pending | {paper} | |")
        else:
            lines.append(f"| {task} | {r25:.2f} | {paper} | {r25 - paper:+.2f} |")

    if ov_base and all(
        score_from_results(latest_results(LOG / "ov-7b" / f"{t}_0p25") or {}, t) is not None
        for t in TABLE1_TASKS
    ):
        avg = sum(
            score_from_results(latest_results(LOG / "ov-7b" / f"{t}_0p25") or {}, t) / ov_base[t]
            for t in TABLE1_TASKS
        ) / len(TABLE1_TASKS) * 100
        lines.append(f"\n**OV Average % @ R=25%: {avg:.2f}** (paper {PAPER_OV['avg_pct_r25']})")

    lines += ["", "## Table 2 (OV R=25%)", ""]
    es = egoschema_server(LOG / "ov-7b" / "egoschema_0p25")
    pt = score_from_results(latest_results(LOG / "ov-7b" / "perceptiontest_val_mc_0p25") or {}, "perceptiontest_val_mc")
    lines.append(f"- EgoSchema (server): {es if es else 'pending'} (paper {PAPER_OV['egoschema']})")
    lines.append(f"- PerceptionTest: {pt if pt else 'pending'} (paper {PAPER_OV['perceptiontest_val_mc']})")

    lines += ["", "## LLaVA-Video @ R=25% Avg", ""]
    vid_base, vid_r25 = {}, {}
    for task in TABLE1_TASKS:
        b = score_from_results(latest_results(LOG / "vid-7b" / f"{task}_baseline") or {}, task)
        r = score_from_results(latest_results(LOG / "vid-7b" / f"{task}_0p25") or {}, task)
        if b is not None:
            vid_base[task] = b
        if r is not None:
            vid_r25[task] = r
    if vid_base and len(vid_r25) == len(TABLE1_TASKS):
        avg_v = sum(vid_r25[t] / vid_base[t] for t in TABLE1_TASKS) / len(TABLE1_TASKS) * 100
        lines.append(f"**Video Average % @ R=25%: {avg_v:.2f}** (paper {PAPER_VID['avg_pct_r25']})")

    qwen_ub = latest_results(ROOT / "logs" / "repro" / "qwen2vl" / "videomme_baseline")
    qwen_r25 = latest_results(ROOT / "logs" / "repro" / "qwen2vl" / "videomme_r25")
    if qwen_ub and qwen_r25:
        ub_l = videomme_long(qwen_ub)
        r_l = videomme_long(qwen_r25)
        if ub_l and r_l:
            lines += ["", "## Qwen2-VL VideoMME-Long", ""]
            lines.append(f"- Relative: {100 * r_l / ub_l:.1f}% (paper 101.2%)")

    report = "\n".join(lines) + "\n"
    out = ROOT / "REPRODUCTION_REPORT.md"
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
