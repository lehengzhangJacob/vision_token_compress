#!/usr/bin/env python3
"""Build AOT reproduction report comparing reproduced results to paper/committed references.
Usage: python make_report.py  -> prints tables and writes REPRODUCTION_REPORT.md
"""
import json, glob, os, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(ROOT, "logs/repro")
COMMIT_OV = os.path.join(ROOT, "logs/ov-7b-OTA-Sinknorn/260323/lmms-lab__llava-onevision-qwen2-7b-ov")

REF_OV = {
    "mvbench":              {10: 57.0, 15: 57.8, 20: 58.1, 25: 58.7},
    "egoschema":            {10: 60.6, 15: 61.3, 20: 61.3, 25: 61.3},
    "longvideobench_val_v": {10: 54.2, 15: 55.2, 20: 56.2, 25: 56.3},
    "videomme":             {10: 56.1, 15: 56.6, 20: 57.2, 25: 57.5},
}
REF_VID = {
    "mvbench":              {15: 57.8, 25: 58.8},
    "egoschema":            {15: 55.2, 25: 55.4},
    "longvideobench_val_v": {15: 55.0, 25: 56.2},
    "videomme":             {15: 62.0, 25: 62.4},
}
PRETTY = {"mvbench": "MVBench", "egoschema": "EgoSchema",
          "longvideobench_val_v": "LongVideoBench", "videomme": "VideoMME"}

def latest(d, pat):
    fs = sorted(glob.glob(os.path.join(d, "**", pat), recursive=True))
    return fs[-1] if fs else None

def score_of(task, ratio, model):
    d = os.path.join(REPRO, f"{model}-7b", f"{task}_{ratio}")
    if not os.path.isdir(d):
        return None
    if task == "egoschema":
        # compare submission to committed (OV only); report agreement %
        sub = latest(d, "inference_results_egoschema_*.json") or latest(d, "*submission*.json")
        if not sub:
            return None
        mine = json.load(open(sub))
        ref_path = os.path.join(COMMIT_OV, f"egoschema_{ratio}.json")
        if model == "ov" and os.path.exists(ref_path):
            ref = json.load(open(ref_path))
            common = set(mine) & set(ref)
            if not common:
                return None
            agree = sum(1 for k in common if str(mine[k]) == str(ref[k])) / len(common) * 100
            return ("agree", agree, len(common))
        return ("nsub", len(mine), 0)
    rj = latest(d, "*_results.json")
    if not rj:
        return None
    res = json.load(open(rj)).get("results", {})
    if task == "mvbench":
        accs = [v["mvbench_accuracy,none"] for k, v in res.items()
                if k != "mvbench" and "mvbench_accuracy,none" in v]
        return sum(accs) / len(accs) if accs else None
    if task == "longvideobench_val_v":
        r = res.get("longvideobench_val_v", {})
        return r["lvb_acc,none"] * 100 if "lvb_acc,none" in r else None
    if task == "videomme":
        r = res.get("videomme", {})
        return r.get("videomme_perception_score,none")
    return None

def build(model, REF):
    lines = [f"### LLaVA-{'OneVision' if model=='ov' else 'Video'}-7B "
             f"({'Table 1' if model=='ov' else 'Table 2'})", "",
             "| Benchmark | Ratio | Reproduced | Paper | Delta |",
             "|---|---|---|---|---|"]
    for task in ["mvbench", "egoschema", "longvideobench_val_v", "videomme"]:
        for ratio in sorted(REF.get(task, {}).keys()):
            ref = REF[task][ratio]
            sc = score_of(task, ratio, model)
            if sc is None:
                cell, delta = "pending", ""
            elif isinstance(sc, tuple) and sc[0] == "agree":
                cell, delta = f"{sc[1]:.1f}% pred-agree (n={sc[2]})", "(submission)"
            elif isinstance(sc, tuple):
                cell, delta = f"submission n={sc[1]}", "(no GT)"
            else:
                cell, delta = f"{sc:.2f}", f"{sc-ref:+.2f}"
            lines.append(f"| {PRETTY[task]} | {ratio}% | {cell} | {ref:.1f} | {delta} |")
    return "\n".join(lines)

def main():
    out = [f"# AOT (CVPR 2026) Reproduction Report",
           f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M}_  ",
           "Model: LLaVA-OneVision-7B / LLaVA-Video-7B - 2x RTX 4090, hf-mirror, training-free AOT.",
           "EgoSchema (full test set) has no public ground truth; validated by prediction-agreement "
           "with the authors' committed submission files (100% agreement => identical server score).",
           "", build("ov", REF_OV), "", build("vid", REF_VID), ""]
    rep = "\n".join(out)
    open(os.path.join(ROOT, "REPRODUCTION_REPORT.md"), "w").write(rep)
    print(rep)

if __name__ == "__main__":
    main()
