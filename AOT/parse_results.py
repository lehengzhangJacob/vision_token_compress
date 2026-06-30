#!/usr/bin/env python3
"""Parse AOT reproduction results and compare to paper / committed references."""
import json, glob, os, sys

REPRO = os.path.join(os.path.dirname(__file__), "logs/repro")

# Paper Table 1 (LLaVA-OneVision-7B) AOT rows = committed reference numbers
REF_OV = {
    "mvbench":              {10: 57.0, 15: 57.8, 20: 58.1, 25: 58.7},
    "egoschema":            {10: 60.6, 15: 61.3, 20: 61.3, 25: 61.3},
    "longvideobench_val_v": {10: 54.2, 15: 55.2, 20: 56.2, 25: 56.3},
    "videomme":             {10: 56.1, 15: 56.6, 20: 57.2, 25: 57.5},
}
# Paper Table 2 (LLaVA-Video-7B) AOT rows
REF_VID = {
    "mvbench":              {15: 57.8, 25: 58.8},
    "egoschema":            {15: 55.2, 25: 55.4},
    "longvideobench_val_v": {15: 55.0, 25: 56.2},
    "videomme":             {15: 62.0, 25: 62.4},
}

def score_from_results(results, task):
    if task == "mvbench":
        accs = [v["mvbench_accuracy,none"] for k, v in results.items()
                if k != "mvbench" and "mvbench_accuracy,none" in v]
        return sum(accs) / len(accs) if accs else None
    if task == "longvideobench_val_v":
        r = results.get("longvideobench_val_v", {})
        return r.get("lvb_acc,none", None) * 100 if "lvb_acc,none" in r else None
    if task == "videomme":
        r = results.get("videomme", {})
        return r.get("videomme_perception_score,none", None)
    if task.startswith("egoschema"):
        r = results.get(task, results.get("egoschema", {}))
        for key in ("egoschema_accuracy,none", "accuracy,none", "submission,none"):
            if key in r and isinstance(r[key], (int, float)):
                return r[key] * (100 if r[key] <= 1 else 1)
        return None
    return None

def latest_results_json(d):
    files = sorted(glob.glob(os.path.join(d, "**", "*_results.json"), recursive=True))
    return files[-1] if files else None

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ov"
    base = os.path.join(REPRO, f"{model}-7b")
    REF = REF_OV if model == "ov" else REF_VID
    print(f"\n{'task':<24}{'ratio':>6}{'repro':>9}{'paper':>9}{'delta':>9}")
    print("-" * 57)
    for task in ["mvbench", "egoschema", "longvideobench_val_v", "videomme"]:
        for ratio in sorted(REF.get(task, {}).keys()):
            d = os.path.join(base, f"{task}_{ratio}")
            rj = latest_results_json(d) if os.path.isdir(d) else None
            ref = REF[task][ratio]
            if rj:
                results = json.load(open(rj)).get("results", {})
                sc = score_from_results(results, task)
                if sc is not None:
                    print(f"{task:<24}{ratio:>5}%{sc:>9.2f}{ref:>9.1f}{sc-ref:>+9.2f}")
                else:
                    print(f"{task:<24}{ratio:>5}%{'(no score)':>9}{ref:>9.1f}{'':>9}")
            else:
                print(f"{task:<24}{ratio:>5}%{'(pending)':>9}{ref:>9.1f}{'':>9}")
    print()

if __name__ == "__main__":
    main()
