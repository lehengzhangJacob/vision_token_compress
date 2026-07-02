#!/usr/bin/env python3
"""Convert ST-LLM fullset predictions -> EgoSchema validation-server submission.

Submission format (matches lmms_eval egoschema utils: doc["video_idx"] -> int):
    { "<video-uid-without-.mp4>": <answer_index 0..4>, ... }  # all 5031 entries

Usage:
    python submit_egoschema.py [--res <merged.json>] [--out <submission.json>] [--submit]
Without --submit it only writes the JSON and prints the manual curl command.
"""
import argparse, json, os, re, sys

DEF_RES = "/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-egoschema-fullset/stllm_prunevid_egoschema_fullset.json"
DEF_OUT = "/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-egoschema-fullset/egoschema_fullset_submission.json"
SERVER = "https://validation-server.onrender.com/api/upload"


def to_int(pred_norm, pred):
    for s in (pred_norm or "", pred or ""):
        m = re.search(r"\(?\b([A-Ea-e])\b\)?", s)
        if m:
            return ord(m.group(1).upper()) - ord("A")
    return 0  # safe fallback (server needs 0..4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=DEF_RES)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    data = json.load(open(a.res))
    res = data.get("res_list", [])
    sub = {}
    bad = 0
    for o in res:
        uid = os.path.splitext(str(o.get("video", "")))[0]
        if not uid:
            continue
        v = to_int(o.get("pred_norm"), o.get("pred"))
        if not (0 <= v <= 4):
            v = 0; bad += 1
        sub[uid] = v

    print(f"res_list items: {len(res)} | unique uids: {len(sub)} | fallback/bad preds: {bad}")
    if len(sub) != 5031:
        print(f"WARNING: expected 5031 unique uids, got {len(sub)} "
              f"({'predictions incomplete' if len(sub) < 5031 else 'duplicates?'})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(sub, open(a.out, "w"))
    print(f"wrote submission -> {a.out}")

    # local sanity accuracy on the 500 labeled subset items (should be ~59-60%)
    labeled = [o for o in res if o.get("correct") is not None]
    if labeled:
        acc = 100 * sum(1 for o in labeled if o["correct"]) / len(labeled)
        print(f"[sanity] labeled subset acc ({len(labeled)} items): {acc:.2f}%")

    if a.submit:
        try:
            import requests
            with open(a.out, "rb") as f:
                r = requests.post(SERVER, files={"file": ("submission.json", f, "application/json")}, timeout=120)
            print("server status:", r.status_code)
            print("server response:", r.text[:2000])
        except Exception as e:
            print("submit failed:", repr(e))
            print(f"\nManual submission:\n  curl -X POST -F 'file=@{a.out}' {SERVER}")
    else:
        print(f"\nTo submit:\n  curl -X POST -F 'file=@{a.out}' {SERVER}\n  (or re-run with --submit)")


if __name__ == "__main__":
    sys.exit(main())
