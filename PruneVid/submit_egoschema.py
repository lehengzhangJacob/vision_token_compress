#!/usr/bin/env python3
"""Convert ST-LLM fullset predictions -> EgoSchema validation-server submission.

The repo's egoschema_fullset.json has a NON-official candidate ordering (and mostly
nulled labels), so the model's chosen option letter is meaningless for the server.
We remap: model letter -> chosen option TEXT (via fullset json candidates) ->
index into the OFFICIAL option ordering (lmms-lab/egoschema MC parquet).

Submission format (matches lmms_eval egoschema utils: video_idx -> int 0..4):
    { "<video_idx>": <official_answer_index>, ... }  # all 5031 entries

Usage:
    python submit_egoschema.py [--submit]
"""
import argparse, json, os, re, sys, glob
import pandas as pd

MC_PARQUET = glob.glob("/home/msj_team/.cache/huggingface/hub/datasets--lmms-lab--egoschema/snapshots/*/MC/test-00000-of-00001.parquet")[0]
FULLSET_JSON = "/home/msj_team/Jacob/nk/PruneVid/DATAS/ego_schema/json/egoschema_fullset.json"
DEF_RES = "/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-egoschema-fullset/stllm_prunevid_egoschema_fullset.json"
DEF_OUT = "/home/msj_team/Jacob/nk/PruneVid/test_results/stllm-prunevid-egoschema-fullset/egoschema_fullset_submission.json"
SERVER = "https://validation-server.onrender.com/api/upload/"


def clean(s):
    return re.sub(r'^[A-E][\.\)]\s*', '', str(s).strip()).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=DEF_RES)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    mc = pd.read_parquet(MC_PARQUET)
    full = json.load(open(FULLSET_JSON))
    full_by_qidx = {str(x.get("question_idx")): x for x in full}
    res = json.load(open(a.res)).get("res_list", [])

    # index predictions by question_idx (fallback: video uid)
    res_by_qidx, res_by_vid = {}, {}
    for o in res:
        if o.get("question_idx") is not None:
            res_by_qidx[str(o["question_idx"])] = o
        res_by_vid[str(o.get("video", "")).replace(".mp4", "")] = o

    sub = {}
    remapped = fallback = missing = 0
    for _, row in mc.iterrows():
        qidx = str(row["question_idx"]); vid = row["video_idx"]
        off = [clean(x) for x in row["option"]]
        o = res_by_qidx.get(qidx) or res_by_vid.get(vid)
        if not o:
            sub[vid] = 0; missing += 1; continue
        m = re.search(r'\(([A-E])\)', o.get("pred_norm", "") or "")
        f = full_by_qidx.get(qidx)
        chosen = None
        if m and f:
            idx = ord(m.group(1)) - 65
            if 0 <= idx < len(f["candidates"]):
                chosen = clean(f["candidates"][idx])
        if chosen is not None and chosen in off:
            sub[vid] = off.index(chosen); remapped += 1
        else:
            # fallback: try to match model's raw pred text against official options
            praw = clean(o.get("pred", ""))
            hit = next((i for i, t in enumerate(off) if t and t in praw), None)
            sub[vid] = hit if hit is not None else 0
            fallback += 1

    print(f"official rows: {len(mc)} | remapped: {remapped} | fallback: {fallback} | missing-pred: {missing}")
    print(f"unique video_idx in submission: {len(sub)} (expect 5031)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(sub, open(a.out, "w"))
    print(f"wrote submission -> {a.out}")

    if a.submit:
        try:
            import requests
            r = requests.post(SERVER, headers={"Content-Type": "application/json"}, json=sub, timeout=300)
            print("server status:", r.status_code)
            print("server response:", r.text[:2000])
        except Exception as e:
            print("submit failed:", repr(e))
            print(f"\nManual:\n  curl -X POST -H 'Content-Type: application/json' -d @{a.out} {SERVER}")
    else:
        print(f"\nTo submit:\n  curl -X POST -H 'Content-Type: application/json' -d @{a.out} {SERVER}\n  (or re-run with --submit)")


if __name__ == "__main__":
    sys.exit(main())
