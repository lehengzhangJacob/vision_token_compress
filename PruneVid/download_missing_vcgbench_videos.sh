#!/usr/bin/env bash
# Download the ~41 VCGBench videos missing from Test_Videos.zip (YouTube IDs in v_<id> names).
set -euo pipefail
cd "$(dirname "$0")"
source /home/msj_team/Jacob/0/env.sh
source prunevid_env.sh

BENCH=DATAS/VCGBench/Videos/Benchmarking
mkdir -p "$BENCH" logs

/home/msj_team/.conda/envs/PruneVid/bin/pip install -q yt-dlp

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

root = Path("DATAS/VCGBench")
bench = root / "Videos/Benchmarking"
need = set()
for j in ["generic_qa.json", "temporal_qa.json", "consistency_qa.json"]:
    for item in json.load(open(root / f"Zero_Shot_QA/Benchmarking_QA/{j}")):
        need.add(item["video_name"])
have = {p.stem for p in bench.glob("*.mp4") if p.stat().st_size > 1000}
missing = sorted(need - have)
print(f"missing {len(missing)} / {len(need)} unique videos", flush=True)
if not missing:
    raise SystemExit(0)

log = Path("logs/vcgbench_ytdlp.log")
yt_dlp = os.environ.get("YT_DLP_BIN", "yt-dlp")

proxies = []
for key in ("https_proxy", "http_proxy", "all_proxy"):
    value = os.environ.get(key)
    if value and value not in proxies:
        proxies.append(value)

for value in list(proxies):
    if value.startswith("socks5://"):
        socks5h = "socks5h://" + value[len("socks5://") :]
        if socks5h not in proxies:
            proxies.append(socks5h)

clients = ["android", "ios", "web"]

with log.open("a") as lf:
    for name in missing:
        vid = name[2:] if name.startswith("v_") else name
        out = bench / f"{name}.mp4"
        if out.exists() and out.stat().st_size > 1000:
            continue
        if out.exists():
            out.unlink()
        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"==== download {name} ====", flush=True)
        ok = False
        for proxy in proxies:
            for client in clients:
                if out.exists() and out.stat().st_size > 1000:
                    ok = True
                    break
                if out.exists():
                    out.unlink()
                print(f"==== try {name}: proxy={proxy} client={client} ====", flush=True)
                print(f"\n==== try {name}: proxy={proxy} client={client} ====", file=lf, flush=True)
                cmd = [
                    yt_dlp,
                    "--proxy",
                    proxy,
                    "--force-ipv4",
                    "--socket-timeout",
                    "30",
                    "--extractor-args",
                    f"youtube:player_client={client}",
                    "-f",
                    "best[ext=mp4]/best",
                    "-o",
                    str(out),
                    "--no-playlist",
                    "--retries",
                    "3",
                    "--fragment-retries",
                    "3",
                    url,
                ]
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if r.returncode == 0 and out.exists() and out.stat().st_size > 1000:
                    print(f"==== OK {name}: proxy={proxy} client={client} ====", flush=True)
                    ok = True
                    break
            if ok:
                break
        if not ok:
            if out.exists() and out.stat().st_size < 1000:
                out.unlink()
            print(f"!!!! FAILED {name}", flush=True)
PY

n=$(/home/msj_team/.conda/envs/PruneVid/bin/python - <<'PY'
from pathlib import Path
print(sum(1 for p in Path("DATAS/VCGBench/Videos/Benchmarking").glob("*.mp4") if p.stat().st_size > 1000))
PY
)
echo "===== Benchmarking valid mp4 count: $n ====="
