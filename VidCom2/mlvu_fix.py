#!/usr/bin/env python3
"""Validated re-fetch of corrupt MLVU videos.

The earlier byte-range refetch (mlvu_refetch.py) left 136 videos corrupt (moov atom
not found) because it never validated the extracted MP4. This version processes each
corrupt video individually: re-download its zip byte-range, extract, VALIDATE with
ffprobe, and retry (alternating proxy/mirror) until valid or exhausted.
Reads corrupt list from logs/mlvu_corrupt_list.txt (ffprobe scan output).
"""
import glob, os, sys, time, subprocess, zipfile
import requests

SNAP = glob.glob("/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/*/")[0]
MLVU = "/home/msj_team/.cache/huggingface/mlvu"
REPO = "sy1998/MLVU_dev"
CORRUPT_LIST = "/home/msj_team/Jacob/nk/VidCom2/logs/mlvu_corrupt_list.txt"
PROXY = "http://127.0.0.1:7890"
CHUNK = 16 * 1024 * 1024
MAXRETRY = 8


def url_for(route, name):
    host = "https://hf-mirror.com" if route == "mirror" else "https://huggingface.co"
    return f"{host}/datasets/{REPO}/resolve/main/{name}"


def valid_mp4(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=40)
        return r.returncode == 0 and "video" in (r.stdout or "")
    except Exception:
        return False


def build_plan():
    """corrupt basename -> (zipname, header_off, end_off)"""
    want = set(l.strip() for l in open(CORRUPT_LIST) if l.strip())
    plan = {}
    for z in sorted(glob.glob(SNAP + "/video_part_*.zip")):
        name = os.path.basename(z)
        try:
            zf = zipfile.ZipFile(z)
        except Exception:
            continue
        infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
        offs = [i.header_offset for i in infos] + [os.path.getsize(z)]
        for idx, i in enumerate(infos):
            b = os.path.basename(i.filename)
            if b in want:
                plan[b] = (name, i.header_offset, offs[idx + 1])
        zf.close()
    return plan


def fetch_range(route, name, start, end, fd):
    sess = requests.Session()
    sess.proxies = {"http": PROXY, "https": PROXY} if route == "proxy" else {}
    off = start
    while off < end:
        ln = min(CHUNK, end - off)
        r = sess.get(url_for(route, name), headers={"Range": f"bytes={off}-{off+ln-1}"},
                     timeout=120, allow_redirects=True)
        if r.status_code not in (206, 200):
            raise IOError(f"http {r.status_code}")
        buf = r.content
        if len(buf) != ln:
            raise IOError(f"got {len(buf)} want {ln}")
        os.pwrite(fd, buf, off)
        off += ln


def extract_one(zippath, basename, dest):
    zf = zipfile.ZipFile(zippath)
    tgt = next((i for i in zf.infolist() if os.path.basename(i.filename) == basename), None)
    if tgt is None:
        zf.close(); raise IOError("not in zip")
    tmp = dest + ".part"
    import shutil
    with zf.open(tgt) as s, open(tmp, "wb") as d:
        shutil.copyfileobj(s, d, 1 << 20)
    zf.close()
    os.replace(tmp, dest)


def main():
    log = lambda m: print(f"[{time.strftime('%F %T')}] {m}", flush=True)
    if not os.path.exists(CORRUPT_LIST):
        log(f"no corrupt list at {CORRUPT_LIST}"); return 1
    plan = build_plan()
    log(f"corrupt videos to fix: {len(plan)}")
    fixed = failed = 0
    fd_cache = {}
    def get_fd(zn):
        if zn not in fd_cache:
            fd_cache[zn] = os.open(os.path.realpath(os.path.join(SNAP, zn)), os.O_RDWR)
        return fd_cache[zn]

    for k, (b, (zn, s, e)) in enumerate(sorted(plan.items()), 1):
        dest = os.path.join(MLVU, b)
        if os.path.exists(dest) and valid_mp4(dest):
            fixed += 1; continue
        if os.path.exists(dest):
            os.remove(dest)
        ok = False
        for a in range(MAXRETRY):
            route = "proxy" if a % 2 == 0 else "mirror"
            try:
                fetch_range(route, zn, s, e, get_fd(zn))
                extract_one(os.path.join(SNAP, zn), b, dest)
                if valid_mp4(dest):
                    ok = True; break
                else:
                    os.path.exists(dest) and os.remove(dest)
            except Exception as ex:
                if a == MAXRETRY - 1:
                    log(f"  {b}: give up ({repr(ex)[:60]})")
                time.sleep(min(2 ** a, 15))
        if ok:
            fixed += 1
        else:
            failed += 1
        if k % 10 == 0 or k == len(plan):
            log(f"progress {k}/{len(plan)} fixed={fixed} failed={failed} ({(e-s)/1e6:.0f}MB last)")
    for fd in fd_cache.values():
        os.close(fd)
    onchk = len(glob.glob(MLVU + "/*.mp4"))
    log(f"DONE fixed={fixed} failed={failed} | videos on disk={onchk}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
