#!/usr/bin/env python3
"""Parallel validated re-fetch of corrupt MLVU videos (speedup of mlvu_fix.py).

Same logic (re-download each corrupt video's zip byte-range, extract, ffprobe-validate,
retry) but processes many videos concurrently so we saturate the ~3.8 MB/s aggregate
proxy/mirror bandwidth instead of ~1 MB/s on a single connection.
Resumable: any video that already passes ffprobe is skipped.
Different videos live at non-overlapping zip offsets, so concurrent os.pwrite to the
same zip fd is safe; extraction opens its own read handle.
"""
import glob, os, sys, time, subprocess, zipfile, shutil, threading, queue
import requests

SNAP = glob.glob("/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/*/")[0]
MLVU = "/home/msj_team/.cache/huggingface/mlvu"
REPO = "sy1998/MLVU_dev"
CORRUPT_LIST = "/home/msj_team/Jacob/nk/VidCom2/logs/mlvu_corrupt_list.txt"
PROXY = "http://127.0.0.1:7890"
CHUNK = 16 * 1024 * 1024
MAXRETRY = 8
NWORKERS = 12


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


# one shared RDWR fd per zip (pwrite is offset-addressed and thread-safe)
FDS = {}
FDS_LOCK = threading.Lock()
def get_fd(zn):
    with FDS_LOCK:
        if zn not in FDS:
            FDS[zn] = os.open(os.path.realpath(os.path.join(SNAP, zn)), os.O_RDWR)
        return FDS[zn]


def fetch_range(sess, name, start, end, fd):
    off = start
    while off < end:
        ln = min(CHUNK, end - off)
        r = sess.get(url_for("proxy" if sess._route == "proxy" else "mirror", name),
                     headers={"Range": f"bytes={off}-{off+ln-1}"}, timeout=120, allow_redirects=True)
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
    tmp = dest + f".part{threading.get_ident()}"
    with zf.open(tgt) as s, open(tmp, "wb") as d:
        shutil.copyfileobj(s, d, 1 << 20)
    zf.close()
    os.replace(tmp, dest)


done = {"fixed": 0, "failed": 0}
DLOCK = threading.Lock()


def worker(wid, q, log):
    sess = requests.Session()
    sess._route = "proxy" if wid % 3 != 0 else "mirror"   # ~2/3 proxy, 1/3 mirror
    sess.proxies = {"http": PROXY, "https": PROXY} if sess._route == "proxy" else {}
    while True:
        try:
            b, zn, s, e = q.get_nowait()
        except queue.Empty:
            return
        dest = os.path.join(MLVU, b)
        if valid_mp4(dest):
            with DLOCK: done["fixed"] += 1
            q.task_done(); continue
        if os.path.exists(dest):
            try: os.remove(dest)
            except: pass
        ok = False
        for a in range(MAXRETRY):
            # alternate route on retry
            sess._route = "mirror" if (a % 2 == 1) else ("proxy" if wid % 3 != 0 else "mirror")
            sess.proxies = {"http": PROXY, "https": PROXY} if sess._route == "proxy" else {}
            try:
                fetch_range(sess, zn, s, e, get_fd(zn))
                extract_one(os.path.join(SNAP, zn), b, dest)
                if valid_mp4(dest):
                    ok = True; break
                elif os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                time.sleep(min(2 ** a, 15))
        with DLOCK:
            if ok: done["fixed"] += 1
            else:  done["failed"] += 1
        q.task_done()


def main():
    log = lambda m: print(f"[{time.strftime('%F %T')}] {m}", flush=True)
    plan = build_plan()
    # skip already-valid up front for accurate progress
    todo = [(b, zn, s, e) for b, (zn, s, e) in sorted(plan.items()) if not valid_mp4(os.path.join(MLVU, b))]
    already = len(plan) - len(todo)
    log(f"corrupt total={len(plan)} already-valid={already} to-fetch={len(todo)} workers={NWORKERS}")
    q = queue.Queue()
    for t in todo: q.put(t)
    done["fixed"] = already
    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i, q, log), daemon=True) for i in range(NWORKERS)]
    for t in ts: t.start()
    while any(t.is_alive() for t in ts):
        time.sleep(30)
        f, x = done["fixed"], done["failed"]
        log(f"progress fixed={f}/{len(plan)} failed={x} ({time.time()-t0:.0f}s)")
    for t in ts: t.join()
    for fd in FDS.values(): os.close(fd)
    onchk = len(glob.glob(MLVU + "/*.mp4"))
    log(f"DONE fixed={done['fixed']} failed={done['failed']} | videos on disk={onchk}")
    return 0 if done["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
