#!/usr/bin/env python3
"""Targeted re-fetch of the 218 MLVU videos that hole-filling couldn't recover.

The zips have correct central-directory offsets (56 videos extracted fine after the
hole-fill) but the DATA at some entries' offsets is garbage (not just zero-holes), so
those files fail with 'Bad magic number'. Fix: for each still-missing video, re-fetch
its byte-range [header_offset, next_entry_offset) from the remote and overwrite the
local blob at that offset, then extract. ~47GB total (vs 248GB full re-download).
Resumable: recomputes the missing set each run.
"""
import glob, os, time, threading, queue, zipfile, shutil
import pandas as pd, requests

SNAP = glob.glob("/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/*/")[0]
MLVU = "/home/msj_team/.cache/huggingface/mlvu"
REPO = "sy1998/MLVU_dev"
CHUNK = 32 * 1024 * 1024
PROXY = "http://127.0.0.1:7890"
N_PROXY, N_MIRROR = 12, 8
MAXRETRY = 6

def url_for(route, name):
    host = "https://hf-mirror.com" if route == "mirror" else "https://huggingface.co"
    return f"{host}/datasets/{REPO}/resolve/main/{name}"

need = set(pd.read_parquet("/home/msj_team/.cache/huggingface/mlvu_meta/test-00000-of-00001.parquet")['video_name'].unique())

def build_jobs():
    jobs = []       # (partname, blob_fd, dl_off, dl_len)
    fds = {}
    total = 0
    plan = {}       # video -> (partname, header_off, end_off)
    for z in sorted(glob.glob(SNAP + "/video_part_*.zip")):
        name = os.path.basename(z)
        zf = zipfile.ZipFile(z)
        infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
        offs = [i.header_offset for i in infos] + [os.path.getsize(z)]
        fd = os.open(os.path.realpath(z), os.O_RDWR); fds[name] = fd
        for idx, i in enumerate(infos):
            base = os.path.basename(i.filename)
            if base in need and not os.path.exists(os.path.join(MLVU, base)):
                start, end = i.header_offset, offs[idx + 1]
                plan[base] = (name, start, end)
                off = start
                while off < end:
                    ln = min(CHUNK, end - off)
                    jobs.append((name, fd, off, ln)); off += ln
                    total += ln
        zf.close()
    return jobs, fds, total, plan

done = 0; lock = threading.Lock()

def worker(route, q, log):
    global done
    sess = requests.Session()
    sess.proxies = {"http": PROXY, "https": PROXY} if route == "proxy" else {}
    while True:
        try: name, fd, off, ln = q.get_nowait()
        except queue.Empty: return
        ok = False
        for a in range(MAXRETRY):
            try:
                r = sess.get(url_for(route, name), headers={"Range": f"bytes={off}-{off+ln-1}"},
                             timeout=90, allow_redirects=True)
                if r.status_code not in (206, 200): raise IOError(f"http {r.status_code}")
                buf = r.content
                if len(buf) != ln: raise IOError(f"got {len(buf)} want {ln}")
                os.pwrite(fd, buf, off); ok = True; break
            except Exception:
                time.sleep(min(2**a, 20))
        if ok:
            with lock: done += ln
        else:
            q.put((name, fd, off, ln)); time.sleep(1)
        q.task_done()

def main():
    log = lambda m: print(f"[{time.strftime('%F %T')}] {m}", flush=True)
    jobs, fds, total, plan = build_jobs()
    log(f"refetch {len(plan)} videos, {total/1e9:.2f} GB in {len(jobs)} chunks")
    if jobs:
        q = queue.Queue()
        for j in jobs: q.put(j)
        t0 = time.time()
        ts = [threading.Thread(target=worker, args=("proxy", q, log), daemon=True) for _ in range(N_PROXY)]
        ts += [threading.Thread(target=worker, args=("mirror", q, log), daemon=True) for _ in range(N_MIRROR)]
        for t in ts: t.start()
        while any(t.is_alive() for t in ts):
            time.sleep(30); el = time.time()-t0; sp = done/1e6/max(el,1)
            log(f"progress {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done/total:.1f}%) {sp:.1f} MB/s "
                f"ETA {(total-done)/1e6/max(sp,0.1)/60:.0f}min q~{q.qsize()}")
        for t in ts: t.join()
        log(f"download done in {(time.time()-t0)/60:.0f}min")
    # extract now that byte-ranges are correct
    rec = fail = 0
    for z in sorted(glob.glob(SNAP + "/video_part_*.zip")):
        try: zf = zipfile.ZipFile(z)
        except Exception as e: log(f"UNREADABLE {z}: {e}"); continue
        for i in zf.infolist():
            base = os.path.basename(i.filename)
            if not base.endswith('.mp4') or base not in need: continue
            tp = os.path.join(MLVU, base)
            if os.path.exists(tp): continue
            tmp = tp + ".part"
            try:
                with zf.open(i) as s, open(tmp, 'wb') as d: shutil.copyfileobj(s, d, 1 << 20)
                os.replace(tmp, tp); rec += 1
            except Exception as e:
                fail += 1
                if os.path.exists(tmp): os.remove(tmp)
                if fail <= 10: log(f"FAIL {base}: {repr(e)[:70]}")
        zf.close()
    onchk = len(glob.glob(MLVU + "/*.mp4"))
    log(f"extracted: recovered={rec} failed={fail} | on disk={onchk} (need 1122)")
    return 0 if onchk >= 1122 else 2

if __name__ == "__main__":
    import sys; sys.exit(main())
