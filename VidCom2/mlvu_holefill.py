#!/usr/bin/env python3
"""Fill the zero-holes in the truncated MLVU zips via HTTP Range requests.

The zips (video_part_2..8) were downloaded with gaps: they are sparse files whose
logical size is correct but ~70GB of interior bytes are missing (zero holes). This
enumerates the holes (SEEK_HOLE/SEEK_DATA) and re-fetches ONLY those byte ranges,
using proxy + mirror routes in parallel. Idempotent/resumable: re-running only
fetches whatever is still a hole.
"""
import os, sys, time, threading, queue
import requests

SNAP = "/home/msj_team/.cache/huggingface/hub/datasets--sy1998--MLVU_dev/snapshots/96207eb9aa7101e2a495dd147684a7e618c79e12"
REPO = "sy1998/MLVU_dev"
PARTS = [2, 3, 4, 5, 6, 7, 8]          # part 1 is intact
CHUNK = 32 * 1024 * 1024               # 32 MiB
PROXY = "http://127.0.0.1:7890"
N_PROXY, N_MIRROR = 12, 8              # worker counts per route
MAXRETRY = 6

def url_for(route, name):
    if route == "mirror":
        return f"https://hf-mirror.com/datasets/{REPO}/resolve/main/{name}"
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}"

def enum_holes(fd, size):
    """Yield (start, length) of every zero-hole in the file."""
    pos = 0
    while pos < size:
        try:
            data = os.lseek(fd, pos, os.SEEK_DATA)
        except OSError:
            yield (pos, size - pos); return          # rest is hole
        if data > pos:
            yield (pos, data - pos)                    # [pos,data) hole
        try:
            hole = os.lseek(fd, data, os.SEEK_HOLE)
        except OSError:
            return
        pos = hole

def build_jobs():
    jobs = []          # (part, offset, length)
    fds = {}
    total_hole = 0
    for p in PARTS:
        blob = os.path.realpath(os.path.join(SNAP, f"video_part_{p}.zip"))
        fd = os.open(blob, os.O_RDWR)
        fds[p] = (fd, blob, os.fstat(fd).st_size)
        for (start, length) in enum_holes(fd, fds[p][2]):
            total_hole += length
            off = start
            while off < start + length:
                ln = min(CHUNK, start + length - off)
                jobs.append((p, off, ln)); off += ln
    return jobs, fds, total_hole

done_bytes = 0
lock = threading.Lock()

def worker(route, q, fds, log):
    global done_bytes
    sess = requests.Session()
    sess.proxies = {"http": PROXY, "https": PROXY} if route == "proxy" else {}
    trust = False if route == "mirror" else True
    while True:
        try:
            part, off, ln = q.get_nowait()
        except queue.Empty:
            return
        fd = fds[part][0]; name = f"video_part_{part}.zip"
        ok = False
        for attempt in range(MAXRETRY):
            try:
                h = {"Range": f"bytes={off}-{off+ln-1}"}
                r = sess.get(url_for(route, name), headers=h, timeout=90,
                             allow_redirects=True, stream=True)
                if r.status_code not in (206, 200):
                    raise IOError(f"http {r.status_code}")
                buf = r.content
                if len(buf) != ln:
                    raise IOError(f"got {len(buf)} want {ln}")
                os.pwrite(fd, buf, off)
                ok = True; break
            except Exception as e:
                time.sleep(min(2**attempt, 20))
        if ok:
            with lock:
                done_bytes += ln
        else:
            q.put((part, off, ln))          # requeue for another worker/route
            time.sleep(1)
        q.task_done()

def main():
    log = lambda m: print(f"[{time.strftime('%F %T')}] {m}", flush=True)
    log("enumerating holes...")
    jobs, fds, total = build_jobs()
    log(f"holes: {total/1e9:.2f} GB in {len(jobs)} chunks across parts {PARTS}")
    if not jobs:
        log("no holes -> nothing to do"); return 0
    q = queue.Queue()
    for j in jobs: q.put(j)
    t0 = time.time()
    threads = []
    for i in range(N_PROXY):  threads.append(threading.Thread(target=worker, args=("proxy", q, fds, log), daemon=True))
    for i in range(N_MIRROR): threads.append(threading.Thread(target=worker, args=("mirror", q, fds, log), daemon=True))
    for t in threads: t.start()
    # progress
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        el = time.time() - t0; sp = done_bytes/1e6/max(el,1)
        rem = (total - done_bytes)/1e6/max(sp,0.1)
        log(f"progress {done_bytes/1e9:.2f}/{total/1e9:.2f} GB ({100*done_bytes/total:.1f}%) "
            f"{sp:.1f} MB/s ETA {rem/60:.0f}min qsize~{q.qsize()}")
    for t in threads: t.join()
    # verify holes gone
    left = 0
    for p in PARTS:
        fd = fds[p][0]
        for (_, ln) in enum_holes(fd, fds[p][2]): left += ln
    log(f"DONE fill in {(time.time()-t0)/60:.0f}min; residual holes = {left/1e6:.1f} MB")
    return 0 if left == 0 else 2

if __name__ == "__main__":
    sys.exit(main())
