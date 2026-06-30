#!/usr/bin/env python3
"""Download large xethub LFS files from Hugging Face via curl resume."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata

# Fallback when HEAD metadata omits Content-Length (xethub quirk).
KNOWN_SIZES: dict[tuple[str, str], int] = {
    ("sy1998/MLVU_dev", "video_part_1.zip"): 36626152774,
    ("sy1998/MLVU_dev", "video_part_2.zip"): 34311867313,
    ("sy1998/MLVU_dev", "video_part_3.zip"): 41427664931,
    ("sy1998/MLVU_dev", "video_part_4.zip"): 43162717558,
    ("sy1998/MLVU_dev", "video_part_5.zip"): 44702247576,
    ("sy1998/MLVU_dev", "video_part_6.zip"): 29624788310,
    ("sy1998/MLVU_dev", "video_part_7.zip"): 33061239060,
    ("sy1998/MLVU_dev", "video_part_8.zip"): 22147421187,
    ("lmms-lab/PerceptionTest_Val", "valid_audios.zip"): 35506467738,
    ("lmms-lab/PerceptionTest_Val", "videos_chunked_01.zip"): 43064882555,
    ("lmms-lab/PerceptionTest_Val", "videos_chunked_02.zip"): 32310330255,
}


def _repo_cache_slug(repo_id: str) -> str:
    return "datasets--" + repo_id.replace("/", "--")


def _list_incomplete_blobs(repo_id: str) -> list[Path]:
    hub = Path(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")))
    blobs = hub / _repo_cache_slug(repo_id) / "blobs"
    if not blobs.is_dir():
        return []
    return sorted(blobs.glob("*.incomplete"), key=lambda p: p.stat().st_size, reverse=True)


def _lookup_size(repo_id: str, filename: str) -> int | None:
    key = (repo_id, filename)
    if key in KNOWN_SIZES:
        return KNOWN_SIZES[key]
    for entry in HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        if entry.path == filename and getattr(entry, "size", None):
            return int(entry.size)
    return None


def _resolve_metadata(repo_id: str, filename: str, repo_type: str, attempts: int = 8):
    saved_endpoint = os.environ.get("HF_ENDPOINT")
    last_err: Exception | None = None

    try:
        for i in range(1, attempts + 1):
            for endpoint in (None, "https://hf-mirror.com"):
                if endpoint is None:
                    os.environ.pop("HF_ENDPOINT", None)
                    source = "huggingface.co"
                else:
                    os.environ["HF_ENDPOINT"] = endpoint
                    source = endpoint
                url = hf_hub_url(repo_id, filename, repo_type=repo_type)
                try:
                    meta = get_hf_file_metadata(url)
                    size = meta.size or _lookup_size(repo_id, filename)
                    if meta.location and size:
                        return meta.location, int(size)
                    last_err = RuntimeError(
                        f"incomplete metadata for {filename}: location={bool(meta.location)} size={size}"
                    )
                except Exception as e:
                    last_err = e
                print(f"==== METADATA retry {i}/{attempts} via {source} for {filename}: {last_err} ====", flush=True)
            if i < attempts:
                wait = min(30 * i, 120)
                time.sleep(wait)
    finally:
        if saved_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = saved_endpoint

    size = _lookup_size(repo_id, filename)
    if size:
        raise RuntimeError(
            f"Could not resolve download URL for {repo_id}/{filename} (expected {size} bytes): {last_err}"
        )
    raise RuntimeError(f"Could not resolve metadata for {repo_id}/{filename}: {last_err}")


def adopt_hf_incomplete(repo_id: str, dest: Path, expected_size: int) -> bool:
    """Copy the best matching HF .incomplete blob into dest for curl resume."""
    if dest.exists() and dest.stat().st_size >= expected_size:
        return False
    if dest.exists() and dest.stat().st_size > 0:
        return False

    best: Path | None = None
    best_size = 0
    for blob in _list_incomplete_blobs(repo_id):
        size = blob.stat().st_size
        if size <= 0 or size >= expected_size:
            continue
        if size > best_size:
            best, best_size = blob, size

    if best is None:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"==== ADOPT incomplete {best.name} ({best_size} bytes) -> {dest} ====", flush=True)
    shutil.copy2(best, dest)
    return True


def curl_download(url: str, dest: Path, attempts: int = 8) -> None:
    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy") or ""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for i in range(1, attempts + 1):
        cmd = [
            "curl",
            "--http1.1",
            "-f",
            "-L",
            "-C",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "10",
            "-o",
            str(dest),
            "--connect-timeout",
            "60",
            "--speed-time",
            "300",
            "--speed-limit",
            "1024",
        ]
        if proxy:
            cmd.extend(["--proxy", proxy])
        cmd.append(url)

        have = dest.stat().st_size if dest.exists() else 0
        print(f"==== CURL {dest.name} try {i}/{attempts} (have {have} bytes) ====", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"==== CURL OK {dest.name} in {(time.time() - t0) / 60:.1f} min ====", flush=True)
            return

        tail = (proc.stderr or proc.stdout or "")[-400:]
        print(f"==== CURL FAIL {dest.name}: exit {proc.returncode} {tail}", flush=True)
        if i < attempts:
            wait = min(60 * i, 300)
            print(f"==== RETRY in {wait}s ====", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"curl failed after {attempts} attempts: {dest}")


def download_hf_xethub(
    repo_id: str,
    filename: str,
    dest: Path,
    *,
    repo_type: str = "dataset",
    expected_size: int | None = None,
    attempts: int = 8,
) -> Path:
    """Download a single HF file using xethub redirect + curl resume."""
    dest = Path(dest)
    size = expected_size or _lookup_size(repo_id, filename)
    if size is None:
        _, size = _resolve_metadata(repo_id, filename, repo_type, attempts=3)

    if dest.exists() and dest.stat().st_size == size:
        print(f"==== SKIP {filename} already complete ({size} bytes) ====", flush=True)
        return dest

    if dest.exists() and dest.stat().st_size > size:
        print(f"==== TRUNCATE {filename}: {dest.stat().st_size} > {size} ====", flush=True)
        dest.unlink()

    if not dest.exists() or dest.stat().st_size == 0:
        adopt_hf_incomplete(repo_id, dest, size)

    location, size = _resolve_metadata(repo_id, filename, repo_type, attempts=attempts)
    curl_download(location, dest, attempts=attempts)

    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"Size mismatch for {filename}: got {got}, expected {size}")

    return dest


def hf_hub_download_small(repo_id: str, filename: str, repo_type: str = "dataset", attempts: int = 8) -> str:
    from huggingface_hub import hf_hub_download

    saved_endpoint = os.environ.get("HF_ENDPOINT")
    last: Exception | None = None
    try:
        for i in range(1, attempts + 1):
            for endpoint in (None, "https://hf-mirror.com"):
                if endpoint is None:
                    os.environ.pop("HF_ENDPOINT", None)
                    source = "huggingface.co"
                else:
                    os.environ["HF_ENDPOINT"] = endpoint
                    source = endpoint
                try:
                    return hf_hub_download(repo_id, filename, repo_type=repo_type)
                except Exception as e:
                    last = e
                    print(f"==== FAIL {filename} via {source}: {type(e).__name__}: {e}", flush=True)
            if i < attempts:
                wait = min(60 * i, 300)
                print(f"==== RETRY in {wait}s ====", flush=True)
                time.sleep(wait)
    finally:
        if saved_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = saved_endpoint

    raise last  # pragma: no cover
