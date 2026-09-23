"""Prefetch large ModelScope-mirrored base weights into the HF hub cache.

Deploy-pack module (sf-jev-deploy route ⑥); not part of upstream Open-Jev's
inference path — jev.model / jev.serving are untouched. Inspired by kev's
KEV_BASE_HUB=modelscope switch (kev/kev/evaluate.py:_ms_snapshot): the SF
GPU-function network reaches modelscope.cn directly at full speed, while the
injected HF proxy (HF_ENDPOINT) is bandwidth-limited — a multi-GB base
download over the proxy can exceed the startup budget (ledger #29, observed
2026-09-23).

Equivalence evidence (2026-09-23 probe): for Qwen/Qwen3.5-{2B,9B} every
large LFS file's ModelScope master Sha256 equals the pinned HF revision's
LFS oid (2B: aa33250c…; 9B: db6f444b…/31c7d7e2…/7ec36ba3…/b62b0c4c…;
tokenizer.json 5f9e4d49…). Small files (config, tokenizer_config,
index.json) stay on the HF proxy — the pin revision stays authoritative.

Flow: read the checkpoint's model.json (same pin source DecisionModel.load
uses) -> list both trees -> for each file over MS_MIN_BYTES whose MS Sha256
matches the HF lfs oid, stream it from ModelScope into the HF cache
(snapshots/<revision>/ layout, blobs + symlink). huggingface_hub then finds
a complete snapshot and from_pretrained never downloads the big file from
the slow proxy. Any mismatch or missing file: WARN and leave it to the
normal HF path (behavior identical to unprefetched deployment).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# MODELSCOPE_ENDPOINT (e.g. an injected mirror like hf.sc4.ai for HF) replaces the
# public modelscope.cn base; default keeps the direct public endpoint.
MS_API = os.environ.get("MODELSCOPE_ENDPOINT", "https://modelscope.cn").rstrip("/") + "/api/v1/models"
HF_API = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_TOKEN_HEADER = "authorization: Bearer %s" % os.environ["HF_TOKEN"] if os.environ.get("HF_TOKEN") else None
MS_MIN_BYTES = int(os.environ.get("MS_MIN_BYTES", str(100 << 20)))   # 100 MB default per-file threshold
MS_READ_TIMEOUT = int(os.environ.get("MS_READ_TIMEOUT", "120"))


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "openjev-prefetch"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _retry(fn, what, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            wait = 15 * attempt
            print(f"prefetch: {what} failed (attempt {attempt}/{attempts}): {e!r}; retrying in {wait}s",
                  flush=True)
            if attempt == attempts:
                raise
            time.sleep(wait)


def hf_tree(repo, revision):
    files = _retry(lambda: _get(
        f"{HF_API}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
        "?recursive=true"), f"HF tree {repo}@{revision[:10]}")
    out = {}
    for f in files:
        if f.get("type") == "file":
            out[f["path"]] = {"size": f.get("size", 0),
                              "lfs_oid": (f.get("lfs") or {}).get("oid")}
    return out


def ms_tree(repo):
    data = _retry(lambda: _get(f"{MS_API}/{repo}/repo/files?Revision=master&Recursive=true"),
                  f"MS file list {repo}")
    out = {}
    for f in data.get("Data", {}).get("Files", []):
        if f.get("Type") == "tree":
            continue
        out[f["Path"]] = {"size": f.get("Size", 0), "sha256": f.get("Sha256")}
    return out


def _file_sha256(path, chunk=8 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hf_cache_dir():
    root = os.environ.get("HF_HOME")
    if root:
        return Path(root) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def already_prefetched(repo, revision, path, size):
    """A previously completed prefetch (or a slow-proxy download that somehow
    finished) already satisfies the cache: same marker name, same size."""
    snap = hf_cache_dir() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / path
    if not snap.exists():
        return False
    try:
        return snap.stat().st_size == size
    except OSError:
        return False


def install_blob(repo, revision, path, ms_meta):
    """Stream the file from ModelScope into the HF cache layout that
    huggingface_hub expects: blobs/<fname>.incomplete-<rand> -> blobs/<fname>
    (keyed by the MS sha256, which we verified equals the HF lfs oid, so the
    blob doubles as the etag), plus a relative symlink at
    snapshots/<revision>/<path>. from_pretrained then sees a complete
    snapshot and skips the download."""
    import contextlib
    base = hf_cache_dir() / f"models--{repo.replace('/', '--')}"
    snap = base / "snapshots" / revision / path
    blob = base / "blobs" / ms_meta["sha256"]          # sha256 == HF lfs oid -> a valid blob key
    (base / "snapshots" / revision).mkdir(parents=True, exist_ok=True)
    (base / "blobs").mkdir(parents=True, exist_ok=True)
    snap.parent.mkdir(parents=True, exist_ok=True)
    if snap.is_symlink() and snap.resolve() == blob.resolve() and blob.exists():
        print(f"prefetch: {path}: already installed (blob+symlink); skip", flush=True)
        return
    url = f"{MS_API}/{repo}/repo?FilePath={urllib.parse.quote(path)}&Revision=master"
    size = ms_meta["size"] or 0
    got = 0
    for attempt in range(1, 5):
        tmp = blob.with_name(blob.name + f".incomplete.{os.getpid()}")
        try:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            with urllib.request.urlopen(url, timeout=MS_READ_TIMEOUT) as r, open(tmp, "wb") as w:
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    w.write(chunk)
                    got += len(chunk)
                    if got >= (64 << 20) and got % (64 << 20) < (8 << 20):
                        print(f"  {path}: {got / 1e9:.2f}/"f"{size / 1e9:.2f} GB", flush=True)
            if size and got != size:
                raise IOError(f"incomplete read: {got} of {size} bytes")
            os.replace(tmp, blob)                       # atomic: .incomplete is never a usable blob
            # symlink refresh (from an earlier partial attempt's regular file)
            with contextlib.suppress(OSError):
                if snap.is_symlink() or snap.exists():
                    snap.unlink()
            snap.symlink_to(os.path.relpath(blob, snap.parent))
            print(f"  {path}: {got / 1e9:.2f} GB done -> {snap}", flush=True)
            return
        except Exception as e:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            wait = 15 * attempt
            print(f"  {path}: install failed (attempt {attempt}/4): {e!r}; retrying in {wait}s", flush=True)
            if attempt == 4:
                raise
            got = 0
            time.sleep(wait)


def prefetch(repo, revision):
    hf = hf_tree(repo, revision)
    ms = ms_tree(repo)
    picked = []
    for path, meta in hf.items():
        size = meta["size"] or 0
        if size < MS_MIN_BYTES:
            continue
        if not meta["lfs_oid"]:
            print(f"prefetch: WARN {path}: {size} bytes over threshold but not LFS on HF; left to HF path",
                  flush=True)
            continue
        ms_meta = ms.get(path)
        if not ms_meta or not ms_meta["sha256"]:
            print(f"prefetch: WARN {path}: not on ModelScope (or no sha256); left to HF path", flush=True)
            continue
        if ms_meta["sha256"] != meta["lfs_oid"]:
            print(f"prefetch: WARN {path}: ModelScope sha256 {ms_meta['sha256'][:12]}… != "
                  f"HF lfs oid {meta['lfs_oid'][:12]}…; left to HF path", flush=True)
            continue
        if already_prefetched(repo, revision, path, ms_meta["size"]):
            continue
        picked.append((path, ms_meta, size))
    if not picked:
        print("prefetch: nothing to do (no large file verified equal on both hubs)", flush=True)
        return
    total = sum(p[1]["size"] or 0 for p in picked)
    print(f"prefetch: {repo}: {len(picked)} large file(s), {total / 1e9:.2f} GB from ModelScope "
          f"(sha256-verified vs HF lfs oid; small files stay on the HF proxy)", flush=True)
    for path, ms_meta, _hf_size in picked:
        _retry(lambda: install_blob(repo, revision, path, ms_meta), f"install {path}", attempts=1)
        snap = hf_cache_dir() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / path
        if not (snap.exists() and snap.stat().st_size == (ms_meta["size"] or 0)):
            print(f"prefetch: ERROR {path}: expected {snap} with {ms_meta['size']} bytes; "
                  "HF path will re-fetch", flush=True)
    print("prefetch: done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True,
                        help="read model.json from this dir (same pin DecisionModel.load uses)")
    args = parser.parse_args()
    cfg = json.loads((Path(args.checkpoint) / "model.json").read_text())
    prefetch(cfg["model_id"], cfg["revision"])


if __name__ == "__main__":
    main()
