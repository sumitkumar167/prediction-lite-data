#!/usr/bin/env python3
"""
SeizeIT2 — check the size, then download only what you need.

Step 1 (DRY_RUN = True, the default):
    Lists the chosen subjects' files on OpenNeuro's public S3 bucket and
    reports file counts and sizes per subject and modality, plus whether it
    all fits on your disk. Nothing is downloaded.

Step 2 (DRY_RUN = False):
    Downloads those files. Already-complete files are skipped, interrupted
    downloads resume from where they stopped, and every file's size is
    checked against the server's.

Edit CONFIG, then:  python3 seizeit2_download.py
Standard library only. If plain HTTPS access to S3 fails, it falls back to
the AWS CLI (brew install awscli) when that is installed.
"""

import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

# Where the data will be saved (keeps OpenNeuro's sub-XXX/ses-01/eeg/... layout).
OUT_DIR = r"./SeizeIT2"

# Subjects to fetch — the pilot suggested by seizeit2_catalog.py.
SUBJECTS = ["sub-073", "sub-087", "sub-002", "sub-103"]

# Modalities to fetch. Options: "eeg", "ecg", "emg", "mov".
# eeg also contains the _events.tsv annotation files.
MODALITIES = ["eeg", "ecg"]

# True  = only report sizes (safe first run)
# False = actually download
DRY_RUN = False

# Refuse to download if it would leave less than this much free space.
KEEP_FREE_GB = 5

# Parallel downloads. 3-4 is plenty; more rarely helps on home internet.
WORKERS = 4

# =============== END CONFIG — no need to edit below ===============

BUCKET = "openneuro.org"
DATASET = "ds005873"
S3 = f"https://s3.amazonaws.com/{BUCKET}"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
CHUNK = 1 << 20  # 1 MiB


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


# ---------------- listing ----------------

def list_https(prefix):
    """ListObjectsV2 over plain HTTPS (anonymous). Returns [(key, size)]."""
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        url = f"{S3}/?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.findtext("s3:IsTruncated", default="false", namespaces=NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        else:
            return out


def list_awscli(prefix):
    """Fallback: AWS CLI listing. Returns [(key, size)]."""
    if not shutil.which("aws"):
        raise RuntimeError("HTTPS listing failed and the AWS CLI is not installed "
                           "(brew install awscli)")
    res = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive", f"s3://{BUCKET}/{prefix}"],
        capture_output=True, text=True, check=True)
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(None, 3)  # date, time, size, key
        if len(parts) == 4 and parts[2].isdigit():
            out.append((parts[3], int(parts[2])))
    return out


_use_cli = False


def list_prefix(prefix):
    global _use_cli
    if not _use_cli:
        try:
            return list_https(prefix)
        except Exception as exc:
            print(f"  HTTPS listing failed ({type(exc).__name__}); switching to AWS CLI")
            _use_cli = True
    return list_awscli(prefix)


# ---------------- downloading ----------------

def download(key, size, out_root):
    rel = key[len(DATASET) + 1:]
    dest = out_root / rel
    if dest.exists() and dest.stat().st_size == size:
        return "skipped", key, size

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if _use_cli:
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors",
                        f"s3://{BUCKET}/{key}", str(part)], check=True)
    else:
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        req = urllib.request.Request(f"{S3}/{urllib.parse.quote(key)}")
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as fh:
                if have and r.status != 206:  # server ignored Range: start over
                    fh.truncate(0)
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    fh.write(buf)
        except urllib.error.HTTPError as e:
            if e.code != 416:  # 416 = already have the whole file
                raise

    got = part.stat().st_size
    if got != size:
        return f"INCOMPLETE ({human(got)} of {human(size)}) — re-run to resume", key, size
    part.replace(dest)
    return "ok", key, size


# ---------------- main ----------------

def main():
    out_root = Path(OUT_DIR).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"SEIZEIT2 {'SIZE CHECK (dry run)' if DRY_RUN else 'DOWNLOAD'}")
    print(f"  subjects   : {', '.join(SUBJECTS)}")
    print(f"  modalities : {', '.join(MODALITIES)}")
    print("=" * 66)

    files, total = [], 0
    for subj in SUBJECTS:
        for mod in MODALITIES:
            listing = list_prefix(f"{DATASET}/{subj}/ses-01/{mod}/")
            size = sum(s for _, s in listing)
            total += size
            files.extend(listing)
            have = sum(1 for k, s in listing
                       if (out_root / k[len(DATASET) + 1:]).exists()
                       and (out_root / k[len(DATASET) + 1:]).stat().st_size == s)
            note = f"  ({have} already downloaded)" if have else ""
            flag = "  <- nothing found, check subject ID" if not listing else ""
            print(f"  {subj} / {mod:<4} {len(listing):>4} files  {human(size):>10}{note}{flag}")

    remaining = sum(s for k, s in files
                    if not ((out_root / k[len(DATASET) + 1:]).exists()
                            and (out_root / k[len(DATASET) + 1:]).stat().st_size == s))
    free = shutil.disk_usage(out_root).free
    after = free - remaining

    print("-" * 66)
    print(f"  total selected      : {len(files)} files, {human(total)}")
    print(f"  still to download   : {human(remaining)}")
    print(f"  free on this disk   : {human(free)}")
    print(f"  free after download : {human(after)}")
    fits = after >= KEEP_FREE_GB * 1024 ** 3
    print(f"  verdict             : "
          f"{'FITS' if fits else 'DOES NOT FIT'} (keeping at least {KEEP_FREE_GB} GB free)")
    print("=" * 66)

    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False in CONFIG to download.")
        return
    if not fits:
        sys.exit("Not downloading: not enough free space. Reduce SUBJECTS or free up disk.")
    if remaining == 0:
        print("Everything is already downloaded.")
        return

    todo = [(k, s) for k, s in files
            if not ((out_root / k[len(DATASET) + 1:]).exists()
                    and (out_root / k[len(DATASET) + 1:]).stat().st_size == s)]
    print(f"Downloading {len(todo)} files ({human(remaining)}) with {WORKERS} workers ...")

    done_bytes, failures = 0, []
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(download, k, s, out_root) for k, s in todo]
        for i, fut in enumerate(futures, 1):
            try:
                status, key, size = fut.result()
            except Exception as exc:
                key = todo[i - 1][0]
                status, size = f"FAILED ({type(exc).__name__}: {exc})", 0
            if status == "ok":
                done_bytes += size
            elif status != "skipped":
                failures.append((key, status))
            pct = 100 * done_bytes / remaining if remaining else 100
            print(f"  [{i:>4}/{len(todo)}] {pct:5.1f}%  {Path(key).name}  {status}")

    print("=" * 66)
    print(f"  downloaded : {human(done_bytes)}")
    if failures:
        print(f"  problems   : {len(failures)} file(s) — just re-run; finished files are skipped "
              f"and partial ones resume")
        for k, s in failures[:10]:
            print(f"    {Path(k).name}: {s}")
    else:
        print("  all files complete and size-verified")
    print("=" * 66)


if __name__ == "__main__":
    main()