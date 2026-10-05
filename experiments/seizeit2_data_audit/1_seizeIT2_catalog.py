#!/usr/bin/env python3
"""
SeizeIT2 — build a seizure catalogue from annotations only.

The full dataset is ~11,640 hours. Before downloading any signal data, this
fetches only the small text files (participants.tsv, each recording's
_events.tsv) and answers:

  - how many seizures, per subject, of which type
  - where they start (lobe) and on which side
  - which seizures have a clean pre-ictal window (no earlier seizure inside it)
  - which recordings actually have an ECG file
  - which subjects make a good pilot

Then it prints the exact download command for just those subjects' EEG + ECG.

Edit CONFIG, then:  python seizeit2_catalog.py
Standard library only — no pip installs needed.
"""

import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

# Where to cache the downloaded text files and write the catalogue CSVs.
OUT_DIR = r"./seizeit2_meta"

# Pre-ictal window you plan to use, in seconds (team decision: ~4 min).
PRE_ICTAL_S = 240

# A seizure only counts as having a clean pre-ictal window if no earlier
# seizure ended within this many seconds before it (avoids post-ictal overlap).
MIN_GAP_FROM_PREVIOUS_S = 240

# Limit subjects for a quick test run (None = all 125).
MAX_SUBJECTS = None

# How many pilot subjects to suggest.
PILOT_SIZE = 4

# Parallel downloads for the small text files.
WORKERS = 16

# =============== END CONFIG — no need to edit below ===============

REPO = "OpenNeuroDatasets/ds005873"
RAW = f"https://raw.githubusercontent.com/{REPO}/HEAD/"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/HEAD?recursive=1"

LOBE_NAMES = {"front": "frontal", "temp": "temporal", "cen": "central",
              "par": "parietal", "occ": "occipital", "ins": "insular"}


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return None


def cached(path_in_repo, cache_dir):
    local = cache_dir / path_in_repo
    if local.exists():
        return local.read_text(encoding="utf-8", errors="replace")
    text = fetch(RAW + path_in_repo)
    if text is not None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text, encoding="utf-8")
    return text


def list_files():
    """One GitHub API call lists every path in the repository."""
    try:
        text = fetch(TREE_API)
        if text:
            data = json.loads(text)
            if "tree" in data and not data.get("truncated"):
                return [x["path"] for x in data["tree"]]
            if data.get("truncated"):
                print("  file listing truncated — falling back to probing")
            elif "message" in data:
                print(f"  GitHub API: {data['message'][:90]} — falling back to probing")
    except Exception as exc:
        print(f"  file listing failed ({exc}) — falling back to probing")
    return None


def probe_runs(subject, cache_dir):
    """Fallback: discover runs by requesting run-01, run-02, ... until misses."""
    found, misses, run = [], 0, 1
    while misses < 2 and run < 200:
        base = f"{subject}/ses-01/eeg/{subject}_ses-01_task-szMonitoring_run-{run:02d}"
        if cached(base + "_events.tsv", cache_dir) is not None:
            found.append(base + "_events.tsv")
            misses = 0
        else:
            misses += 1
        run += 1
    return found


def has_ecg_probe(events_path, cache_dir):
    ecg = events_path.replace("/eeg/", "/ecg/").replace("_events.tsv", "_ecg.json")
    return cached(ecg, cache_dir) is not None


def lobes_of(loc):
    if not loc or loc in ("n/a", "un"):
        return []
    return [LOBE_NAMES.get(p, p) for p in loc.split("_")]


def parse_events(text):
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def main():
    out = Path(OUT_DIR).expanduser()
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    print("Fetching participants.tsv ...")
    ptxt = cached("participants.tsv", cache)
    if not ptxt:
        sys.exit("could not fetch participants.tsv — check your internet connection")
    participants = {r["participant_id"]: r for r in
                    csv.DictReader(ptxt.splitlines(), delimiter="\t")}
    subjects = sorted(participants)
    if MAX_SUBJECTS:
        subjects = subjects[:MAX_SUBJECTS]
    print(f"  {len(participants)} subjects in dataset, cataloguing {len(subjects)}")

    print("Listing repository files ...")
    paths = list_files()
    ecg_runs = set()
    if paths:
        events_files = [p for p in paths if p.endswith("_events.tsv")
                        and p.split("/")[0] in subjects]
        for p in paths:
            if "/ecg/" in p and p.endswith("_ecg.edf"):
                ecg_runs.add(re.sub(r"_ecg\.edf$", "", p.replace("/ecg/", "/eeg/")))
        print(f"  {len(events_files)} recordings found via file listing")
    else:
        with ThreadPoolExecutor(WORKERS) as ex:
            lists = list(ex.map(lambda s: probe_runs(s, cache), subjects))
        events_files = [p for lst in lists for p in lst]
        print(f"  {len(events_files)} recordings found by probing")

    print("Downloading annotation files (text only) ...")
    with ThreadPoolExecutor(WORKERS) as ex:
        texts = list(ex.map(lambda p: cached(p, cache), events_files))

    if not paths:
        print("Checking which recordings have ECG ...")
        with ThreadPoolExecutor(WORKERS) as ex:
            flags = list(ex.map(lambda p: has_ecg_probe(p, cache), events_files))
        ecg_runs = {p.replace("_events.tsv", "") for p, f in zip(events_files, flags) if f}

    # ---------- build tables ----------
    seizures, recordings = [], []
    for path, text in zip(events_files, texts):
        if not text:
            continue
        subject = path.split("/")[0]
        run = re.search(r"run-(\d+)", path).group(1)
        key = path.replace("_events.tsv", "")
        rows = parse_events(text)
        rec_dur = 0.0
        if rows and rows[0].get("recordingDuration") not in (None, "", "n/a"):
            rec_dur = float(rows[0]["recordingDuration"])
        has_ecg = key in ecg_runs
        recordings.append({"subject": subject, "run": run,
                           "duration_h": round(rec_dur / 3600, 2), "has_ecg": has_ecg})

        szs = sorted((r for r in rows if r.get("eventType", "").startswith("sz")),
                     key=lambda r: float(r["onset"]))
        prev_end = None
        for r in szs:
            onset, dur = float(r["onset"]), float(r["duration"])
            gap_ok = prev_end is None or (onset - prev_end) >= MIN_GAP_FROM_PREVIOUS_S
            seizures.append({
                "subject": subject, "run": run,
                "onset_s": onset, "duration_s": dur,
                "type": r.get("eventType", ""),
                "lateralization": r.get("lateralization", "n/a"),
                "localization": r.get("localization", "n/a"),
                "vigilance": r.get("vigilance", "n/a"),
                "has_ecg": has_ecg,
                "preictal_ok": onset >= PRE_ICTAL_S and gap_ok,
            })
            prev_end = onset + dur

    with open(out / "seizures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seizures[0].keys()) if seizures else ["subject"])
        w.writeheader(); w.writerows(seizures)
    with open(out / "recordings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "run", "duration_h", "has_ecg"])
        w.writeheader(); w.writerows(recordings)

    # ---------- summary ----------
    total_h = sum(r["duration_h"] for r in recordings)
    usable = [s for s in seizures if s["preictal_ok"] and s["has_ecg"]]
    print("\n" + "=" * 64)
    print("SEIZEIT2 CATALOGUE")
    print("=" * 64)
    print(f"  subjects catalogued      : {len(subjects)}")
    print(f"  recordings               : {len(recordings)}  "
          f"({sum(r['has_ecg'] for r in recordings)} with ECG)")
    print(f"  recorded hours           : {total_h:,.0f}")
    print(f"  seizures                 : {len(seizures)}")
    print(f"  with clean {PRE_ICTAL_S//60}-min pre-ictal : "
          f"{sum(s['preictal_ok'] for s in seizures)}")
    print(f"  ... and ECG present      : {len(usable)}   <- usable for EEG+ECG work")

    def show(title, counter, total):
        print(f"\n  {title}")
        for k, v in counter.most_common():
            print(f"    {k:<14} {v:>4}  ({100*v/total:4.1f}%)")

    if seizures:
        lobe_counter = Counter()
        for s in seizures:
            lobes = lobes_of(s["localization"])
            if not lobes:
                lobe_counter["unclear"] += 1
            for lb in lobes:
                lobe_counter[lb] += 1
        show("Seizure onset lobe (multi-lobe seizures counted in each lobe):",
             lobe_counter, len(seizures))
        show("Seizure onset side:",
             Counter(s["lateralization"] if s["lateralization"] not in ("", "n/a")
                     else "un" for s in seizures), len(seizures))
        show("Vigilance:", Counter(s["vigilance"] or "n/a" for s in seizures), len(seizures))

    # ---------- pilot suggestion ----------
    per_subj = defaultdict(list)
    for s in usable:
        per_subj[s["subject"]].append(s)

    def dominant(lst, key, fn=lambda x: x):
        c = Counter(fn(s[key]) for s in lst)
        return c.most_common(1)[0][0] if c else "unclear"

    ranked = sorted(per_subj.items(), key=lambda kv: -len(kv[1]))
    pilot, seen_lobes = [], set()
    for subj, lst in ranked:  # first pass: diversity of lobe
        lobe = dominant(lst, "localization",
                        lambda l: (lobes_of(l) or ["unclear"])[0])
        if lobe not in seen_lobes and lobe != "unclear" and len(lst) >= 3:
            pilot.append(subj); seen_lobes.add(lobe)
        if len(pilot) == PILOT_SIZE:
            break
    for subj, lst in ranked:  # fill up with most seizures
        if len(pilot) == PILOT_SIZE:
            break
        if subj not in pilot:
            pilot.append(subj)

    print("\n  Suggested pilot (usable seizures, diverse onset lobes):")
    for subj in pilot:
        lst = per_subj[subj]
        lobe = dominant(lst, "localization", lambda l: (lobes_of(l) or ["unclear"])[0])
        side = dominant(lst, "lateralization")
        print(f"    {subj}: {len(lst)} usable seizures, mostly {lobe}, side {side}")

    print("\n  Download ONLY EEG + ECG for the pilot (skips EMG and movement):")
    inc = " ".join(f'--include "{s}/*/eeg/*" --include "{s}/*/ecg/*"' for s in pilot)
    print(f"    aws s3 sync --no-sign-request s3://openneuro.org/ds005873 ./SeizeIT2 \\\n"
          f'      --exclude "*" {inc}')
    print(f"\n  Written: {out/'seizures.csv'}  and  {out/'recordings.csv'}")
    print("=" * 64)


if __name__ == "__main__":
    main()