#!/usr/bin/env python3
"""
SeizeIT2 — export cached segments as Edge Impulse CSV files.

One CSV per SEGMENT (not per window). Edge Impulse cuts the 2-s windows itself
(Time series input block: window 2000 ms, stride 500 ms), so every window from a
segment lands on the same side of the train/test split. Subjects listed in
TEST_SUBJECTS go to testing/ in full, giving a genuine held-out test.

Labels (same classes as the ISVLSI 2026 model):
    ictal       -> seizure       (training/ or testing/)
    interictal  -> non_seizure   (training/ or testing/)
    preictal    -> NOT uploaded; written to preictal_eval/ for evaluating the
                   risk layer later (does risk rise before onset?)

The label is the filename prefix before the first dot ("seizure.sub-073_...csv"),
which the Edge Impulse uploader reads as the label.

Edit CONFIG, then:  python 5_export_edge_impulse.py
Requires: numpy.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

CACHE_DIR = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/SiezeIT2_ws/seizeit2_cache"
# Export is large (CSV text) — keep it outside the repo.
EXPORT_DIR = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/SiezeIT2_ws/edge_impulse_export"

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_CSV = SCRIPT_DIR / "seizeit2_segments.csv"        # from 4_extract_segments.py

# Held-out subjects: ALL their segments go to testing/. [] = everything to training/
# (then let Edge Impulse split — still per segment, but subjects appear on both sides).
TEST_SUBJECTS = ["sub-103"]

LEAD_ONLY = False          # True = drop segments of cluster / unknown-lead seizures
INCLUDE_ECG = False        # True = add the ECG as a third column (EEG-only model by default)
DECIMALS = 2               # µV precision written to CSV

# =============== END CONFIG — no need to edit below ===============

try:
    import numpy as np
except ImportError:
    sys.exit("need numpy:  pip install numpy")

LABEL = {"ictal": "seizure", "interictal": "non_seizure", "preictal": "preictal"}


def write_csv(path, eeg, ecg, fs, names):
    n = eeg.shape[1]
    ts = (np.arange(n) * 1000.0 / fs)
    cols = [ts] + [eeg[i] for i in range(eeg.shape[0])]
    header = ["timestamp"] + names
    if INCLUDE_ECG:
        cols.append(ecg[:n] if ecg.size >= n else np.pad(ecg, (0, n - ecg.size), mode="edge"))
        header.append("ecg")
    data = np.column_stack(cols)
    # Timestamps need 5 decimals: one sample = 1000/256 = 3.90625 ms exactly.
    # With fewer decimals Edge Impulse detects ~255.75 Hz instead of 256 Hz.
    fmt = ["%.5f"] + [f"%.{DECIMALS}f"] * (data.shape[1] - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="", fmt=fmt)


def main():
    cache, out = Path(CACHE_DIR).expanduser(), Path(EXPORT_DIR).expanduser()
    if not INDEX_CSV.exists():
        sys.exit(f"segment index not found: {INDEX_CSV}\nRun 4_extract_segments.py first.")
    rows = list(csv.DictReader(open(INDEX_CSV)))
    if LEAD_ONLY:
        lead_onsets = {(r["subject"], r["run"], r["seizure_onset_s"]) for r in rows
                       if r["kind"] == "preictal" and r["lead_seizure"] == "yes"}
        rows = [r for r in rows if r["kind"] == "interictal" or
                (r["subject"], r["run"], r["seizure_onset_s"]) in lead_onsets]

    counts, manifest = Counter(), []
    for r in rows:
        kind, subj = r["kind"], r["subject"]
        if kind == "preictal":
            split = "preictal_eval"
        else:
            split = "testing" if subj in TEST_SUBJECTS else "training"
        d = np.load(cache / r["file"])
        eeg, ecg, fs = d["eeg"], d["ecg"], float(d["fs_eeg"])
        # Generic channel names so left- and right-side subjects share one model input.
        names = ["bte", "cross"]
        label = LABEL[kind]
        dest = out / split / f"{label}.{r['segment_id']}.csv"
        write_csv(dest, eeg, ecg, fs, names)
        counts[(split, label)] += 1
        manifest.append({"split": split, "label": label, "subject": subj,
                         "segment_id": r["segment_id"], "lead_seizure": r["lead_seizure"],
                         "vigilance": r["vigilance"], "duration_s": r["duration_s"],
                         "file": str(dest.relative_to(out))})

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    size = sum(p.stat().st_size for p in out.rglob("*.csv"))
    print("=" * 66)
    print("EDGE IMPULSE EXPORT")
    print("=" * 66)
    for split in ("training", "testing", "preictal_eval"):
        parts = [f"{lab} {n}" for (sp, lab), n in sorted(counts.items()) if sp == split]
        if parts:
            print(f"  {split:<14}: " + ", ".join(parts))
    print(f"  held-out subjects : {', '.join(TEST_SUBJECTS) or 'none'}")
    print(f"  columns           : timestamp (ms), bte, cross" + (", ecg" if INCLUDE_ECG else ""))
    print(f"  size on disk      : {size/1e6:.0f} MB in {out}")
    print("-" * 66)
    print("  Upload (label is read from the filename prefix):")
    print(f'    edge-impulse-uploader --category training "{out}/training/"*.csv')
    if counts.get(("testing", "seizure")) or counts.get(("testing", "non_seizure")):
        print(f'    edge-impulse-uploader --category testing  "{out}/testing/"*.csv')
    print("  Do NOT upload preictal_eval/ — it is for evaluating the risk layer.")
    print("  Impulse: Time series data, window 2000 ms, stride 500 ms, frequency "
          f"{fs:g} Hz.")
    print("=" * 66)


if __name__ == "__main__":
    main()