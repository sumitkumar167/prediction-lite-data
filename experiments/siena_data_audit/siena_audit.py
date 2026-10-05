#!/usr/bin/env python3
"""
Siena Scalp EEG — per-subject audit.

Checks a subject folder before any pipeline work:
  - what channels exist, and whether our 4 bipolar pairs can be built
  - whether an EKG/ECG channel is present
  - sampling rate(s) and recording duration
  - seizure times parsed from Seizures-list-PNxx.txt (midnight-aware offsets)
  - localization / lateralization from subject_info.csv

Edit the CONFIG block below, then just run:

    python siena_audit.py

Requires: mne  (pip install mne)
"""

import csv
import re
import sys
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

# Root folder of the downloaded Siena dataset (the one holding PN00, PN01, ...).
SIENA_ROOT = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/Scalp_EEG_Dataset/physionet.org/files/siena-scalp-eeg/1.0.0"

# Which subjects to audit.
#   ["PN01"]            -> just PN01
#   ["PN01", "PN03"]    -> a couple
#   "ALL"               -> every PNxx folder found under SIENA_ROOT
SUBJECTS = ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07"]

# subject_info.csv — leave as-is to look for it in SIENA_ROOT, or give a full path.
SUBJECT_INFO_CSV = None

# Print every channel name in each EDF (verbose).
LIST_ALL_CHANNELS = False

# The four bipolar pairs Prediction-Lite uses.
BIPOLAR_PAIRS = [("F3", "C3"), ("C3", "P3"), ("F4", "C4"), ("C4", "P4")]

# EEG is resampled to this to match the trained model; cardiac stays native.
TARGET_SFREQ = 256.0

# =============== END CONFIG — no need to edit below ===============

# EDF reading: MNE first, pyedflib as fallback.
# MNE pulls in SciPy lazily, and some macOS/ARM SciPy wheels fail to dlopen;
# pyedflib reads EDF headers without SciPy, so the audit still runs.
def read_edf_meta(path):
    """Return (channel_names, sfreq, n_samples, backend_name)."""
    problems = []

    try:
        import mne
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        return list(raw.ch_names), float(raw.info["sfreq"]), int(raw.n_times), "mne"
    except Exception as exc:
        problems.append(f"mne -> {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

    try:
        import pyedflib
        f = pyedflib.EdfReader(str(path))
        try:
            names = list(f.getSignalLabels())
            freqs = [float(f.getSampleFrequency(i)) for i in range(f.signals_in_file)]
            sfreq = max(freqs) if freqs else 0.0
            n = int(round(f.getFileDuration() * sfreq))
            return names, sfreq, n, "pyedflib"
        finally:
            f.close()
    except Exception as exc:
        problems.append(f"pyedflib -> {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

    raise RuntimeError(
        "could not read EDF with either backend:\n    " + "\n    ".join(problems)
        + "\n  Fix MNE/SciPy with:  pip install --upgrade --force-reinstall numpy scipy"
        + "\n  or install the fallback reader:  pip install pyedflib"
    )

REQUIRED_ELECTRODES = sorted({e for pair in BIPOLAR_PAIRS for e in pair})


def normalise(name):
    """'EEG F3' / 'EEG F3-Ref' / 'F3 ' -> 'F3' for matching."""
    n = name.upper()
    n = re.sub(r"\bEEG\b", "", n)
    n = re.sub(r"-?REF\b", "", n)
    n = re.sub(r"[^A-Z0-9]", "", n)
    return n.strip()


def find_electrode(ch_names, electrode):
    """Return the raw channel name matching an electrode label, or None."""
    target = electrode.upper()
    for ch in ch_names:
        if normalise(ch) == target:
            return ch
    return None


def find_cardiac(ch_names):
    """Siena labels the cardiac channel EKG; other sets use ECG. Match both."""
    hits = []
    for ch in ch_names:
        if re.search(r"E[KC]G", ch, flags=re.IGNORECASE):
            hits.append(ch)
    return hits


def to_seconds(stamp):
    """'19:58:36' -> seconds since midnight."""
    h, m, sec = (int(x) for x in stamp.split(":"))
    return h * 3600 + m * 60 + sec


def offset_from_start(reg_start, stamp):
    """
    Seconds from registration start to stamp, handling recordings that
    cross midnight (Siena has overnight registrations, e.g. 19:00 -> 08:29).
    """
    if not reg_start or not stamp:
        return None
    delta = to_seconds(stamp) - to_seconds(reg_start)
    if delta < 0:
        delta += 24 * 3600  # rolled past midnight
    return delta


def parse_seizure_list(path):
    """
    Pull 'Registration start/end' and 'Seizure start/end' times out of
    Seizures-list-PNxx.txt. The format is loosely structured, so match on
    keywords rather than fixed line positions.
    """
    if not path.exists():
        return []

    entries = []
    current = {}
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        fname = re.search(r"(PN\d+(?:-\d+)?\.edf)", line, flags=re.IGNORECASE)
        if fname:
            if current:
                entries.append(current)
            current = {"file": fname.group(1)}
            continue

        time = re.search(r"(\d{1,2}[.:]\d{2}[.:]\d{2})", line)
        if not time:
            continue
        stamp = time.group(1).replace(".", ":")
        low = line.lower()

        if "registration start" in low:
            current["reg_start"] = stamp
        elif "registration end" in low:
            current["reg_end"] = stamp
        elif "seizure start" in low or "start time" in low:
            current.setdefault("seizure_starts", []).append(stamp)
        elif "seizure end" in low or "end time" in low:
            current.setdefault("seizure_ends", []).append(stamp)

    if current:
        entries.append(current)
    return entries


def read_subject_info(csv_path, subject):
    if not csv_path or not Path(csv_path).exists():
        return None
    with open(csv_path, newline="", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            joined = " ".join(str(v) for v in row.values() if v)
            if subject.upper() in joined.upper():
                return row
    return None


def audit_edf(edf_path):
    print(f"\n--- {edf_path.name} ---")
    try:
        ch_names, sfreq, n_times, backend = read_edf_meta(edf_path)
    except RuntimeError as exc:
        print(f"  COULD NOT READ: {exc}")
        return {"file": edf_path.name, "sfreq": None, "duration_min": None,
                "montage_ok": False, "missing": ["<unread>"], "cardiac": [],
                "all_channels": []}

    dur_s = n_times / sfreq if sfreq else 0.0
    print(f"  read via      : {backend}")
    print(f"  sampling rate : {sfreq:g} Hz")
    print(f"  duration      : {dur_s/60:.1f} min ({dur_s/3600:.2f} h)")
    print(f"  channels      : {len(ch_names)}")

    # Electrodes needed for the bipolar montage
    found, missing = {}, []
    for e in REQUIRED_ELECTRODES:
        ch = find_electrode(ch_names, e)
        if ch:
            found[e] = ch
        else:
            missing.append(e)

    odd = [c for c in ch_names if len(normalise(c)) <= 1]
    if odd:
        print(f"  ODD LABELS    : {', '.join(repr(c) for c in odd)} "
              f"\u2014 likely corrupted (e.g. 'O1' stored as '1')")

    if missing:
        print(f"  MONTAGE       : INCOMPLETE — missing {', '.join(missing)}")
    else:
        print("  MONTAGE       : OK — all 6 electrodes present")
        for a, b in BIPOLAR_PAIRS:
            print(f"                  {a}-{b}  =  {found[a]!r} - {found[b]!r}")

    # Cardiac channel
    cardiac = find_cardiac(ch_names)
    if cardiac:
        print(f"  CARDIAC       : {len(cardiac)} channel(s) — {', '.join(cardiac)}")
    else:
        print("  CARDIAC       : NONE FOUND — no ECG/EKG in this file")

    if sfreq != TARGET_SFREQ:
        print(f"  NOTE          : resample EEG {sfreq:g} -> {TARGET_SFREQ:g} Hz "
              f"for the trained model; keep cardiac at {sfreq:g} Hz for R-peak precision")

    return {
        "file": edf_path.name,
        "sfreq": sfreq,
        "duration_min": dur_s / 60,
        "montage_ok": not missing,
        "missing": missing,
        "cardiac": cardiac,
        "all_channels": ch_names,
    }


def audit_subject(folder, info_csv):
    subject = folder.name
    print("=" * 62)
    print(f"SIENA AUDIT — {subject}")
    print("=" * 62)

    info = read_subject_info(info_csv, subject)
    if info:
        print("\nsubject_info.csv:")
        for k, v in info.items():
            if v:
                print(f"  {k.strip()}: {str(v).strip()}")
    elif info_csv:
        print(f"\nsubject_info.csv: no row matched {subject}")

    seizures = parse_seizure_list(folder / f"Seizures-list-{subject}.txt")
    if seizures:
        print(f"\nSeizure list ({len(seizures)} block(s)):")
        for e in seizures:
            rs, re_ = e.get("reg_start"), e.get("reg_end")
            print(f"  file: {e.get('file','?')}")
            if rs and re_:
                dur = offset_from_start(rs, re_)
                crosses = to_seconds(re_) < to_seconds(rs)
                print(f"    registration : {rs}\u2013{re_}  ({dur/3600:.2f} h"
                      f"{', CROSSES MIDNIGHT' if crosses else ''})")
            else:
                print(f"    registration : {rs or '?'}\u2013{re_ or '?'}")
            starts = e.get("seizure_starts", [])
            ends = e.get("seizure_ends", [])
            for i, st in enumerate(starts):
                en = ends[i] if i < len(ends) else None
                off = offset_from_start(rs, st)
                line = f"    seizure {i+1}: {st}"
                if en:
                    line += f"\u2013{en}"
                if off is not None:
                    line += f"   @ {off:.0f}s ({off/60:.1f} min) from registration start"
                if en:
                    d = to_seconds(en) - to_seconds(st)
                    line += f", lasts {d + (86400 if d < 0 else 0):.0f}s"
                print(line)
            if not starts:
                print("    (no seizure times parsed)")
    else:
        print("\nSeizure list: not found or unparsed \u2014 open the .txt manually")

    edfs = sorted(folder.glob("*.edf"))
    if not edfs:
        print("  no .edf files in this folder\n")
        return []

    # The seizure list's filenames don't always match what's on disk
    # (e.g. list says PN01.edf, disk has PN01-1.edf).
    on_disk = {p.name.lower() for p in edfs}
    named = [e["file"] for e in seizures if e.get("file")]
    unmatched = [n for n in named if n.lower() not in on_disk]
    if unmatched:
        print(f"\n  FILENAME MISMATCH: seizure list names {', '.join(unmatched)}, "
              f"but disk has {', '.join(sorted(on_disk))}")
        if len(edfs) == 1:
            print("    (single EDF in folder \u2014 mapping is unambiguous, safe to proceed)")
        else:
            print("    (MULTIPLE EDFs \u2014 resolve which file each seizure belongs to "
                  "before extracting windows)")

    results = [audit_edf(p) for p in edfs]

    if LIST_ALL_CHANNELS:
        for r in results:
            print(f"\nchannels in {r['file']}:")
            for ch in r["all_channels"]:
                print(f"    {ch}")

    print("\n" + "=" * 62)
    usable = [r for r in results if r["montage_ok"] and r["cardiac"]]
    print(f"VERDICT for {subject}: {len(usable)}/{len(results)} file(s) "
          f"usable for EEG+ECG work")
    for r in results:
        flags = []
        if not r["montage_ok"]:
            flags.append("missing " + ",".join(r["missing"]))
        if not r["cardiac"]:
            flags.append("no cardiac channel")
        print(f"  {r['file']}: {'OK' if not flags else '; '.join(flags)}")
    print("=" * 62 + "\n")

    for r in results:
        r["subject"] = subject
    return results


def main():
    root = Path(SIENA_ROOT).expanduser()
    if not root.is_dir():
        sys.exit(f"SIENA_ROOT is not a directory: {root}\n"
                 f"Edit the CONFIG block at the top of this file.")

    info_csv = SUBJECT_INFO_CSV
    if info_csv is None:
        guess = root / "subject_info.csv"
        info_csv = str(guess) if guess.exists() else None

    if isinstance(SUBJECTS, str) and SUBJECTS.upper() == "ALL":
        folders = sorted(p for p in root.iterdir()
                         if p.is_dir() and re.fullmatch(r"PN\d+", p.name, re.IGNORECASE))
    else:
        folders = []
        for name in SUBJECTS:
            f = root / name
            if f.is_dir():
                folders.append(f)
            else:
                print(f"skipping {name}: no folder at {f}")

    if not folders:
        sys.exit("no subject folders to audit \u2014 check SIENA_ROOT and SUBJECTS.")

    all_results = []
    for f in folders:
        all_results.extend(audit_subject(f, info_csv))

    if len(folders) > 1:
        print("=" * 62)
        print("SUMMARY")
        print("=" * 62)
        ok = [r for r in all_results if r["montage_ok"] and r["cardiac"]]
        subs_ok = sorted({r["subject"] for r in ok})
        print(f"  files usable for EEG+ECG : {len(ok)}/{len(all_results)}")
        print(f"  subjects with >=1 usable : {len(subs_ok)}/{len(folders)}"
              f"  ({', '.join(subs_ok) if subs_ok else 'none'})")
        bad = [r for r in all_results if not (r["montage_ok"] and r["cardiac"])]
        for r in bad:
            why = []
            if not r["montage_ok"]:
                why.append("missing " + ",".join(r["missing"]))
            if not r["cardiac"]:
                why.append("no cardiac")
            print(f"    {r['subject']}/{r['file']}: {'; '.join(why)}")
        print("=" * 62)


if __name__ == "__main__":
    main()