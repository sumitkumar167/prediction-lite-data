#!/usr/bin/env python3
"""
SeizeIT2 audit — checks every recording and every seizure before any training.

Per recording (run):
  - EEG and ECG files pair up (same run number)
  - durations agree: EEG EDF, ECG EDF, _eeg.json and _events.tsv
  - channel names and sampling rates

Per seizure:
  - the seizure lies inside its recording
  - a clean pre-ictal window exists (default 4 min, no earlier seizure inside it)
  - ECG quality IN THAT PRE-ICTAL WINDOW (not a sample from mid-recording):
    R-peak rate, beat regularity, mains noise, saturation, notch recovery
  - EEG quality in the same window: flat channels, extreme amplitudes, mains

Writes one row per seizure to a CSV with a final `usable` flag.

Edit CONFIG, then:  python3 seizeit2_audit.py
Requires: numpy, pyedflib   (pip install numpy pyedflib)  — no SciPy needed.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

# Folder holding sub-XXX folders (the OUT_DIR of seizeit2_download.py).
SEIZEIT2_ROOT = r"./SeizeIT2"

# Subjects to audit: ["sub-002"], several, or "ALL" (every sub-* folder found).
SUBJECTS = "ALL"

# Pre-ictal window checked before each seizure, seconds (team decision: ~4 min).
PRE_ICTAL_S = 240

# A seizure needs this many seconds since the previous seizure ended, so its
# pre-ictal window is not contaminated by the last seizure's aftermath.
MIN_GAP_FROM_PREVIOUS_S = 240

# EEG peak-to-peak above this (microvolts) in the pre-ictal window = artifact.
EEG_MAX_P2P_UV = 1000

# Durations from different sources may differ by this much (seconds).
DURATION_TOLERANCE_S = 2.0

# Output CSV (one row per seizure).
OUT_CSV = r"./seizeit2_audit_seizures.csv"

# Print one line per seizure (False = per-subject summaries only).
VERBOSE = True

# =============== END CONFIG — no need to edit below ===============

try:
    import numpy as np
    import pyedflib
except ImportError:
    sys.exit("need numpy and pyedflib:  pip install numpy pyedflib")

GOOD_ECG = {"clean", "usable", "recovered"}


# ---------------------------------------------------------------- EDF access

def edf_info(path):
    """Return dict(labels, fs list, duration_s, dims, pmin, pmax) or None."""
    try:
        f = pyedflib.EdfReader(str(path))
    except OSError as exc:
        return {"error": str(exc).splitlines()[0][:120]}
    try:
        n = f.signals_in_file
        return {
            "labels": [f.getLabel(i) for i in range(n)],
            "fs": [float(f.getSampleFrequency(i)) for i in range(n)],
            "duration_s": float(f.getFileDuration()),
            "dims": [f.getPhysicalDimension(i) for i in range(n)],
            "pmin": [f.getPhysicalMinimum(i) for i in range(n)],
            "pmax": [f.getPhysicalMaximum(i) for i in range(n)],
        }
    finally:
        f.close()


def to_uv(sig, dim):
    d = (dim or "").strip().lower().replace("µ", "u")
    if d in ("uv",):
        return sig
    if d == "mv":
        return sig * 1e3
    if d == "v":
        return sig * 1e6
    return sig  # unknown unit: leave as-is


def read_window(path, idx, fs, t0, dur):
    """Read samples [t0, t0+dur) seconds of signal idx."""
    f = pyedflib.EdfReader(str(path))
    try:
        total = f.getNSamples()[idx]
        start = max(0, int(round(t0 * fs)))
        n = min(int(round(dur * fs)), total - start)
        if n <= 0:
            return np.array([])
        return np.asarray(f.readSignal(idx, start=start, n=n), dtype=float)
    finally:
        f.close()


# ------------------------------------------------- signal quality (numpy only)

def dominant_frequency(sig, fs):
    x = np.asarray(sig, float) - np.mean(sig)
    n = min(x.size, int(fs * 30))
    if n < int(fs * 2):
        return None, 0.0
    x = x[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = freqs >= 1.0
    if not band.any():
        return None, 0.0
    peak = float(freqs[band][np.argmax(spec[band])])
    total = float(np.sum(spec[band] ** 2))
    mains = 0.0
    if total > 0:
        for f0 in (50.0, 60.0):
            if f0 < fs / 2:
                sel = (freqs >= f0 - 2) & (freqs <= f0 + 2)
                mains = max(mains, float(np.sum(spec[sel] ** 2) / total))
    return peak, mains


def notch_mains(sig, fs, base_hz, width_hz=1.0):
    x = np.asarray(sig, float) - np.median(sig)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    h = 1
    while base_hz * h < fs / 2.0:
        f0 = base_hz * h
        X[(freqs >= f0 - width_hz) & (freqs <= f0 + width_hz)] = 0
        h += 1
    return np.fft.irfft(X, n=x.size)


def fir_bandpass(sig, fs, lo=5.0, hi=15.0, numtaps=201):
    if numtaps % 2 == 0:
        numtaps += 1
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    nyq = fs / 2.0
    lo, hi = max(lo, 0.1) / nyq, min(hi, nyq * 0.95) / nyq

    def lp(c):
        return np.sinc(c * n) * c * np.hamming(numtaps)

    h = lp(hi) - lp(lo)
    h -= h.mean()
    centred = sig - np.median(sig)
    if sig.size < numtaps * 2:
        return centred
    return np.convolve(centred, h, mode="same")


def estimate_bpm(sig, fs):
    """Pan-Tompkins style. Returns (bpm, rr_cv) or (None, None)."""
    if sig.size < int(fs * 5):
        return None, None
    filt = fir_bandpass(np.asarray(sig, float), fs)
    trim = min(250, filt.size // 10)
    if trim and filt.size > 3 * trim:
        filt = filt[trim:-trim]
    sq = np.diff(filt, prepend=filt[0]) ** 2
    win = max(1, int(0.15 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    sd = np.std(integ)
    if sd == 0:
        return None, None
    minutes = filt.size / fs / 60.0
    min_gap = int(0.3 * fs)
    best = None
    for k in (1.0, 0.7, 0.5, 0.3, 1.5, 2.0):
        above = integ > np.mean(integ) + k * sd
        if not above.any():
            continue
        onsets = np.flatnonzero(above & ~np.r_[False, above[:-1]])
        picks, last = [], -min_gap
        for i in onsets:
            if i - last >= min_gap:
                picks.append(i)
                last = i
        if len(picks) < 3:
            continue
        bpm = len(picks) / minutes
        best = bpm if best is None else best
        rr = np.diff(picks) / fs
        cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else None
        if 40 <= bpm <= 180:
            return bpm, cv
    return best, None


def classify_ecg(sig, fs, pmin, pmax):
    """Returns (verdict, bpm, rr_cv, mains_share)."""
    if sig.size == 0:
        return "empty", None, None, 0.0
    if np.std(sig) == 0 or np.ptp(sig) == 0 or np.mean(sig == 0) > 0.98:
        return "dead", None, None, 0.0

    span = pmax - pmin
    if span > 0:
        tol = 0.01 * span
        rails = float(np.mean((sig >= pmax - tol) | (sig <= pmin + tol)))
        if rails > 0.05:
            return "saturated", None, None, 0.0

    peak, mains = dominant_frequency(sig, fs)
    work = sig
    recovered = False
    if mains > 0.5:
        base = 60.0 if (peak is not None and abs(peak - 60) < 3) else 50.0
        work = notch_mains(sig, fs, base)
        recovered = True

    bpm, cv = estimate_bpm(work, fs)
    if bpm is None:
        return ("mains" if recovered else "no_beats"), None, None, mains
    if not 40 <= bpm <= 180:
        return ("mains" if recovered else "suspect"), bpm, cv, mains
    cv_ = cv if cv is not None else 1.0
    if cv_ <= 0.10:
        tier = "clean"
    elif cv_ <= 0.30:
        tier = "usable"
    else:
        return ("mains" if recovered else "poor"), bpm, cv, mains
    return ("recovered" if recovered else tier), bpm, cv, mains


def check_eeg(path, info, t0, dur):
    """Flag problems in each EEG channel over [t0, t0+dur)."""
    flags = []
    for i, label in enumerate(info["labels"]):
        if re.search(r"annotation", label, re.I):
            continue
        sig = to_uv(read_window(path, i, info["fs"][i], t0, dur), info["dims"][i])
        if sig.size == 0:
            flags.append(f"{label}:empty")
            continue
        if np.std(sig) == 0:
            flags.append(f"{label}:flat")
            continue
        if np.ptp(sig) > EEG_MAX_P2P_UV:
            flags.append(f"{label}:p2p {np.ptp(sig):.0f}uV")
        _, mains = dominant_frequency(sig, info["fs"][i])
        if mains > 0.5:
            flags.append(f"{label}:mains {mains*100:.0f}%")
    return flags


# ---------------------------------------------------------------- metadata

def read_events(path):
    rows = list(csv.DictReader(path.read_text(errors="replace").splitlines(), delimiter="\t"))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def json_duration(path):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return None
    v = d.get("RecordingDuration")
    return float(v) if isinstance(v, (int, float)) else None


# ---------------------------------------------------------------- main

def audit_run(events_path):
    base = str(events_path)[: -len("_events.tsv")]
    eeg_edf = Path(base + "_eeg.edf")
    eeg_json = Path(base + "_eeg.json")
    ecg_dir = events_path.parent.parent / "ecg"
    ecg_edf = ecg_dir / (Path(base).name + "_ecg.edf")

    issues = []
    eeg = edf_info(eeg_edf) if eeg_edf.exists() else None
    if eeg is None:
        issues.append("EEG EDF missing")
    elif "error" in eeg:
        issues.append(f"EEG EDF unreadable: {eeg['error']}")
        eeg = None

    ecg = edf_info(ecg_edf) if ecg_edf.exists() else None
    if ecg is None:
        issues.append("no ECG file for this run")
    elif "error" in ecg:
        issues.append(f"ECG EDF unreadable: {ecg['error']}")
        ecg = None

    rows = read_events(events_path)
    ann_dur = None
    if rows and rows[0].get("recordingDuration") not in (None, "", "n/a"):
        ann_dur = float(rows[0]["recordingDuration"])

    ref = eeg["duration_s"] if eeg else ann_dur
    for name, val in (("ECG EDF", ecg["duration_s"] if ecg else None),
                      ("_eeg.json", json_duration(eeg_json)),
                      ("_events.tsv", ann_dur)):
        if ref is not None and val is not None and abs(val - ref) > DURATION_TOLERANCE_S:
            issues.append(f"duration mismatch: EEG {ref:.0f}s vs {name} {val:.0f}s")

    ecg_idx = None
    if ecg:
        for i, lab in enumerate(ecg["labels"]):
            if re.search(r"E[CK]G", lab, re.I):
                ecg_idx = i
                break
        if ecg_idx is None:
            ecg_idx = 0

    seizures_out = []
    szs = sorted((r for r in rows if r.get("eventType", "").startswith("sz")),
                 key=lambda r: float(r["onset"]))
    prev_end = None
    for r in szs:
        onset, dur = float(r["onset"]), float(r["duration"])
        rec = {
            "run": re.search(r"run-(\d+)", base).group(1),
            "onset_s": onset, "duration_s": dur,
            "type": r.get("eventType", ""),
            "localization": r.get("localization", "n/a"),
            "lateralization": r.get("lateralization", "n/a"),
            "vigilance": r.get("vigilance", "n/a"),
            "in_recording": ref is None or onset + dur <= ref + DURATION_TOLERANCE_S,
            "preictal_ok": onset >= PRE_ICTAL_S and (
                prev_end is None or onset - prev_end >= MIN_GAP_FROM_PREVIOUS_S),
            "ecg_verdict": "n/a", "bpm": "", "rr_cv": "", "ecg_mains_pct": "",
            "eeg_flags": "",
        }
        prev_end = onset + dur

        if rec["in_recording"] and rec["preictal_ok"]:
            t0 = onset - PRE_ICTAL_S
            if ecg is not None:
                fs = ecg["fs"][ecg_idx]
                sig = read_window(ecg_edf, ecg_idx, fs, t0, PRE_ICTAL_S)
                v, bpm, cv, mains = classify_ecg(
                    sig, fs, ecg["pmin"][ecg_idx], ecg["pmax"][ecg_idx])
                rec.update(ecg_verdict=v,
                           bpm=f"{bpm:.0f}" if bpm else "",
                           rr_cv=f"{cv:.2f}" if cv is not None else "",
                           ecg_mains_pct=f"{mains*100:.0f}")
            else:
                rec["ecg_verdict"] = "no_ecg"
            if eeg is not None:
                rec["eeg_flags"] = "; ".join(check_eeg(eeg_edf, eeg, t0, PRE_ICTAL_S))

        rec["usable"] = (rec["in_recording"] and rec["preictal_ok"]
                         and rec["ecg_verdict"] in GOOD_ECG and not rec["eeg_flags"])
        seizures_out.append(rec)

    meta = {
        "eeg_channels": ", ".join(eeg["labels"]) if eeg else "",
        "eeg_fs": sorted(set(eeg["fs"])) if eeg else [],
        "ecg_channel": ecg["labels"][ecg_idx] if ecg else "",
        "ecg_fs": ecg["fs"][ecg_idx] if ecg else None,
        "duration_h": (ref or 0) / 3600,
    }
    return issues, seizures_out, meta


def main():
    root = Path(SEIZEIT2_ROOT).expanduser()
    if not root.is_dir():
        sys.exit(f"SEIZEIT2_ROOT is not a directory: {root}\nEdit CONFIG at the top of this file.")

    if isinstance(SUBJECTS, str) and SUBJECTS.upper() == "ALL":
        subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    else:
        subjects = list(SUBJECTS)
    if not subjects:
        sys.exit("no sub-* folders found under SEIZEIT2_ROOT")

    all_rows, ecg_tally, eeg_flag_count = [], Counter(), 0
    per_subject = {}

    for subj in subjects:
        events = sorted((root / subj).glob("ses-*/eeg/*_events.tsv"))
        print("=" * 70)
        print(f"SEIZEIT2 AUDIT — {subj}   ({len(events)} recordings)")
        print("=" * 70)
        if not events:
            print("  no recordings found (not downloaded?)")
            continue

        run_issues, shown_meta, hours = 0, False, 0.0
        subj_rows = []
        for ev in events:
            issues, rows, meta = audit_run(ev)
            hours += meta["duration_h"]
            if not shown_meta and meta["eeg_channels"]:
                print(f"  EEG channels : {meta['eeg_channels']}  @ {meta['eeg_fs']} Hz")
                print(f"  ECG channel  : {meta['ecg_channel']}  @ {meta['ecg_fs']} Hz")
                shown_meta = True
            run = re.search(r"run-(\d+)", ev.name).group(1)
            if issues:
                run_issues += 1
                for msg in issues:
                    print(f"  run-{run}: {msg}")
            for r in rows:
                r["subject"] = subj
                subj_rows.append(r)
                if VERBOSE:
                    tag = "USABLE" if r["usable"] else "skip  "
                    why = []
                    if not r["in_recording"]:
                        why.append("outside recording")
                    if not r["preictal_ok"]:
                        why.append("no clean pre-ictal")
                    if r["eeg_flags"]:
                        why.append(f"EEG {r['eeg_flags']}")
                    bpm = f"{r['bpm']} bpm cv {r['rr_cv']}" if r["bpm"] else ""
                    print(f"  {tag} run-{run} @{r['onset_s']:>8.0f}s  {r['type']:<16} "
                          f"ECG {r['ecg_verdict']:<9} {bpm:<18} {'; '.join(why)}")

        usable = sum(r["usable"] for r in subj_rows)
        per_subject[subj] = (len(subj_rows), usable, hours, run_issues)
        print(f"  -> {len(subj_rows)} seizures, {usable} usable, {hours:.1f} h, "
              f"{run_issues} recording(s) with issues")
        all_rows.extend(subj_rows)
        for r in subj_rows:
            if r["preictal_ok"] and r["in_recording"]:
                ecg_tally[r["ecg_verdict"]] += 1
            if r["eeg_flags"]:
                eeg_flag_count += 1

    if not all_rows:
        sys.exit("no seizures found")

    fields = ["subject", "run", "onset_s", "duration_s", "type", "localization",
              "lateralization", "vigilance", "in_recording", "preictal_ok",
              "ecg_verdict", "bpm", "rr_cv", "ecg_mains_pct", "eeg_flags", "usable"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in all_rows)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for subj, (n, u, h, ri) in per_subject.items():
        print(f"  {subj}: {u:>3}/{n:<3} seizures usable   {h:7.1f} h   "
              f"{ri} recording(s) with issues")
    tot = len(all_rows)
    usable = sum(r["usable"] for r in all_rows)
    print("-" * 70)
    print(f"  seizures total            : {tot}")
    print(f"  with clean pre-ictal      : {sum(r['preictal_ok'] and r['in_recording'] for r in all_rows)}")
    print(f"  pre-ictal ECG verdicts    : "
          + ", ".join(f"{k} {v}" for k, v in ecg_tally.most_common()))
    print(f"  pre-ictal EEG flagged     : {eeg_flag_count}")
    print(f"  USABLE (EEG + ECG)        : {usable}  ({100*usable/tot:.0f}%)")
    print(f"  written                   : {OUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()