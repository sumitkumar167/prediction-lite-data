#!/usr/bin/env python3
"""
SeizeIT2 audit (v2) — checks every recording and every seizure before training.

Per recording (run):
  - EEG and ECG files pair up; durations agree (EEG EDF, ECG EDF, json, events)
  - channel names and sampling rates

Per seizure, in its pre-ictal window (default 4 min):
  - ECG quality: R-peak rate, beat regularity, mains (with notch recovery),
    saturation
  - EEG quality PER 2-SECOND WINDOW (Prediction-Lite's window length), after a
    mains notch and per-window detrending. A window is bad if any channel is
    flat or exceeds the amplitude limit. The seizure keeps its EEG if enough
    of the pre-ictal windows are clean — one chewing spike no longer rejects
    the whole four minutes.
  - lead vs cluster seizure: time since the previous seizure. Runs in SeizeIT2
    are discontinuous and their timestamps anonymized, so across runs this is
    a LOWER BOUND (time from the run start) unless the EDF headers carry real
    start times.

Writes one row per seizure to a CSV with `usable` and `usable_lead` flags.

Edit CONFIG, then:  python3 seizeit2_audit.py
Requires: numpy, pyedflib  (pip install numpy pyedflib) — no SciPy needed.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

# Dataset location (outside the repo). Holds the sub-XXX folders.
SEIZEIT2_ROOT = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/SiezeIT2_ws/SeizeIT2"

# Outputs are written next to this script (i.e. inside the git repo).
SCRIPT_DIR = Path(__file__).resolve().parent
SUBJECTS = "ALL"                        # ["sub-002"], several, or "ALL"

PRE_ICTAL_S = 240                       # pre-ictal window checked (s)
MIN_GAP_FROM_PREVIOUS_S = 240           # min time since previous seizure ended
LEAD_GAP_S = 1800                       # >= this since previous = lead seizure

WIN_S = 2.0                             # EEG analysis window (Prediction-Lite: 2 s)
EEG_MAX_P2P_UV = 1000                   # window rejected above this peak-to-peak
MIN_CLEAN_FRACTION = 0.75               # share of clean windows needed to keep EEG
MAINS_HZ = 50                           # Europe (SeizeIT2) = 50 Hz

DURATION_TOLERANCE_S = 2.0
OUT_CSV = SCRIPT_DIR / "seizeit2_audit_seizures.csv"
VERBOSE = True                          # one line per seizure

# =============== END CONFIG — no need to edit below ===============

try:
    import numpy as np
    import pyedflib
except ImportError:
    sys.exit("need numpy and pyedflib:  pip install numpy pyedflib")

GOOD_ECG = {"clean", "usable", "recovered"}


# ---------------------------------------------------------------- EDF access

def edf_info(path):
    try:
        f = pyedflib.EdfReader(str(path))
    except OSError as exc:
        return {"error": str(exc).splitlines()[0][:120]}
    try:
        n = f.signals_in_file
        try:
            start = f.getStartdatetime()
        except Exception:
            start = None
        return {
            "labels": [f.getLabel(i) for i in range(n)],
            "fs": [float(f.getSampleFrequency(i)) for i in range(n)],
            "duration_s": float(f.getFileDuration()),
            "dims": [f.getPhysicalDimension(i) for i in range(n)],
            "pmin": [f.getPhysicalMinimum(i) for i in range(n)],
            "pmax": [f.getPhysicalMaximum(i) for i in range(n)],
            "start": start,
        }
    finally:
        f.close()


def to_uv(sig, dim):
    d = (dim or "").strip().lower().replace("µ", "u")
    return sig * {"mv": 1e3, "v": 1e6}.get(d, 1.0)


def read_window(path, idx, fs, t0, dur):
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
        if np.mean((sig >= pmax - tol) | (sig <= pmin + tol)) > 0.05:
            return "saturated", None, None, 0.0
    peak, mains = dominant_frequency(sig, fs)
    work, notched = sig, False
    if mains > 0.5:
        base = 60.0 if (peak is not None and abs(peak - 60) < 3) else float(MAINS_HZ)
        work, notched = notch_mains(sig, fs, base), True
    bpm, cv = estimate_bpm(work, fs)
    if bpm is None:
        return ("mains" if notched else "no_beats"), None, None, mains
    if not 40 <= bpm <= 180:
        return ("mains" if notched else "suspect"), bpm, cv, mains
    c = cv if cv is not None else 1.0
    if c <= 0.10:
        tier = "clean"
    elif c <= 0.30:
        tier = "usable"
    else:
        return ("mains" if notched else "poor"), bpm, cv, mains
    return ("recovered" if notched else tier), bpm, cv, mains


def eeg_window_quality(path, info, t0, dur):
    """
    Notch mains, then judge each 2-s window. A window is bad if any EEG
    channel is flat or (after linear detrend) exceeds EEG_MAX_P2P_UV.
    Returns (clean_fraction, notes).
    """
    notes, chans = [], []
    for i, label in enumerate(info["labels"]):
        if re.search(r"annotation", label, re.I):
            continue
        fs = info["fs"][i]
        sig = to_uv(read_window(path, i, fs, t0, dur), info["dims"][i])
        if sig.size == 0:
            return 0.0, [f"{label}: empty"]
        _, mains = dominant_frequency(sig, fs)
        if mains > 0.2:
            notes.append(f"{label}: {mains*100:.0f}% mains notched")
        chans.append((label, notch_mains(sig, fs, float(MAINS_HZ)), fs))

    n_win = int(dur // WIN_S)
    if n_win == 0 or not chans:
        return 0.0, notes + ["no windows"]
    good, bad_by = 0, Counter()
    for w in range(n_win):
        ok = True
        for label, sig, fs in chans:
            a, b = int(w * WIN_S * fs), int((w + 1) * WIN_S * fs)
            seg = sig[a:b]
            if seg.size < 2 or np.std(seg) == 0:
                ok = False
                bad_by[label] += 1
                continue
            t = np.arange(seg.size)
            seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
            if np.ptp(seg) > EEG_MAX_P2P_UV:
                ok = False
                bad_by[label] += 1
        good += ok
    frac = good / n_win
    if bad_by:
        notes.append("artifact windows: " + ", ".join(
            f"{k} {v}/{n_win}" for k, v in bad_by.most_common()))
    return frac, notes


# ---------------------------------------------------------------- metadata

def read_events(path):
    rows = list(csv.DictReader(path.read_text(errors="replace").splitlines(), delimiter="\t"))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def json_duration(path):
    if not path.exists():
        return None
    try:
        v = json.loads(path.read_text(errors="replace")).get("RecordingDuration")
    except json.JSONDecodeError:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def load_run(events_path):
    """Pass 1: everything about a run except signal-quality checks."""
    base = str(events_path)[: -len("_events.tsv")]
    name = Path(base).name
    eeg_edf, eeg_json = Path(base + "_eeg.edf"), Path(base + "_eeg.json")
    ecg_edf = events_path.parent.parent / "ecg" / (name + "_ecg.edf")

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
    dur = eeg["duration_s"] if eeg else ann_dur
    for label, val in (("ECG EDF", ecg["duration_s"] if ecg else None),
                       ("_eeg.json", json_duration(eeg_json)),
                       ("_events.tsv", ann_dur)):
        if dur is not None and val is not None and abs(val - dur) > DURATION_TOLERANCE_S:
            issues.append(f"duration mismatch: EEG {dur:.0f}s vs {label} {val:.0f}s")

    ecg_idx = None
    if ecg:
        ecg_idx = next((i for i, l in enumerate(ecg["labels"])
                        if re.search(r"E[CK]G", l, re.I)), 0)

    szs = sorted(({"onset": float(r["onset"]), "duration": float(r["duration"]),
                   "type": r.get("eventType", ""),
                   "localization": r.get("localization", "n/a"),
                   "lateralization": r.get("lateralization", "n/a"),
                   "vigilance": r.get("vigilance", "n/a")}
                  for r in rows if r.get("eventType", "").startswith("sz")),
                 key=lambda s: s["onset"])
    return {"run": re.search(r"run-(\d+)", name).group(1), "issues": issues,
            "eeg": eeg, "eeg_edf": eeg_edf, "ecg": ecg, "ecg_edf": ecg_edf,
            "ecg_idx": ecg_idx, "duration": dur, "seizures": szs,
            "start": eeg["start"] if eeg else None}


def timing_basis(runs):
    """Use EDF start times only if they are real (distinct, not a placeholder)."""
    starts = [r["start"] for r in runs]
    if not starts or any(s is None for s in starts):
        return False
    distinct = len(set(starts)) == len(starts)
    placeholder = all(s.year <= 2000 for s in starts)
    return distinct and not placeholder


# ---------------------------------------------------------------- main

def audit_subject(subj, root):
    events = sorted((root / subj).glob("ses-*/eeg/*_events.tsv"))
    print("=" * 74)
    print(f"SEIZEIT2 AUDIT — {subj}   ({len(events)} recordings)")
    print("=" * 74)
    if not events:
        print("  no recordings found (not downloaded?)")
        return [], None

    runs = [load_run(e) for e in events]
    use_clock = timing_basis(runs)
    first = next((r for r in runs if r["eeg"]), None)
    if first:
        print(f"  EEG channels : {', '.join(first['eeg']['labels'])}  @ "
              f"{sorted(set(first['eeg']['fs']))} Hz")
    if first and first["ecg"]:
        print(f"  ECG channel  : {first['ecg']['labels'][first['ecg_idx']]}  @ "
              f"{first['ecg']['fs'][first['ecg_idx']]} Hz")
    print("  run timing   : " + ("EDF start times (exact across runs)" if use_clock else
                               "anonymized — across runs, time since previous seizure "
                               "is a lower bound"))
    for r in runs:
        for msg in r["issues"]:
            print(f"  run-{r['run']}: {msg}")

    # -------- time since previous seizure, lead flag
    flat = []
    for r in runs:
        for s in r["seizures"]:
            abs_on = ((r["start"] - runs[0]["start"]).total_seconds() + s["onset"]
                      if use_clock else None)
            flat.append((r, s, abs_on))
    if use_clock:
        flat.sort(key=lambda x: x[2])

    prev_end_clock, prev_in_run = None, {}
    for r, s, abs_on in flat:
        if use_clock:
            if prev_end_clock is None:
                s["since_prev"], s["basis"] = None, "first seizure"
            else:
                s["since_prev"], s["basis"] = abs_on - prev_end_clock, "EDF clock"
            prev_end_clock = abs_on + s["duration"]
        else:
            pe = prev_in_run.get(r["run"])
            if pe is not None:
                s["since_prev"], s["basis"] = s["onset"] - pe, "same run"
            else:
                s["since_prev"], s["basis"] = s["onset"], ">= (from run start)"
            prev_in_run[r["run"]] = s["onset"] + s["duration"]

        sp, basis = s["since_prev"], s["basis"]
        if basis == "first seizure" or (sp is not None and sp >= LEAD_GAP_S):
            s["lead"] = "yes"        # exact, or a lower bound that already clears it
        elif basis.startswith(">="):
            s["lead"] = "unknown"    # an earlier run could hold a recent seizure
        else:
            s["lead"] = "no"

    # -------- quality checks
    out = []
    for r, s, _ in flat:
        dur, on = r["duration"], s["onset"]
        rec = {"subject": subj, "run": r["run"], "onset_s": on, "duration_s": s["duration"],
               "type": s["type"], "localization": s["localization"],
               "lateralization": s["lateralization"], "vigilance": s["vigilance"],
               "since_prev_s": "" if s["since_prev"] is None else f"{s['since_prev']:.0f}",
               "since_prev_basis": s["basis"], "lead_seizure": s["lead"],
               "in_recording": dur is None or on + s["duration"] <= dur + DURATION_TOLERANCE_S,
               "ecg_verdict": "n/a", "bpm": "", "rr_cv": "", "ecg_mains_pct": "",
               "eeg_clean_pct": "", "eeg_notes": ""}
        gap_ok = s["since_prev"] is None or s["since_prev"] >= MIN_GAP_FROM_PREVIOUS_S
        rec["preictal_ok"] = on >= PRE_ICTAL_S and gap_ok

        eeg_ok = False
        if rec["in_recording"] and rec["preictal_ok"]:
            t0 = on - PRE_ICTAL_S
            if r["ecg"] is not None:
                i = r["ecg_idx"]
                fs = r["ecg"]["fs"][i]
                v, bpm, cv, mains = classify_ecg(
                    read_window(r["ecg_edf"], i, fs, t0, PRE_ICTAL_S), fs,
                    r["ecg"]["pmin"][i], r["ecg"]["pmax"][i])
                rec.update(ecg_verdict=v, bpm=f"{bpm:.0f}" if bpm else "",
                           rr_cv=f"{cv:.2f}" if cv is not None else "",
                           ecg_mains_pct=f"{mains*100:.0f}")
            else:
                rec["ecg_verdict"] = "no_ecg"
            if r["eeg"] is not None:
                frac, notes = eeg_window_quality(r["eeg_edf"], r["eeg"], t0, PRE_ICTAL_S)
                rec["eeg_clean_pct"] = f"{frac*100:.0f}"
                rec["eeg_notes"] = "; ".join(notes)
                eeg_ok = frac >= MIN_CLEAN_FRACTION

        rec["usable"] = (rec["in_recording"] and rec["preictal_ok"]
                         and rec["ecg_verdict"] in GOOD_ECG and eeg_ok)
        rec["usable_lead"] = rec["usable"] and rec["lead_seizure"] == "yes"
        out.append(rec)

        if VERBOSE:
            if not rec["in_recording"]:
                why = "outside recording"
            elif not rec["preictal_ok"]:
                why = "no clean pre-ictal"
            else:
                bits = []
                if rec["ecg_verdict"] not in GOOD_ECG:
                    bits.append(f"ECG {rec['ecg_verdict']}")
                if not eeg_ok:
                    bits.append(f"EEG only {rec['eeg_clean_pct']}% clean")
                why = ", ".join(bits)
            tag = "USABLE" if rec["usable"] else "skip  "
            lead = {"yes": "lead", "no": "clust", "unknown": "lead?"}[rec["lead_seizure"]]
            hr = f"{rec['bpm']}bpm cv{rec['rr_cv']}" if rec["bpm"] else ""
            eegp = f"EEG {rec['eeg_clean_pct']}%" if rec["eeg_clean_pct"] else ""
            print(f"  {tag} run-{r['run']} @{on:>7.0f}s {lead:<5} ECG {rec['ecg_verdict']:<10}"
                  f"{hr:<14}{eegp:<9} {why}")

    hours = sum(r["duration"] or 0 for r in runs) / 3600
    u = sum(x["usable"] for x in out)
    ul = sum(x["usable_lead"] for x in out)
    print(f"  -> {len(out)} seizures, {u} usable ({ul} lead), {hours:.1f} h, "
          f"{sum(bool(r['issues']) for r in runs)} recording(s) with issues")
    return out, (len(out), u, ul, hours)


def main():
    root = Path(SEIZEIT2_ROOT).expanduser()
    if not root.is_dir():
        sys.exit(f"SEIZEIT2_ROOT is not a directory: {root}\nEdit CONFIG at the top of this file.")
    subjects = (sorted(p.name for p in root.glob("sub-*") if p.is_dir())
                if isinstance(SUBJECTS, str) and SUBJECTS.upper() == "ALL" else list(SUBJECTS))
    if not subjects:
        sys.exit("no sub-* folders found under SEIZEIT2_ROOT")

    all_rows, per = [], {}
    for subj in subjects:
        rows, stats = audit_subject(subj, root)
        all_rows.extend(rows)
        if stats:
            per[subj] = stats
    if not all_rows:
        sys.exit("no seizures found")

    fields = ["subject", "run", "onset_s", "duration_s", "type", "localization",
              "lateralization", "vigilance", "since_prev_s", "since_prev_basis",
              "lead_seizure", "in_recording", "preictal_ok", "ecg_verdict", "bpm",
              "rr_cv", "ecg_mains_pct", "eeg_clean_pct", "eeg_notes", "usable", "usable_lead"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in all_rows)

    checked = [r for r in all_rows if r["preictal_ok"] and r["in_recording"]]
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for subj, (n, u, ul, h) in per.items():
        print(f"  {subj}: {u:>3}/{n:<3} usable  ({ul:>3} lead)   {h:7.1f} h")
    print("-" * 74)
    tot = len(all_rows)
    usable = sum(r["usable"] for r in all_rows)
    lead = Counter(r["lead_seizure"] for r in all_rows)
    eeg_ok = sum(1 for r in checked if r["eeg_clean_pct"] and
                 float(r["eeg_clean_pct"]) >= MIN_CLEAN_FRACTION * 100)
    print(f"  seizures total             : {tot}")
    print(f"  clean pre-ictal window     : {len(checked)}")
    print(f"  pre-ictal ECG verdicts     : " + ", ".join(
        f"{k} {v}" for k, v in Counter(r['ecg_verdict'] for r in checked).most_common()))
    print(f"  pre-ictal EEG >= {MIN_CLEAN_FRACTION*100:.0f}% clean : {eeg_ok}/{len(checked)}")
    print(f"  lead / cluster / unknown   : {lead['yes']} / {lead['no']} / {lead['unknown']}"
          f"   (lead = {LEAD_GAP_S//60}+ min since previous seizure)")
    print(f"  USABLE (EEG + ECG)         : {usable}  ({100*usable/tot:.0f}%)")
    print(f"  USABLE and lead            : {sum(r['usable_lead'] for r in all_rows)}")
    print(f"  written                    : {OUT_CSV}")
    print("=" * 74)


if __name__ == "__main__":
    main()