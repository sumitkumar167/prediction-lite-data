#!/usr/bin/env python3
"""
SeizeIT2 — extract training segments into a local cache (read the raw EDFs once).

For every seizure the audit marked usable:
  - PRE-ICTAL segment : the PRE_ICTAL_S seconds before onset
  - ICTAL segment     : onset -> onset + duration           (optional)
For each subject, matched INTERICTAL segments of the same length:
  - far from every annotated seizure in that run (SEIZURE_GAP_S)
  - away from the run's start and end (EDGE_GAP_S), since runs are
    discontinuous and a neighbouring run may hold an unseen seizure
  - passing the same ECG + EEG quality checks as the pre-ictal windows
  - chosen with a FIXED random seed, so the selection is reproducible

Each segment is saved as one compressed .npz holding the raw EEG (µV), raw ECG
(µV), sampling rates, and a per-window quality mask for 2-s windows at 0.5-s
stride. Windows themselves are NOT stored: they overlap by 75%, so the feature
step slices them from the segment instead.

Outputs
  - signal arrays : CACHE_DIR (outside the repo — can be large)
  - segment index : seizeit2_segments.csv next to this script (small, commit it)

Edit CONFIG, then:  python 4_extract_segments.py
Requires: numpy, pyedflib — no SciPy.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

# Dataset (outside the repo) and where to put the extracted signal arrays.
SEIZEIT2_ROOT = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/SiezeIT2_ws/SeizeIT2"
CACHE_DIR = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/SiezeIT2_ws/seizeit2_cache"

# Small files next to this script (inside the repo).
SCRIPT_DIR = Path(__file__).resolve().parent
AUDIT_CSV = SCRIPT_DIR / "seizeit2_audit_seizures.csv"     # from 3_dataset_audit.py
INDEX_CSV = SCRIPT_DIR / "seizeit2_segments.csv"            # written by this script

SELECT = "usable"               # "usable" = all usable seizures, "usable_lead" = lead only
INCLUDE_ICTAL = True            # also save the seizure itself (for the detector)

PRE_ICTAL_S = 240               # must match the audit
WIN_S = 2.0                     # Prediction-Lite window
STRIDE_S = 0.5                  # Prediction-Lite stride

INTERICTAL_PER_PREICTAL = 1     # 1 = balanced
SEIZURE_GAP_S = 1800            # interictal must be this far from any seizure in the run
EDGE_GAP_S = 300                # ... and this far from the run's start and end
CANDIDATE_STEP_S = 60           # spacing of candidate interictal start times
SEED = 42                       # fixed: interictal selection is reproducible

EEG_MAX_P2P_UV = 1000           # window rejected above this (after notch + detrend)
MIN_CLEAN_FRACTION = 0.75       # interictal candidates need this share of clean windows
MAINS_HZ = 50

OVERWRITE = False               # False = reuse segments already in the cache (resumable)

# =============== END CONFIG — no need to edit below ===============

try:
    import numpy as np
    import pyedflib
except ImportError:
    sys.exit("need numpy and pyedflib:  pip install numpy pyedflib")

GOOD_ECG = {"clean", "usable", "recovered"}


# ---------------------------------------------------------------- EDF helpers

def to_uv(sig, dim):
    d = (dim or "").strip().lower().replace("µ", "u")
    return sig * {"mv": 1e3, "v": 1e6}.get(d, 1.0)


def channel_map(eeg_edf, ecg_edf):
    """Find BTE / CROSS EEG channels and the ECG channel, with fs and units."""
    f = pyedflib.EdfReader(str(eeg_edf))
    try:
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        bte = next((i for i, l in enumerate(labels) if re.search(r"BTE", l, re.I)), None)
        cross = next((i for i, l in enumerate(labels) if re.search(r"CROSS", l, re.I)), None)
        if bte is None or cross is None:
            eeg_idx = [i for i, l in enumerate(labels) if not re.search(r"annotation", l, re.I)][:2]
            if len(eeg_idx) < 2:
                return None
            bte, cross = eeg_idx
        side = "right" if re.search(r"right", labels[bte], re.I) else (
            "left" if re.search(r"left", labels[bte], re.I) else "n/a")
        eeg = {"idx": [bte, cross], "labels": [labels[bte], labels[cross]],
               "fs": float(f.getSampleFrequency(bte)),
               "dims": [f.getPhysicalDimension(bte), f.getPhysicalDimension(cross)],
               "dur": float(f.getFileDuration()), "side": side}
    finally:
        f.close()

    g = pyedflib.EdfReader(str(ecg_edf))
    try:
        labels = [g.getLabel(i) for i in range(g.signals_in_file)]
        i = next((k for k, l in enumerate(labels) if re.search(r"E[CK]G", l, re.I)), 0)
        ecg = {"idx": i, "label": labels[i], "fs": float(g.getSampleFrequency(i)),
               "dim": g.getPhysicalDimension(i),
               "pmin": g.getPhysicalMinimum(i), "pmax": g.getPhysicalMaximum(i)}
    finally:
        g.close()
    return eeg, ecg


def read(path, idx, fs, t0, dur):
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
        X[(freqs >= base_hz * h - width_hz) & (freqs <= base_hz * h + width_hz)] = 0
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
    if sig.size == 0:
        return "empty", None, None
    if np.std(sig) == 0 or np.ptp(sig) == 0 or np.mean(sig == 0) > 0.98:
        return "dead", None, None
    span = pmax - pmin
    if span > 0:
        tol = 0.01 * span
        if np.mean((sig >= pmax - tol) | (sig <= pmin + tol)) > 0.05:
            return "saturated", None, None
    peak, mains = dominant_frequency(sig, fs)
    work, notched = sig, False
    if mains > 0.5:
        base = 60.0 if (peak is not None and abs(peak - 60) < 3) else float(MAINS_HZ)
        work, notched = notch_mains(sig, fs, base), True
    bpm, cv = estimate_bpm(work, fs)
    if bpm is None or not 40 <= bpm <= 180:
        return ("mains" if notched else "suspect"), bpm, cv
    c = cv if cv is not None else 1.0
    if c > 0.30:
        return ("mains" if notched else "poor"), bpm, cv
    if notched:
        return "recovered", bpm, cv
    return ("clean" if c <= 0.10 else "usable"), bpm, cv


def window_mask(eeg, fs):
    """eeg: (2, N) raw µV. Notch, then flag each WIN_S window at STRIDE_S stride."""
    w, s = int(round(WIN_S * fs)), int(round(STRIDE_S * fs))
    n = eeg.shape[1]
    if n < w:
        return np.zeros(0, dtype=bool)
    clean = np.vstack([notch_mains(ch, fs, float(MAINS_HZ)) for ch in eeg])
    n_win = (n - w) // s + 1
    t = np.arange(w)
    ok = np.ones(n_win, dtype=bool)
    for k in range(n_win):
        a = k * s
        for ch in clean:
            seg = ch[a:a + w]
            if np.std(seg) == 0:
                ok[k] = False
                break
            seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
            if np.ptp(seg) > EEG_MAX_P2P_UV:
                ok[k] = False
                break
    return ok


# ---------------------------------------------------------------- run metadata

def load_runs(subject, root):
    """All runs of a subject: paths, duration, every annotated seizure, ECG presence."""
    runs = {}
    for ev in sorted((root / subject).glob("ses-*/eeg/*_events.tsv")):
        base = str(ev)[: -len("_events.tsv")]
        run = re.search(r"run-(\d+)", ev.name).group(1)
        ecg = ev.parent.parent / "ecg" / (Path(base).name + "_ecg.edf")
        rows = list(csv.DictReader(ev.read_text(errors="replace").splitlines(), delimiter="\t"))
        szs = [(float(r["onset"]), float(r["duration"])) for r in rows
               if (r.get("eventType") or "").startswith("sz")]
        runs[run] = {"eeg_edf": Path(base + "_eeg.edf"), "ecg_edf": ecg,
                     "has_ecg": ecg.exists(), "seizures": sorted(szs), "map": None}
    return runs


def get_map(r):
    if r["map"] is None:
        r["map"] = channel_map(r["eeg_edf"], r["ecg_edf"])
    return r["map"]


def extract(r, t0, dur):
    eeg_m, ecg_m = get_map(r)
    eeg = np.vstack([to_uv(read(r["eeg_edf"], i, eeg_m["fs"], t0, dur), d)
                     for i, d in zip(eeg_m["idx"], eeg_m["dims"])])
    ecg = to_uv(read(r["ecg_edf"], ecg_m["idx"], ecg_m["fs"], t0, dur), ecg_m["dim"])
    return eeg, ecg


def save(path, eeg, ecg, mask, eeg_m, ecg_m):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, eeg=eeg.astype(np.float32), ecg=ecg.astype(np.float32),
        fs_eeg=eeg_m["fs"], fs_ecg=ecg_m["fs"], window_ok=mask,
        win_s=WIN_S, stride_s=STRIDE_S,
        eeg_labels=np.array(eeg_m["labels"]), ecg_label=np.array(ecg_m["label"]))


def interictal_candidates(r, dur):
    """Start times whose [t, t+dur] avoids seizures (±SEIZURE_GAP_S) and run edges."""
    eeg_m, _ = get_map(r)
    lo, hi = EDGE_GAP_S, eeg_m["dur"] - EDGE_GAP_S - dur
    out, t = [], lo
    while t <= hi:
        clear = all(t + dur <= on - SEIZURE_GAP_S or t >= on + d + SEIZURE_GAP_S
                    for on, d in r["seizures"])
        if clear:
            out.append(t)
        t += CANDIDATE_STEP_S
    return out


# ---------------------------------------------------------------- main

def main():
    root, cache = Path(SEIZEIT2_ROOT).expanduser(), Path(CACHE_DIR).expanduser()
    if not root.is_dir():
        sys.exit(f"SEIZEIT2_ROOT is not a directory: {root}")
    if not AUDIT_CSV.exists():
        sys.exit(f"audit CSV not found: {AUDIT_CSV}\nRun 3_dataset_audit.py first.")
    cache.mkdir(parents=True, exist_ok=True)

    audit = [r for r in csv.DictReader(open(AUDIT_CSV)) if r.get(SELECT) == "True"]
    if not audit:
        sys.exit(f"no seizures with {SELECT} == True in {AUDIT_CSV.name}")
    by_subj = defaultdict(list)
    for r in audit:
        by_subj[r["subject"]].append(r)

    rng = np.random.default_rng(SEED)
    index, shortfall = [], {}

    for subj in sorted(by_subj):
        runs = load_runs(subj, root)
        print("=" * 72)
        print(f"{subj}: {len(by_subj[subj])} seizures selected ({SELECT})")
        print("=" * 72)

        def emit(kind, run, t0, dur, meta, ecg_info=None):
            r = runs[run]
            eeg_m, ecg_m = get_map(r)
            sid = f"{subj}_run-{run}_{kind}_{int(round(t0))}"
            path = cache / subj / f"{sid}.npz"
            if path.exists() and not OVERWRITE:
                mask = np.load(path)["window_ok"]
            else:
                eeg, ecg = extract(r, t0, dur)
                mask = window_mask(eeg, eeg_m["fs"])
                save(path, eeg, ecg, mask, eeg_m, ecg_m)
            ecg_info = ecg_info or {}
            row = {"segment_id": sid, "subject": subj, "run": run, "kind": kind,
                   "t_start_s": f"{t0:.1f}", "duration_s": f"{dur:.1f}",
                   "seizure_onset_s": meta.get("onset_s", ""),
                   "lead_seizure": meta.get("lead_seizure", ""),
                   "vigilance": meta.get("vigilance", ""),
                   "localization": meta.get("localization", ""),
                   "lateralization": meta.get("lateralization", ""),
                   "bte_side": eeg_m["side"],
                   "n_windows": int(mask.size), "n_windows_ok": int(mask.sum()),
                   "ecg_verdict": ecg_info.get("verdict", meta.get("ecg_verdict", "")),
                   "bpm": ecg_info.get("bpm", meta.get("bpm", "")),
                   "rr_cv": ecg_info.get("rr_cv", meta.get("rr_cv", "")),
                   "file": str(path.relative_to(cache))}
            index.append(row)
            return row

        # ---- pre-ictal + ictal
        n_pre = 0
        for s in sorted(by_subj[subj], key=lambda x: (x["run"], float(x["onset_s"]))):
            run, on, d = s["run"], float(s["onset_s"]), float(s["duration_s"])
            if run not in runs:
                print(f"  run-{run}: files not found, skipped")
                continue
            row = emit("preictal", run, on - PRE_ICTAL_S, PRE_ICTAL_S, s)
            n_pre += 1
            line = f"  pre-ictal  run-{run} @{on:>7.0f}s  {row['n_windows_ok']}/{row['n_windows']} windows ok"
            if INCLUDE_ICTAL:
                ri = emit("ictal", run, on, d, s)
                line += f"   ictal {d:>4.0f}s {ri['n_windows_ok']}/{ri['n_windows']}"
            print(line)

        # ---- matched interictal
        need = int(round(n_pre * INTERICTAL_PER_PREICTAL))
        cands = [(run, t) for run, r in runs.items() if r["has_ecg"]
                 for t in interictal_candidates(r, PRE_ICTAL_S)]
        order = rng.permutation(len(cands)) if cands else []
        chosen, tried, rejected = [], 0, Counter()
        for k in order:
            if len(chosen) >= need:
                break
            run, t0 = cands[k]
            if any(run == cr and abs(t0 - ct) < PRE_ICTAL_S for cr, ct in chosen):
                continue
            tried += 1
            r = runs[run]
            eeg_m, ecg_m = get_map(r)
            ecg = to_uv(read(r["ecg_edf"], ecg_m["idx"], ecg_m["fs"], t0, PRE_ICTAL_S), ecg_m["dim"])
            v, bpm, cv = classify_ecg(ecg, ecg_m["fs"], ecg_m["pmin"], ecg_m["pmax"])
            if v not in GOOD_ECG:
                rejected[f"ECG {v}"] += 1
                continue
            eeg, _ = extract(r, t0, PRE_ICTAL_S)
            if window_mask(eeg, eeg_m["fs"]).mean() < MIN_CLEAN_FRACTION:
                rejected["EEG artifact"] += 1
                continue
            emit("interictal", run, t0, PRE_ICTAL_S, {},
                 {"verdict": v, "bpm": f"{bpm:.0f}" if bpm else "",
                  "rr_cv": f"{cv:.2f}" if cv is not None else ""})
            chosen.append((run, t0))

        got = len(chosen)
        print(f"  interictal: {got}/{need} found from {len(cands)} candidate positions "
              f"({tried} checked" + (", rejected: " + ", ".join(
                  f"{k} {v}" for k, v in rejected.most_common()) if rejected else "") + ")")
        if got < need:
            shortfall[subj] = (got, need)

    # ---- index + summary
    fields = list(index[0].keys())
    with open(INDEX_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(index)

    size = sum(p.stat().st_size for p in cache.rglob("*.npz"))
    kinds = Counter(r["kind"] for r in index)
    win = Counter()
    for r in index:
        win[r["kind"] + "_ok"] += r["n_windows_ok"]
        win[r["kind"]] += r["n_windows"]
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k in ("preictal", "ictal", "interictal"):
        if kinds[k]:
            print(f"  {k:<11}: {kinds[k]:>4} segments, {win[k+'_ok']:>6}/{win[k]:<6} windows ok")
    if shortfall:
        print("  interictal shortfall (not enough clean seizure-free data):")
        for s, (g, n) in shortfall.items():
            print(f"    {s}: {g}/{n}")
    print(f"  cache size : {size/1e6:.1f} MB in {cache}")
    print(f"  index      : {INDEX_CSV}")
    print(f"  seed       : {SEED}  (re-running reproduces the same interictal picks)")
    print("=" * 72)


if __name__ == "__main__":
    main()