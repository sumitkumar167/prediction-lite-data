#!/usr/bin/env python3
"""
Siena — is the EKG channel actually carrying data?

A flat trace in a viewer can mean either a dead channel or just a scaling
artifact. This reads the raw samples and decides:

  - header units and physical range vs. what the samples actually occupy
  - std, peak-to-peak, fraction of exactly-zero and of constant samples
  - a crude R-peak count -> plausible heart rate (40-180 bpm = real ECG)

Edit CONFIG, then:  python check_ekg.py

Requires: pyedflib   (pip install pyedflib)
Optional: numpy      (already a pyedflib dependency)
"""

import re
import sys
from pathlib import Path

# ==================== CONFIG — EDIT THESE ====================

SIENA_ROOT = r"/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/Scalp_EEG_Dataset/physionet.org/files/siena-scalp-eeg/1.0.0"

# Subjects to check: ["PN03"], ["PN00","PN03"], or "ALL"
SUBJECTS = "ALL"



# How many seconds to sample from the middle of each recording.
# Keep modest — the whole point is a quick verdict, not a full load.
SAMPLE_SECONDS = 120

# Also report on one EEG channel, as a working reference to compare against.
REFERENCE_EEG = "F3"

# =============== END CONFIG — no need to edit below ===============

try:
    import numpy as np
    import pyedflib
except ImportError:
    sys.exit("need pyedflib and numpy.  pip install pyedflib")


def normalise(name):
    n = name.upper()
    n = re.sub(r"\bEEG\b", "", n)
    n = re.sub(r"-?REF\b", "", n)
    return re.sub(r"[^A-Z0-9]", "", n).strip()


def pick_channels(labels):
    """Return {role: index} for the cardiac channel and a reference EEG channel."""
    out = {}
    for i, lab in enumerate(labels):
        if "cardiac" not in out and re.search(r"E[KC]G", lab, flags=re.IGNORECASE):
            out["cardiac"] = i
        if "eeg_ref" not in out and normalise(lab) == REFERENCE_EEG.upper():
            out["eeg_ref"] = i
    return out


def rail_fraction(sig, phys_min, phys_max):
    """Share of samples sitting at (or within 1%% of) the header's physical limits."""
    if phys_max <= phys_min:
        return 0.0
    span = phys_max - phys_min
    tol = 0.01 * span
    at_top = np.sum(sig >= phys_max - tol)
    at_bot = np.sum(sig <= phys_min + tol)
    return float((at_top + at_bot) / sig.size)


def dominant_frequency(sig, fs):
    """
    Peak of the amplitude spectrum above 1 Hz, plus the share of total power
    sitting near 50/60 Hz mains. Numpy FFT only, no scipy.
    """
    x = np.asarray(sig, dtype=float)
    x = x - np.mean(x)
    n = min(x.size, int(fs * 30))          # up to 30 s is plenty
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
            sel = (freqs >= f0 - 2) & (freqs <= f0 + 2)
            mains = max(mains, float(np.sum(spec[sel] ** 2) / total))
    return peak, mains


def _fir_bandpass(sig, fs, lo=5.0, hi=15.0, numtaps=201):
    """
    Windowed-sinc bandpass, numpy only (no scipy — some macOS wheels are broken).
    5-15 Hz is the standard Pan-Tompkins band for emphasising QRS complexes.
    """
    if numtaps % 2 == 0:
        numtaps += 1
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    nyq = fs / 2.0
    lo = max(lo, 0.1) / nyq
    hi = min(hi, nyq * 0.95) / nyq

    def sinc_lp(cut):
        h = np.sinc(cut * n) * cut
        return h * np.hamming(numtaps)

    h = sinc_lp(hi) - sinc_lp(lo)
    h -= h.mean()
    if sig.size < numtaps * 2:
        return sig - np.median(sig)

    # Remove the DC offset BEFORE convolving. Siena's EKG rides several mV
    # from zero; zero-padded convolution on that turns the signal edges into
    # a transient far larger than any R-peak, which wrecks thresholding.
    centred = sig - np.median(sig)
    return np.convolve(centred, h, mode="same")


def quantisation_step(sig):
    """Smallest non-zero difference between consecutive samples (ADC step)."""
    d = np.abs(np.diff(np.unique(sig)))
    d = d[d > 0]
    return float(d.min()) if d.size else 0.0


def effective_rate(sig, fs):
    """
    If the channel was sampled slower and stored by sample-and-hold, a large
    share of consecutive samples are identical. Estimate the real rate.
    """
    if sig.size < 2:
        return fs
    repeats = float(np.mean(np.diff(sig) == 0))
    return fs * (1.0 - repeats) if repeats > 0.2 else fs


def estimate_bpm(sig, fs):
    """
    Pan-Tompkins style R-peak detection:
      bandpass 5-15 Hz -> differentiate -> square -> moving-window integrate
      -> adaptive threshold with a 300 ms refractory period.
    Robust to the large DC offset and baseline wander seen in Siena's EKG.
    Returns (bpm, rr_cv) or (None, None).
    """
    if sig.size < int(fs * 5):
        return None, None

    filt = _fir_bandpass(np.asarray(sig, dtype=float), fs)

    # Discard filter edge transients before thresholding.
    trim = min(250, filt.size // 10)
    if trim > 0 and filt.size > 3 * trim:
        filt = filt[trim:-trim]

    deriv = np.diff(filt, prepend=filt[0])
    squared = deriv ** 2

    win = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    sd = np.std(integrated)
    if sd == 0:
        return None, None

    # Adaptive: try a few thresholds, keep the first giving a plausible rate.
    minutes = filt.size / fs / 60.0
    min_gap = int(0.3 * fs)
    best_bpm = None
    for k in (1.0, 0.7, 0.5, 0.3, 1.5, 2.0):
        thresh = np.mean(integrated) + k * sd
        above = integrated > thresh
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
        best_bpm = bpm if best_bpm is None else best_bpm
        rr = np.diff(picks) / fs
        rr_cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else None
        if 40 <= bpm <= 180:
            return bpm, rr_cv
    return best_bpm, None


def describe(name, sig, fs, units, phys_min, phys_max, is_cardiac):
    print(f"\n  channel {name!r}")
    print(f"    header       : units={units!r}  physical range "
          f"{phys_min:g} .. {phys_max:g}")
    if sig.size == 0:
        print("    samples      : NONE READ")
        return "empty"

    sd = float(np.std(sig))
    p2p = float(np.ptp(sig))
    zeros = float(np.mean(sig == 0))
    const = float(np.mean(np.diff(sig) == 0)) if sig.size > 1 else 1.0

    print(f"    samples      : n={sig.size}  min={sig.min():.4g}  "
          f"max={sig.max():.4g}")
    print(f"    variation    : std={sd:.4g}  peak-to-peak={p2p:.4g}")
    print(f"    flatness     : {zeros*100:.1f}% exactly zero, "
          f"{const*100:.1f}% no change from previous sample")

    if sd == 0 or p2p == 0:
        print("    VERDICT      : DEAD — constant signal, no information")
        return "dead"
    if zeros > 0.98:
        print("    VERDICT      : DEAD — almost entirely zeros")
        return "dead"

    if is_cardiac:
        offset = float(np.mean(sig))
        step = quantisation_step(sig)
        eff_fs = effective_rate(sig, fs)
        print(f"    DC offset    : {offset:.4g} {units} "
              f"(signal rides this far from zero)")
        if step:
            levels = p2p / step if step else 0
            print(f"    resolution   : {step:.4g} {units}/bit -> only "
                  f"~{levels:.0f} levels across the waveform"
                  f"{'  [COARSE]' if levels < 100 else ''}")
        if eff_fs < fs * 0.9:
            print(f"    effective fs : ~{eff_fs:.0f} Hz (stored at {fs:g} Hz, "
                  f"likely sample-and-hold from a slower channel)")

        rails = rail_fraction(sig, phys_min, phys_max)
        peak_hz, mains = dominant_frequency(sig, fs)
        if rails > 0.01:
            print(f"    saturation   : {rails*100:.1f}% of samples at the physical "
                  f"limits  [RAILING]")
        if peak_hz is not None:
            print(f"    spectrum     : dominant {peak_hz:.1f} Hz, "
                  f"{mains*100:.0f}% of power near mains (50/60 Hz)")

        if rails > 0.05:
            print("    VERDICT      : SATURATED — channel is railing, not usable "
                  "for R-peak detection")
            return "saturated"
        if mains > 0.5:
            print("    VERDICT      : MAINS-DOMINATED — >50% of power at 50/60 Hz; "
                  "notch-filter and re-check, likely unusable")
            return "mains"

        bpm, rr_cv = estimate_bpm(sig, fs)
        if bpm is None:
            print("    VERDICT      : SUSPECT — no R-peaks found even after "
                  "bandpass filtering")
            return "suspect"
        print(f"    R-peak est.  : {bpm:.0f} bpm"
              + (f"  (RR variability CV={rr_cv:.2f})" if rr_cv is not None else ""))
        if 40 <= bpm <= 180:
            cv = rr_cv if rr_cv is not None else 1.0
            if cv <= 0.10:
                print("    VERDICT      : CLEAN ECG — plausible rate, regular RR")
                return "ok"
            if cv <= 0.30:
                print("    VERDICT      : USABLE — plausible rate, some missed/extra "
                      "beats (RR CV above resting range); clean before HRV")
                return "usable"
            print("    VERDICT      : POOR DETECTION — RR intervals very irregular; "
                  "inspect visually before using")
            return "check"
        print("    VERDICT      : SUSPECT — rate outside physiological range")
        return "suspect"

    print("    VERDICT      : has signal")
    return "ok"


def check_file(path):
    print(f"\n--- {path.name} ---")
    try:
        f = pyedflib.EdfReader(str(path))
    except OSError as exc:
        msg = str(exc)
        if "Filesize" in msg or "filesize" in msg:
            print("  INCOMPLETE — file is smaller than its header says; "
                  "the download probably hasn't finished")
            return "incomplete"
        print(f"  UNREADABLE — {msg.splitlines()[0][:140]}")
        return "unreadable"
    try:
        labels = f.getSignalLabels()
        roles = pick_channels(labels)

        if "cardiac" not in roles:
            print("  no cardiac channel in this file")
            return None

        verdicts = {}
        for role in ("cardiac", "eeg_ref"):
            if role not in roles:
                continue
            i = roles[role]
            fs = float(f.getSampleFrequency(i))
            total = f.getNSamples()[i]
            n = min(int(SAMPLE_SECONDS * fs), total)
            start = max(0, (total - n) // 2)  # sample from the middle
            sig = f.readSignal(i, start=start, n=n)
            verdicts[role] = describe(
                labels[i], np.asarray(sig, dtype=float), fs,
                f.getPhysicalDimension(i), f.getPhysicalMinimum(i),
                f.getPhysicalMaximum(i), is_cardiac=(role == "cardiac"))

        if "eeg_ref" in verdicts:
            print(f"\n    (compare: cardiac={verdicts['cardiac']}, "
                  f"{REFERENCE_EEG}={verdicts['eeg_ref']})")
        return verdicts.get("cardiac")
    finally:
        f.close()


def main():
    root = Path(SIENA_ROOT).expanduser()
    if not root.is_dir():
        sys.exit(f"SIENA_ROOT is not a directory: {root}\n"
                 f"Edit the CONFIG block at the top of this file.")

    if isinstance(SUBJECTS, str) and SUBJECTS.upper() == "ALL":
        folders = sorted(p for p in root.iterdir()
                         if p.is_dir() and re.fullmatch(r"PN\d+", p.name, re.I))
    else:
        folders = [root / s for s in SUBJECTS if (root / s).is_dir()]

    if not folders:
        sys.exit("no subject folders found — check SIENA_ROOT and SUBJECTS.")

    tally = {}
    for folder in folders:
        print("=" * 62)
        print(f"EKG CHECK — {folder.name}")
        print("=" * 62)
        for edf in sorted(folder.glob("*.edf")):
            v = check_file(edf)
            if v:
                tally.setdefault(folder.name, []).append((edf.name, v))
        print()

    print("=" * 62)
    print("SUMMARY — cardiac channel usability")
    print("=" * 62)
    label = {"ok": "clean", "usable": "usable", "check": "poor detection",
             "suspect": "non-physiological", "saturated": "saturated",
             "mains": "mains-dominated", "dead": "dead", "empty": "empty",
             "incomplete": "INCOMPLETE DOWNLOAD", "unreadable": "unreadable"}
    subj_clean, subj_any = [], []
    for subj, rows in tally.items():
        clean = sum(1 for _, v in rows if v == "ok")
        usable = sum(1 for _, v in rows if v == "usable")
        print(f"  {subj}: {clean} clean + {usable} usable of {len(rows)} file(s)")
        for fname, v in rows:
            if v not in ("ok",):
                print(f"      {fname}: {label.get(v, v)}")
        if clean:
            subj_clean.append(subj)
        if clean or usable:
            subj_any.append(subj)
    print("-" * 62)
    print(f"  subjects with >=1 CLEAN file  : {len(subj_clean)}/{len(tally)}  "
          f"({', '.join(subj_clean) or 'none'})")
    print(f"  subjects with >=1 usable file : {len(subj_any)}/{len(tally)}  "
          f"({', '.join(subj_any) or 'none'})")
    print("  NOTE: verdicts describe only the sampled middle segment, not the "
          "periods around seizures.")
    print("=" * 62)


if __name__ == "__main__":
    main()