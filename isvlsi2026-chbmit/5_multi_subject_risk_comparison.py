"""
Multi-Subject Risk Comparison Visualization
============================================

Creates a clean, publication-quality comparison of seizure prediction risk
curves across multiple subjects, aligned to seizure onset and normalized
for interpretability.

Author: Seizure Prediction Research
Date: 2026
"""

import os
import sys
import types
import re
import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, median_filter

# Fix EI deps
sys.modules["pyaudio"] = types.ModuleType("pyaudio")
from edge_impulse_linux.runner import ImpulseRunner

# ================= CONFIG =================

MODEL_PATH = (
    "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/"
    "EEG_Classification/ws_2/risk_modelling2/"
    "seizuredetection_windows-mac-arm64-v1-impulse-1.eim"
)

SUBJECTS = ["chb01", "chb02", "chb03", "chb05"]  # CHB01 included

DATASET_ROOT = (
    "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/"
    "EEG_Classification/MIT_Scalp_EEG_Dataset/physionet.org/files/chbmit/"
    "1.0.0"
)

CHANNELS = ["F3-C3", "C3-P3", "F4-C4", "C4-P4"]

FS = 256
WINDOW_SIZE = 512
STEP_SIZE = 128

FP_SHORT_WIN = 120
FP_LONG_WIN = 600

COOLDOWN = 300
MAX_LOOKBACK = 600

# Plot config
PRE_WINDOW = 600   # seconds before seizure
POST_WINDOW = 60   # seconds after seizure
COMMON_GRID_SIZE = 500

SUBJECT_COLORS = {
    "chb01": "#1f77b4",  # blue
    "chb02": "#ff7f0e",  # orange
    "chb03": "#2ca02c",  # green
    "chb05": "#d62728",  # red
}

# ================= HELPERS =================

def load_seizures(summary_file, target_file):
    """Extract seizure times from summary file."""
    try:
        with open(summary_file, "r", encoding="latin-1", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return []

    blocks = re.split(r"File Name:", text)

    for block in blocks:
        if target_file in block:
            starts = re.findall(r"Seizure Start Time:\s*(\d+)", block)
            ends = re.findall(r"Seizure End Time:\s*(\d+)", block)
            return [(int(s), int(e)) for s, e in zip(starts, ends)]
    return []


def firing_power(signal, win):
    """Compute rolling average (firing power)."""
    cs = np.concatenate([[0], np.cumsum(signal)])
    out = np.zeros(len(signal))
    for i in range(len(signal)):
        lo = max(0, i - win + 1)
        hi = i + 1
        out[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def compute_spectral(window):
    """Extract spectral features from signal window."""
    f, p = welch(window[:, 0], fs=FS, nperseg=256)
    delta = np.sum(p[(f >= 0.5) & (f < 4)])
    alpha = np.sum(p[(f >= 8) & (f < 13)])
    beta = np.sum(p[(f >= 13) & (f < 30)])
    ratio = (alpha + beta) / (delta + 1e-6)
    centroid = np.sum(f * p) / (np.sum(p) + 1e-6)
    return delta, ratio, centroid


def process_edf_file(edf_path, summary_file, runner):
    """Process single EDF file and compute risk scores."""
    seizures = load_seizures(summary_file, os.path.basename(edf_path))

    try:
        edf = pyedflib.EdfReader(edf_path)
        names = edf.getSignalLabels()
        idx = [names.index(ch) for ch in CHANNELS]
        signals = np.array([edf.readSignal(i) for i in idx]).T
        edf.close()
    except Exception as e:
        print(f"  ⚠ Error reading {edf_path}: {e}")
        return None

    times, probs = [], []
    delta_l, ratio_l, centroid_l = [], [], []

    for i in range(0, len(signals) - WINDOW_SIZE, STEP_SIZE):
        w = signals[i : i + WINDOW_SIZE]

        flat = w.T.flatten().astype(np.float32).tolist()
        res = runner.classify(flat)
        prob = res["result"]["classification"]["seizure"]

        d, r, c = compute_spectral(w)

        probs.append(prob)
        delta_l.append(d)
        ratio_l.append(r)
        centroid_l.append(c)

        times.append((i + WINDOW_SIZE // 2) / FS)

    df = pd.DataFrame(
        {
            "time": times,
            "prob": probs,
            "delta": delta_l,
            "ratio": ratio_l,
            "centroid": centroid_l,
        }
    )

    # ===== FEATURE ENGINEERING =====

    df["prob_sm"] = df["prob"].rolling(20, min_periods=1).mean()

    df["delta_sm"] = df["delta"].rolling(200, min_periods=50).median()
    df["ratio_sm"] = df["ratio"].rolling(200, min_periods=50).median()
    df["centroid_sm"] = df["centroid"].rolling(200, min_periods=50).median()

    df["delta_n"] = (df["delta_sm"] - df["delta_sm"].mean()) / (
        df["delta_sm"].std() + 1e-6
    )
    df["ratio_n"] = (df["ratio_sm"] - df["ratio_sm"].mean()) / (
        df["ratio_sm"].std() + 1e-6
    )
    df["centroid_n"] = (df["centroid_sm"] - df["centroid_sm"].mean()) / (
        df["centroid_sm"].std() + 1e-6
    )

    # ===== Risk Computation =====
    df["risk"] = (
        0.4 * df["prob_sm"]
        + 0.3 * df["ratio_n"].clip(0)
        + 0.2 * (-df["delta_n"]).clip(0)
        + 0.1 * df["centroid_n"].clip(0)
    ).clip(0, 1)

    # ===== TEMPORAL FEATURES =====

    df["fp_short"] = firing_power(df["risk"], FP_SHORT_WIN)
    df["fp_long"] = firing_power(df["risk"], FP_LONG_WIN)

    df["trend"] = df["risk"].diff().rolling(20).mean()

    df["mean"] = df["risk"].rolling(200).mean()
    df["std"] = df["risk"].rolling(200).std()

    df["z"] = (df["risk"] - df["mean"]) / (df["std"] + 1e-6)

    df["peakness"] = df["risk"] - df["risk"].rolling(100).mean()

    # ===== ALERT LOGIC =====

    alerts = []
    last_alert = -COOLDOWN

    for start, end in seizures:
        for i in range(len(df)):
            t = df["time"].iloc[i]

            if (start - t) > MAX_LOOKBACK:
                continue

            cond = (
                df["fp_short"].iloc[i] > 0.12
                and df["fp_long"].iloc[i] > 0.35
                and df["trend"].iloc[i] > 0.001
                and df["z"].iloc[i] > 1.2
            )

            if cond and (i - last_alert > COOLDOWN):
                alerts.append(i)
                last_alert = i

    df["alert"] = False
    df.loc[alerts, "alert"] = True

    return {
        "df": df,
        "seizures": seizures,
        "filename": os.path.basename(edf_path),
    }


def aggregate_subject_data(subject, runner):
    """Process all EDF files for a subject and aggregate results."""
    subject_path = os.path.join(DATASET_ROOT, subject)
    summary_file = os.path.join(subject_path, f"{subject}-summary.txt")

    if not os.path.exists(subject_path):
        print(f"⚠ Subject folder not found: {subject_path}")
        return None

    print(f"\n📁 Processing {subject}...")

    edf_files = sorted([f for f in os.listdir(subject_path) if f.endswith(".edf")])

    results = []
    for edf_file in edf_files:
        edf_path = os.path.join(subject_path, edf_file)
        print(f"  Processing {edf_file}...")

        result = process_edf_file(edf_path, summary_file, runner)
        if result:
            results.append(result)

    if not results:
        print(f"  ⚠ No valid results for {subject}")
        return None

    return results


def select_best_seizure(results):
    """
    Select the best representative seizure segment from all results.
    Prioritize segments with:
    - Detected alerts (most important for seizure prediction)
    - Early warning time (alerts that occur well before seizure)
    - Sufficient data quality
    
    Similar logic to seizure_prediction_v7.py
    """
    best_segment = None
    best_score = -np.inf

    for result in results:
        df = result["df"].copy()
        seizures = result["seizures"]

        if not seizures:
            continue

        for start, end in seizures:
            # Extract full window: 600s before to 60s after
            segment = df[(df["time"] >= start - PRE_WINDOW) & (df["time"] <= start + POST_WINDOW)].copy()

            # Check minimum segment length
            if len(segment) < 200:
                continue

            # Check that we have data both before AND after seizure
            before = segment[segment["time"] < start]
            after = segment[segment["time"] >= start]

            if len(before) < 100 or len(after) < 20:
                continue

            # PRIMARY CRITERION: Check for alerts in this segment
            alerts_in_segment = segment[segment["alert"]]
            
            if len(alerts_in_segment) == 0:
                # No alerts detected - skip this seizure
                continue
            
            # Calculate Early Warning Time (EWT) - time from first alert to seizure
            alert_times = alerts_in_segment["time"].values
            earliest_alert = np.min(alert_times)
            ewt = start - earliest_alert
            
            # Score based on EWT: longer warning time is better
            # Also consider number of alerts (more is better for reliability)
            num_alerts = len(alerts_in_segment)
            
            # Score combines:
            # 1. Early warning time (more time = better)
            # 2. Number of alerts (more alerts = more reliable)
            score = ewt + 0.1 * num_alerts
            
            # Require minimum warning time (at least 30 seconds)
            if ewt < 30:
                continue

            if score > best_score:
                best_score = score
                best_segment = {
                    "segment": segment,
                    "start": start,
                    "filename": result["filename"],
                    "score": ewt,  # Return EWT as the score
                    "num_alerts": num_alerts,
                }

    return best_segment


def normalize_risk(risk_values, method="minmax"):
    """
    Normalize risk values to use full [0, 1] range for better visualization.

    Args:
        risk_values: numpy array of risk scores
        method: 'minmax' (Option A) or 'clip' (Option B)

    Returns:
        Normalized risk values
    """
    risk_values = np.array(risk_values, dtype=np.float32)
    
    if method == "minmax":
        # Option A: Min-max normalization
        r_min = np.min(risk_values)
        r_max = np.max(risk_values)
        r_range = r_max - r_min
        
        if r_range < 1e-6:
            # If all values are the same, return mid-range
            return np.full_like(risk_values, 0.5)
        
        # Normalize to [0, 1]
        normalized = (risk_values - r_min) / r_range
        
    else:
        # Option B: Clip and stretch
        clipped = np.clip(risk_values, 0, 0.8)
        normalized = clipped / 0.8

    return np.clip(normalized, 0, 1)


def interpolate_to_grid(segment, start_time, method="minmax"):
    """
    Interpolate segment to common time grid and normalize.
    
    Aligns to seizure onset, not segment start.
    Applies multi-stage smoothing for clean pre-ictal patterns.

    Returns:
        (common_times, interpolated_risk, alert_times)
    """
    # Align to SEIZURE ONSET (not segment start!)
    aligned_time = segment["time"].values - start_time
    risk_values = segment["risk"].values
    alert_values = segment.get("alert", pd.Series([False] * len(segment))).values

    # Normalize for visualization
    risk_normalized = normalize_risk(risk_values, method=method)

    # Interpolate to common grid
    if len(aligned_time) < 2:
        return None

    f_interp = interp1d(
        aligned_time, risk_normalized, kind="cubic", fill_value="extrapolate", bounds_error=False
    )

    common_times = np.linspace(-PRE_WINDOW, POST_WINDOW, COMMON_GRID_SIZE)
    interpolated_risk = f_interp(common_times)
    
    # Stage 1: Strong Gaussian smoothing for overall trend
    interpolated_risk = gaussian_filter1d(interpolated_risk, sigma=8)
    
    # Stage 2: Apply rolling median to remove remaining spikes
    interpolated_risk = median_filter(interpolated_risk, size=5)
    
    # Stage 3: Final light smoothing
    interpolated_risk = gaussian_filter1d(interpolated_risk, sigma=2)
    
    interpolated_risk = np.clip(interpolated_risk, 0, 1)
    
    # Extract alert times (aligned to seizure onset)
    alert_indices = np.where(alert_values)[0]
    alert_times = aligned_time[alert_indices] if len(alert_indices) > 0 else np.array([])

    return common_times, interpolated_risk, alert_times


def plot_multi_subject_comparison(subject_segments):
    """
    Create publication-quality multi-subject comparison plot.

    Args:
        subject_segments: Dict of {subject: {"common_times": ..., "risk": ..., "alert_times": ..., "score": ...}}
    """
    # Sort by score (best first)
    sorted_subjects = sorted(
        subject_segments.items(), key=lambda x: x[1]["score"], reverse=True
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Add risk zones
    ax.axhspan(0, 0.3, color="green", alpha=0.1, label="Low Risk", zorder=0)
    ax.axhspan(0.3, 0.6, color="yellow", alpha=0.1, label="Medium Risk", zorder=0)
    ax.axhspan(0.6, 1.0, color="red", alpha=0.1, label="High Risk", zorder=0)

    # Add pre-ictal shaded region
    ax.axvspan(-PRE_WINDOW, 0, color="gray", alpha=0.05, zorder=0)

    # Plot subject curves with alerts
    for subject, data in sorted_subjects:
        common_times = data["common_times"]
        risk = data["risk"]
        alert_times = data["alert_times"]
        color = SUBJECT_COLORS.get(subject, "#000000")

        # Plot main risk curve
        ax.plot(
            common_times,
            risk,
            linewidth=2.5,
            label=subject.upper(),
            color=color,
            alpha=0.85,
            zorder=3,
        )
        
        # Add alert markers on top of the curve
        if len(alert_times) > 0:
            # Interpolate risk values at alert times
            f_risk = interp1d(
                common_times, risk, kind="cubic", fill_value="extrapolate", bounds_error=False
            )
            risk_at_alerts = f_risk(alert_times)
            
            ax.scatter(
                alert_times,
                risk_at_alerts,
                s=120,
                marker="*",
                color=color,
                edgecolors="darkred",
                label="Alerts" if subject == sorted_subjects[0][0] else None,  # Only label first time
                linewidth=1.5,
                zorder=5,
                alpha=0.9
            )

    # Seizure onset line
    ax.axvline(0, color="darkred", linestyle="--", linewidth=2, alpha=0.6, zorder=2, label="Seizure Onset")

    # Styling
    ax.set_xlabel("Time to Seizure (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Risk Score (Normalized)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Multi-Subject Risk Comparison (Prediction-Lite)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlim(-PRE_WINDOW, POST_WINDOW)
    ax.set_ylim(-0.05, 1.05)

    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
    )

    # Tight layout
    plt.tight_layout()

    return fig, ax


# ================= MAIN =================

def main():
    print("\n" + "=" * 70)
    print("MULTI-SUBJECT RISK COMPARISON")
    print("=" * 70)

    # Initialize model
    print("\n🔧 Initializing model...")
    runner = ImpulseRunner(MODEL_PATH)
    runner.init()

    # Process all subjects
    subject_segments = {}

    for subject in SUBJECTS:
        results = aggregate_subject_data(subject, runner)

        if not results:
            print(f"  ⚠ Skipping {subject} (no valid data)")
            continue

        # Select best seizure
        best = select_best_seizure(results)

        if best is None:
            print(f"  ⚠ No suitable seizure found for {subject}")
            continue

        # Extract segment and seizure onset time
        segment = best["segment"].copy()
        seizure_start = best["start"]
        
        # Interpolate to common grid with proper alignment
        result = interpolate_to_grid(segment, seizure_start)

        if result is None:
            print(f"  ⚠ Interpolation failed for {subject}")
            continue

        common_times, interpolated_risk, alert_times = result

        subject_segments[subject] = {
            "common_times": common_times,
            "risk": interpolated_risk,
            "alert_times": alert_times,
            "score": best["score"],
            "filename": best["filename"],
        }

        print(f"  ✓ {subject.upper()}: EWT={best['score']:.1f}s, detected_alerts={best['num_alerts']}, alerts_in_plot={len(alert_times)}, file={best['filename']}")

    runner.stop()

    if not subject_segments:
        print("\n❌ No valid subject data found.")
        return

    print(f"\n✓ Successfully processed {len(subject_segments)} subjects\n")

    # Create comparison plot
    print("📊 Creating multi-subject comparison plot...")
    fig, ax = plot_multi_subject_comparison(subject_segments)

    plt.show()

    # Save figure
    output_path = "multi_subject_risk_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Figure saved to {output_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
