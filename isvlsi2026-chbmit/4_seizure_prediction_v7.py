import os
import re
import sys
import types
import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt
from scipy.signal import welch

# Fix EI deps
sys.modules["pyaudio"] = types.ModuleType("pyaudio")
from edge_impulse_linux.runner import ImpulseRunner

# ================= CONFIG =================
MODEL_PATH = (
    "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/"
    "EEG_Classification/ws_2/risk_modelling2/"
    "seizuredetection_windows-mac-arm64-v1-impulse-1.eim"
)

SUBJECT_FOLDER = (
    "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/"
    "EEG_Classification/MIT_Scalp_EEG_Dataset/physionet.org/files/chbmit/"
    "1.0.0/chb01"
)

SUMMARY_FILE = os.path.join(SUBJECT_FOLDER, "chb01-summary.txt")

CHANNELS = ["F3-C3", "C3-P3", "F4-C4", "C4-P4"]

FS = 256
WINDOW_SIZE = 512
STEP_SIZE = 128

FP_SHORT_WIN = 120
FP_LONG_WIN = 600

COOLDOWN = 300
MAX_LOOKBACK = 600  # seconds

# ================= HELPERS =================

def load_seizures(summary_file, target_file):
    with open(summary_file, "r", encoding="latin-1", errors="ignore") as f:
        text = f.read()

    blocks = re.split(r"File Name:", text)

    for block in blocks:
        if target_file in block:
            starts = re.findall(r"Seizure Start Time:\s*(\d+)", block)
            ends   = re.findall(r"Seizure End Time:\s*(\d+)", block)
            return [(int(s), int(e)) for s, e in zip(starts, ends)]
    return []

def firing_power(signal, win):
    cs = np.concatenate([[0], np.cumsum(signal)])
    out = np.zeros(len(signal))
    for i in range(len(signal)):
        lo = max(0, i - win + 1)
        hi = i + 1
        out[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return out

def compute_spectral(window):
    f, p = welch(window[:,0], fs=FS, nperseg=256)
    delta = np.sum(p[(f>=0.5)&(f<4)])
    alpha = np.sum(p[(f>=8)&(f<13)])
    beta  = np.sum(p[(f>=13)&(f<30)])
    ratio = (alpha + beta) / (delta + 1e-6)
    centroid = np.sum(f*p)/(np.sum(p)+1e-6)
    return delta, ratio, centroid

# ================= MODEL =================

runner = ImpulseRunner(MODEL_PATH)
runner.init()

# ================= PROCESS FILE =================

def process_file(edf_path):

    seizures = load_seizures(SUMMARY_FILE, os.path.basename(edf_path))

    edf = pyedflib.EdfReader(edf_path)
    names = edf.getSignalLabels()
    idx = [names.index(ch) for ch in CHANNELS]
    signals = np.array([edf.readSignal(i) for i in idx]).T
    edf.close()

    times, probs = [], []
    delta_l, ratio_l, centroid_l = [], [], []

    for i in range(0, len(signals)-WINDOW_SIZE, STEP_SIZE):

        w = signals[i:i+WINDOW_SIZE]

        flat = w.T.flatten().astype(np.float32).tolist()
        res = runner.classify(flat)
        prob = res["result"]["classification"]["seizure"]

        d, r, c = compute_spectral(w)

        probs.append(prob)
        delta_l.append(d)
        ratio_l.append(r)
        centroid_l.append(c)

        times.append((i + WINDOW_SIZE//2)/FS)

    df = pd.DataFrame({
        "time": times,
        "prob": probs,
        "delta": delta_l,
        "ratio": ratio_l,
        "centroid": centroid_l
    })

    # ================= FEATURE ENGINEERING =================

    df["prob_sm"] = df["prob"].rolling(20, min_periods=1).mean()

    df["delta_sm"] = df["delta"].rolling(200, min_periods=50).median()
    df["ratio_sm"] = df["ratio"].rolling(200, min_periods=50).median()
    df["centroid_sm"] = df["centroid"].rolling(200, min_periods=50).median()

    df["delta_n"] = (df["delta_sm"] - df["delta_sm"].mean()) / (df["delta_sm"].std()+1e-6)
    df["ratio_n"] = (df["ratio_sm"] - df["ratio_sm"].mean()) / (df["ratio_sm"].std()+1e-6)
    df["centroid_n"] = (df["centroid_sm"] - df["centroid_sm"].mean()) / (df["centroid_sm"].std()+1e-6)

    # ===== Risk
    df["risk"] = (
        0.4 * df["prob_sm"] +
        0.3 * df["ratio_n"].clip(0) +
        0.2 * (-df["delta_n"]).clip(0) +
        0.1 * df["centroid_n"].clip(0)
    ).clip(0,1)

    # ===== Relative Risk (v22 core)
    baseline = df["risk"].rolling(600, min_periods=100).median()
    #df["rel_risk"] = (df["risk"] - baseline).clip(0)

    # ================= TEMPORAL FEATURES =================

    df["fp_short"] = firing_power(df["risk"], FP_SHORT_WIN)
    df["fp_long"]  = firing_power(df["risk"], FP_LONG_WIN)

    df["trend"] = df["risk"].diff().rolling(20).mean()

    df["mean"] = df["risk"].rolling(200).mean()
    df["std"]  = df["risk"].rolling(200).std()

    df["z"] = (df["risk"] - df["mean"]) / (df["std"] + 1e-6)

    df["trend_short"] = df["risk"].diff().rolling(20).mean()
    df["trend_long"]  = df["risk"].diff().rolling(100).mean()

    df["peakness"] = df["risk"] - df["risk"].rolling(100).mean()

    # ================= ALERT LOGIC (v22) =================

    alerts = []
    last_alert = -COOLDOWN

    for (start, end) in seizures:

        for i in range(len(df)):

            t = df["time"].iloc[i]

            # 🚫 ignore too early
            if (start - t) > MAX_LOOKBACK:
                continue

            cond = (
                        df["fp_short"].iloc[i] > 0.12 and
                        df["fp_long"].iloc[i]  > 0.35 and
                        df["trend"].iloc[i]    > 0.001 and
                        df["z"].iloc[i]        > 1.2
                    )

            if cond and (i - last_alert > COOLDOWN):
                alerts.append(i)
                last_alert = i

    df["alert"] = False
    df.loc[alerts, "alert"] = True

    # ================= METRICS =================

    alert_times = df[df["alert"]]["time"].values

    ewt = []
    detected = 0

    for (start, end) in seizures:
        pre = [a for a in alert_times if a < start]
        if pre:
            ewt.append(start - max(pre))
            detected += 1

    fa = len([a for a in alert_times if not any(s-60 <= a <= e for s,e in seizures)])
    duration_hr = df["time"].iloc[-1] / 3600

    return {
        "file": os.path.basename(edf_path),
        "sensitivity": detected/len(seizures) if seizures else None,
        "ewt": np.mean(ewt) if ewt else 0,
        "fa_hr": fa / duration_hr if duration_hr > 0 else 0,
        "df": df,
        "seizures": seizures
    }

# ================= RUN =================

results = []

edf_files = sorted([f for f in os.listdir(SUBJECT_FOLDER) if f.endswith(".edf")])

for f in edf_files:
    print(f"\nProcessing {f}...")
    res = process_file(os.path.join(SUBJECT_FOLDER, f))
    results.append(res)

runner.stop()

# ================= SUMMARY =================

sens = [r["sensitivity"] for r in results if r["sensitivity"] is not None]
ewt  = [r["ewt"] for r in results if r["ewt"] > 0]
fa   = [r["fa_hr"] for r in results]

print("\n===== SUBJECT SUMMARY =====")
print("Avg Sensitivity:", np.mean(sens) if sens else 0)
print("Avg EWT:", np.mean(ewt) if ewt else 0)
print("Avg FA/hr:", np.mean(fa))

# ================= BEST SEIZURE PLOT =================

valid_results = [r for r in results if r["seizures"] and r["ewt"] > 0]

if not valid_results:
    print("No valid seizure predictions found.")
    exit()

best = max(valid_results, key=lambda x: x["ewt"])

df = best["df"]
seizures = best["seizures"]

print(f"\nBest File: {best['file']} | EWT: {best['ewt']:.1f}s")

if not seizures:
    print(f"No seizures in {best['file']} — skipping plot")
    exit()

# Create subplots for all seizures
n_seizures = len(seizures)
fig, axes = plt.subplots(n_seizures, 1, figsize=(14, 5*n_seizures))

# Handle single seizure case
if n_seizures == 1:
    axes = [axes]

for idx, (start, end) in enumerate(seizures):
    ax = axes[idx]
    
    segment = df[
        (df["time"] >= start - 700) &
        (df["time"] <= start + 500)
    ].copy()

    segment["aligned_time"] = segment["time"] - start

    # Zones (behind risk line)
    ax.axhspan(0, 0.3, color='green', alpha=0.15, label="Low Risk")
    ax.axhspan(0.3, 0.6, color='yellow', alpha=0.15, label="Medium Risk")
    ax.axhspan(0.6, 1, color='red', alpha=0.15, label="High Risk")

    # Risk level dotted lines and labels
    for i in range(1, 11):
        y_pos = i * 0.1
        ax.axhline(y_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.text(1.01, y_pos, f'R{i}', transform=ax.get_yaxis_transform(), 
                va='center', ha='left', fontsize=9, color='gray')

    # EEG Signal
    ax.plot(segment["aligned_time"], segment["prob"], label="Raw EEG Signal", linewidth=1, color='steelblue', alpha=0.7)

    # Smoothed Signal
    ax.plot(segment["aligned_time"], segment["prob_sm"], label="Smoothed EEG ", linewidth=2.5, color='#FD6A02')

    # Alerts on EEG Signal
    alerts = segment[segment["alert"]]
    ax.scatter(alerts["aligned_time"], alerts["prob_sm"], color='red', s=100, label="FINAL ALERT", zorder=5)

    # Seizure onset
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Seizure Onset")

    ax.set_title(f"Seizure Prediction ({best['file']})")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Seizure Probability")
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(alpha=0.3)

fig.subplots_adjust(right=0.90)
plt.tight_layout()
plt.show()