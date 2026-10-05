import os
import numpy as np
import pyedflib
import re

# ----------------------------------------
# CONFIG
# ----------------------------------------

DATASET_PATH = "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/EEG_Classification/MIT_Scalp_EEG_Dataset/physionet.org/files/chbmit/1.0.0/"
OUTPUT_PATH = "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/EEG_Classification/ws_2/edge_impulse_data2/"

WINDOW_SIZE = 512
STEP_SIZE = 512
CHANNELS_TO_USE = [5, 6, 9, 10]

# 👉 Only these subjects
TARGET_SUBJECTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]

# 👉 Header for Edge Impulse
HEADER = ",".join([f"ch{i+1}" for i in range(len(CHANNELS_TO_USE))])

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ----------------------------------------
# Parse summary file
# ----------------------------------------

def parse_summary(summary_file):
    seizure_info = {}
    current_file = None

    # fix encoding issue
    with open(summary_file, 'r', encoding="latin-1") as f:
        for line in f:
            line = line.strip()

            if "File Name:" in line:
                current_file = line.split(":")[1].strip()
                seizure_info[current_file] = []

            if "Seizure Start Time:" in line:
                start = int(re.findall(r'\d+', line)[0])

            if "Seizure End Time:" in line:
                end = int(re.findall(r'\d+', line)[0])
                seizure_info[current_file].append((start, end))

    return seizure_info

# ----------------------------------------
# Check seizure overlap
# ----------------------------------------

def is_seizure_window(start_sec, end_sec, seizure_ranges):
    for s, e in seizure_ranges:
        if not (end_sec < s or start_sec > e):
            return True
    return False

# ----------------------------------------
# Process EDF files
# ----------------------------------------

def process_patient_folder(folder_path):

    summary_file = None

    for file in os.listdir(folder_path):
        if file.endswith("summary.txt"):
            summary_file = os.path.join(folder_path, file)

    if summary_file is None:
        print("❌ No summary file in:", folder_path)
        return

    seizure_map = parse_summary(summary_file)

    for file in os.listdir(folder_path):

        # ❗ Skip macOS garbage files
        if file.startswith("._") or not file.endswith(".edf"):
            continue

        edf_path = os.path.join(folder_path, file)
        print("Processing:", edf_path)

        try:
            f = pyedflib.EdfReader(edf_path)

            # read selected channels
            signals = np.array([f.readSignal(ch) for ch in CHANNELS_TO_USE])
            signals = signals.T

            total_samples = signals.shape[0]
            seizure_ranges = seizure_map.get(file, [])

            for i in range(0, total_samples - WINDOW_SIZE, STEP_SIZE):

                window = signals[i:i+WINDOW_SIZE]

                start_sec = i / 256
                end_sec = (i + WINDOW_SIZE) / 256

                label = 1 if is_seizure_window(start_sec, end_sec, seizure_ranges) else 0
                label_name = "seizure" if label == 1 else "non_seizure"

                out_dir = os.path.join(OUTPUT_PATH, label_name)
                os.makedirs(out_dir, exist_ok=True)

                out_file = os.path.join(out_dir, f"{file}_{i}.csv")

                # ✅ Save with header (NO # symbol)
                np.savetxt(
                    out_file,
                    window,
                    delimiter=",",
                    header=HEADER,
                    comments=""
                )

            f.close()

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# ----------------------------------------
# MAIN LOOP
# ----------------------------------------

for folder in os.listdir(DATASET_PATH):

    if folder not in TARGET_SUBJECTS:
        continue

    folder_path = os.path.join(DATASET_PATH, folder)

    if os.path.isdir(folder_path):
        print("\n📁 Processing folder:", folder)
        process_patient_folder(folder_path)

print("\n✅ Done.")