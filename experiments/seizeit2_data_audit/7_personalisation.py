#!/usr/bin/env python3
"""
Step 2: does adapting the model to one patient improve it?

Simulates how the headgear would be used. Each test patient's recordings are
cut at ONE point in time:
    EARLY : everything recorded before the cut  -> what the device has seen so far
    LATER : everything after the cut            -> the test set (same for every stage)
The cut is placed so that about a third of the patient's seizures are EARLY.
Nothing from the LATER period is ever used for training or calibration.

Four stages, all tested on the same LATER data:
    Stage 1  Generic          trained on the other patients only
    Stage 2  + Baseline       every patient's features expressed relative to their own
                              normal signal; the test patient uses only EARLY normal data
    Stage 3  + First seizures Stage 2, plus the patient's EARLY data added to training,
                              making up a fixed share of the training set
    Stage 4  Personal only    trained ONLY on the patient's EARLY data

Each stage is trained several times with different random seeds and averaged,
so small differences between stages are not mistaken for real improvements.

Run 6_model_training.py first (it creates the features file this script reuses).
Requires: numpy only.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np

# ======================= SETTINGS — EDIT THESE =======================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SCRIPT = SCRIPT_DIR / "6_model_training.py"     # the network and feature code
SEGMENT_INDEX = SCRIPT_DIR / "seizeit2_segments.csv"

TEST_PATIENTS = ["sub-073", "sub-087"]   # the two patients with enough seizures
EARLY_FRACTION = 0.33                    # about a third of the seizures fall before the cut
PERSONAL_SHARE = 0.5                     # Stage 3: the patient's own data = half of training
SEEDS = [1, 2, 3]                        # each stage is trained once per seed, then averaged
SEIZURE_DETECTED_IF = 0.5                # a seizure counts as detected if at least half
                                         # of its windows are flagged

# ======================================================================


# Reuse the network and features from 6_model_training.py
# (its name starts with a digit, so it is loaded by file path)
spec = importlib.util.spec_from_file_location("model_code", MODEL_SCRIPT)
model_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_code)


# ---------------------------------------------------------------------
# Splitting a patient's data in time
# ---------------------------------------------------------------------

def segment_start_times():
    """
    For each segment: (run number, start second). Runs are numbered in recording
    order, so sorting by this pair puts segments in the order they were recorded.
    """
    start_time = {}
    with open(SEGMENT_INDEX) as file:
        for row in csv.DictReader(file):
            start_time[row["segment_id"]] = (int(row["run"]), float(row["t_start_s"]))
    return start_time


def split_at_one_point_in_time(patient, labels, subjects, segment_ids, start_time):
    """
    Find the cut: the start of the first seizure that belongs to the LATER period.
    Every window of this patient recorded before the cut is EARLY, the rest is LATER.
    Using one cut for both seizure and normal data means the device never "sees"
    anything from the future.
    """
    patient_seizures = set(segment_ids[(subjects == patient) & (labels == 1)])
    seizures_in_order = sorted(patient_seizures, key=start_time.get)   # oldest first

    early_count = max(1, int(round(len(seizures_in_order) * EARLY_FRACTION)))
    if early_count >= len(seizures_in_order):
        raise SystemExit(f"{patient}: too few seizures to split into EARLY and LATER")
    cut = start_time[seizures_in_order[early_count]]

    is_early = np.zeros(len(labels), dtype=bool)
    is_later = np.zeros(len(labels), dtype=bool)
    for i in np.where(subjects == patient)[0]:
        if start_time[segment_ids[i]] < cut:
            is_early[i] = True
        else:
            is_later[i] = True

    # Every stage needs early normal data (calibration) and both classes in the test set
    for name, rows in (("EARLY normal", is_early & (labels == 0)),
                       ("LATER normal", is_later & (labels == 0)),
                       ("LATER seizure", is_later & (labels == 1))):
        if not rows.any():
            raise SystemExit(f"{patient}: no {name} windows — change EARLY_FRACTION")
    return is_early, is_later


# ---------------------------------------------------------------------
# Calibration to each person's normal signal
# ---------------------------------------------------------------------

def calibrate(features, reference_features):
    """
    Express features relative to a person's own normal signal:
    0 means "same as this person's normal", 2 means "2 spreads above their normal".
    """
    mean = reference_features.mean(axis=0)
    spread = reference_features.std(axis=0) + 1e-8
    return (features - mean) / spread


def calibrate_training_patients(features, labels, subjects, patients):
    """Training patients are fully known, so each is calibrated with all of their normal data."""
    calibrated = features.copy()
    for patient in patients:
        rows = subjects == patient
        calibrated[rows] = calibrate(features[rows], features[rows & (labels == 0)])
    return calibrated


# ---------------------------------------------------------------------
# Stage 3: mix the patient's own data in at a fixed share
# ---------------------------------------------------------------------

def mix_in_personal_data(other_features, other_labels, own_features, own_labels, seed):
    """
    Add the patient's own windows so they make up PERSONAL_SHARE of each class.
    Their windows are drawn with repetition, because they have far fewer than
    the other patients. Doing it per class keeps the share the same for both
    seizure and normal data, and the same for every test patient.
    """
    rng = np.random.default_rng(seed)
    parts_features = [other_features]
    parts_labels = [other_labels]
    for label in (1, 0):
        others_count = np.sum(other_labels == label)
        own_rows = np.where(own_labels == label)[0]
        wanted = int(others_count * PERSONAL_SHARE / (1 - PERSONAL_SHARE))
        picked = rng.choice(own_rows, wanted, replace=True)
        parts_features.append(own_features[picked])
        parts_labels.append(own_labels[picked])
    return np.concatenate(parts_features), np.concatenate(parts_labels)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def score(scaler, network, features, labels, segment_ids):
    """
    Window level : seizures caught, normal correct, balanced accuracy (their average)
    Seizure level: how many seizures had at least SEIZURE_DETECTED_IF of their windows flagged
    """
    predictions = model_code.predict(scaler, network, features)
    seizure_recall = 100 * np.mean(predictions[labels == 1] == 1)
    normal_recall = 100 * np.mean(predictions[labels == 0] == 0)
    balanced = (seizure_recall + normal_recall) / 2

    detected = 0
    seizure_list = sorted(set(segment_ids[labels == 1]))
    for seizure in seizure_list:
        rows = segment_ids == seizure
        if np.mean(predictions[rows]) >= SEIZURE_DETECTED_IF:
            detected += 1

    return {"balanced": balanced, "seizure": seizure_recall, "normal": normal_recall,
            "detected": detected, "total": len(seizure_list)}


def train_and_score(train_features, train_labels, test_features, test_labels, test_segments, seed):
    model_code.SEED = seed                 # train_model and balance_classes read this setting
    scaler, network = model_code.train_model(train_features, train_labels, show_progress=False)
    return score(scaler, network, test_features, test_labels, test_segments)


# ---------------------------------------------------------------------
# The four stages for one patient
# ---------------------------------------------------------------------

def run_stages(patient, features, labels, subjects, segment_ids, start_time):
    others = []
    for other in sorted(set(subjects)):
        if other != patient:
            others.append(other)
    from_others = np.isin(subjects, others)

    is_early, is_later = split_at_one_point_in_time(patient, labels, subjects, segment_ids, start_time)
    early_seizures = len(set(segment_ids[is_early & (labels == 1)]))
    later_seizures = len(set(segment_ids[is_later & (labels == 1)]))
    early_normal_minutes = np.sum(is_early & (labels == 0)) * model_code.STRIDE_SECONDS / 60
    print(f"\n{patient}: EARLY = {early_seizures} seizures + {early_normal_minutes:.0f} min normal,"
          f" tested on {later_seizures} LATER seizures")
    print(f"  other patients: {', '.join(others)}")

    # Calibrated features (Stages 2-4). The test patient uses only EARLY normal data.
    others_calibrated = calibrate_training_patients(features, labels, subjects, others)
    patient_calibrated = calibrate(features, features[is_early & (labels == 0)])

    raw_test = features[is_later]
    calibrated_test = patient_calibrated[is_later]
    test_labels = labels[is_later]
    test_segments = segment_ids[is_later]

    stage_results = {"Stage 1  Generic": [], "Stage 2  + Baseline": [],
                     "Stage 3  + First seizures": [], "Stage 4  Personal only": []}

    for seed in SEEDS:
        print(f"  training with seed {seed} ...")

        stage_results["Stage 1  Generic"].append(train_and_score(
            features[from_others], labels[from_others],
            raw_test, test_labels, test_segments, seed))

        stage_results["Stage 2  + Baseline"].append(train_and_score(
            others_calibrated[from_others], labels[from_others],
            calibrated_test, test_labels, test_segments, seed))

        mixed_features, mixed_labels = mix_in_personal_data(
            others_calibrated[from_others], labels[from_others],
            patient_calibrated[is_early], labels[is_early], seed)
        stage_results["Stage 3  + First seizures"].append(train_and_score(
            mixed_features, mixed_labels,
            calibrated_test, test_labels, test_segments, seed))

        stage_results["Stage 4  Personal only"].append(train_and_score(
            patient_calibrated[is_early], labels[is_early],
            calibrated_test, test_labels, test_segments, seed))

    return stage_results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def print_stage(stage, runs):
    """Average over seeds; +/- is the spread between seeds."""
    balanced = [run["balanced"] for run in runs]
    seizure = [run["seizure"] for run in runs]
    normal = [run["normal"] for run in runs]
    detected = [run["detected"] for run in runs]
    total = runs[0]["total"]
    print(f"  {stage:<27} balanced {np.mean(balanced):5.1f}% ±{np.std(balanced):4.1f}   "
          f"seizure windows {np.mean(seizure):5.1f}%   normal windows {np.mean(normal):5.1f}%   "
          f"seizures detected {np.mean(detected):4.1f} of {total}")


def main():
    features, labels, subjects, segment_ids = model_code.load_or_build_dataset()
    start_time = segment_start_times()

    all_results = {}
    for patient in TEST_PATIENTS:
        all_results[patient] = run_stages(patient, features, labels, subjects, segment_ids, start_time)

    print("\n" + "=" * 118)
    print(f"PERSONALIZATION RESULTS (EEG only, tested on each patient's LATER data, "
          f"average of {len(SEEDS)} runs)")
    print("balanced = average of seizure windows caught and normal windows correct (50% = guessing)")
    print("=" * 118)
    for patient, stage_results in all_results.items():
        print(f"\n{patient}")
        for stage, runs in stage_results.items():
            print_stage(stage, runs)
    print()


if __name__ == "__main__":
    main()