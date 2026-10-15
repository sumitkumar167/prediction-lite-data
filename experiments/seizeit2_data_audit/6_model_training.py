#!/usr/bin/env python3
"""
Step 1 of the personalization work:
rebuild the Prediction-Lite model in plain Python and check that it gives
about the same results as Edge Impulse.

Edge Impulse results we want to roughly match (EEG only, SeizeIT2 pilot):
    validation (random 20% of windows) : ~90% accuracy
    held-out patient sub-103            : ~61% accuracy

What this script does:
    1. Cut every cached segment into 2-second windows (0.5 s stride)
    2. Turn each window into spectral features (same idea as Edge Impulse)
    3. Train the same small network: 64 -> 32 neurons
    4. Test it two ways: random validation windows, and an unseen patient

Once this works, the same functions are reused for the personal models.

Requires: numpy only. The network is written out by hand below, so there is
no scikit-learn or SciPy (SciPy fails to load on this Mac).
"""

import csv
import random
from pathlib import Path

import numpy as np

# ======================= SETTINGS — EDIT THESE =======================

WORKSPACE = Path("/Users/sumitkumar/Downloads/Lectures/Fall 2026/Individual Instruction/SiezeIT2_ws")
CACHE_DIR = WORKSPACE / "seizeit2_cache"                  # from 4_extract_segments.py
FEATURES_FILE = WORKSPACE / "seizeit2_features_v2.npz"    # created by this script

SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENT_INDEX = SCRIPT_DIR / "seizeit2_segments.csv"      # from 4_extract_segments.py

TEST_PATIENT = "sub-103"      # never used for training
SEED = 42                     # makes every run give the same result

# Prediction-Lite settings (same as the Edge Impulse project)
SAMPLE_RATE = 256             # Hz
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
SCALE = 3                     # "scale axes"
DECIMATION = 3                # "input decimation ratio"
FFT_LENGTH = 128

# ======================================================================


# ---------------------------------------------------------------------
# Part 1: features
# ---------------------------------------------------------------------

def make_lowpass_filter():
    """
    Before keeping every 3rd sample (decimation), frequencies above the new
    Nyquist limit must be removed, otherwise they fold back as false signal.
    This builds a simple low-pass filter for that.
    """
    number_of_taps = 31
    cutoff = 0.8 / DECIMATION                      # fraction of the old Nyquist frequency
    positions = np.arange(number_of_taps) - (number_of_taps - 1) / 2
    taps = cutoff * np.sinc(cutoff * positions) * np.hamming(number_of_taps)
    return taps / np.sum(taps)


LOWPASS_TAPS = make_lowpass_filter()


def channel_features(signal):
    """
    Features for ONE channel of ONE 2-second window.
    Returns 67 numbers: RMS, skewness, kurtosis, and 64 spectral power values.
    """
    # 1. Scale, low-pass, and keep every 3rd sample (256 Hz -> ~85 Hz)
    scaled = signal * SCALE
    smoothed = np.convolve(scaled, LOWPASS_TAPS, mode="same")
    decimated = smoothed[::DECIMATION]
    decimated = decimated - np.mean(decimated)

    # 2. Simple statistics of the signal shape
    std = np.std(decimated)
    rms = np.sqrt(np.mean(decimated ** 2))
    if std > 0:
        skewness = np.mean((decimated / std) ** 3)
        kurtosis = np.mean((decimated / std) ** 4) - 3
    else:
        skewness = 0.0
        kurtosis = 0.0

    # 3. Spectrum: average the power of overlapping 128-sample frames
    frame_step = FFT_LENGTH // 2
    power_sum = np.zeros(FFT_LENGTH // 2 + 1)
    number_of_frames = 0
    start = 0
    while start + FFT_LENGTH <= len(decimated):
        frame = decimated[start:start + FFT_LENGTH]
        power_sum += np.abs(np.fft.rfft(frame)) ** 2 / FFT_LENGTH
        number_of_frames += 1
        start += frame_step
    if number_of_frames == 0:                       # window shorter than one frame
        frame = np.zeros(FFT_LENGTH)
        frame[:len(decimated)] = decimated
        power_sum = np.abs(np.fft.rfft(frame)) ** 2 / FFT_LENGTH
        number_of_frames = 1
    average_power = power_sum / number_of_frames

    # 4. Log of the spectrum, dropping bin 0 (the constant/DC part)
    log_power = np.log10(average_power[1:] + 1e-10)

    return np.concatenate([[rms, skewness, kurtosis], log_power])


def window_features(eeg_window):
    """Features for one window: both EEG channels side by side (134 numbers)."""
    bte_features = channel_features(eeg_window[0])
    cross_features = channel_features(eeg_window[1])
    return np.concatenate([bte_features, cross_features])


# ---------------------------------------------------------------------
# Part 2: build the dataset from the cache
# ---------------------------------------------------------------------

def build_dataset():
    """
    Go through every ictal and interictal segment, cut it into windows,
    and compute features. Pre-ictal segments are skipped (they are kept
    for testing early warning later, same as in Edge Impulse).
    """
    window_size = int(WINDOW_SECONDS * SAMPLE_RATE)       # 512 samples
    stride = int(STRIDE_SECONDS * SAMPLE_RATE)             # 128 samples

    features = []
    labels = []          # 1 = seizure, 0 = non-seizure
    subjects = []
    segment_ids = []

    with open(SEGMENT_INDEX) as file:
        segment_rows = list(csv.DictReader(file))

    for row in segment_rows:
        if row["kind"] == "ictal":
            label = 1
        elif row["kind"] == "interictal":
            label = 0
        else:
            continue                                       # skip pre-ictal

        data = np.load(CACHE_DIR / row["file"])
        eeg = data["eeg"]                                  # shape: (2 channels, samples)

        window_number = 0
        start = 0
        while start + window_size <= eeg.shape[1]:
            # Every window is kept, for BOTH classes. If artifact windows were removed
            # only from non-seizure data, the model could learn the shortcut
            # "big noisy signal = seizure", and would raise false alarms on every
            # head movement in real use. The headgear will see artifacts too.
            features.append(window_features(eeg[:, start:start + window_size]))
            labels.append(label)
            subjects.append(row["subject"])
            segment_ids.append(row["segment_id"])
            start += stride
            window_number += 1

        print(f"  {row['segment_id']}: {window_number} windows")

    return np.array(features), np.array(labels), np.array(subjects), np.array(segment_ids)


def load_or_build_dataset():
    """Computing features takes a few minutes, so save them and reuse them."""
    if FEATURES_FILE.exists():
        print(f"Loading saved features from {FEATURES_FILE.name}")
        saved = np.load(FEATURES_FILE)
        return saved["features"], saved["labels"], saved["subjects"], saved["segment_ids"]

    print("Computing features (first run only) ...")
    features, labels, subjects, segment_ids = build_dataset()
    np.savez_compressed(FEATURES_FILE, features=features, labels=labels,
                        subjects=subjects, segment_ids=segment_ids)
    print(f"Saved features to {FEATURES_FILE}")
    return features, labels, subjects, segment_ids


# ---------------------------------------------------------------------
# Part 3: train and test
# ---------------------------------------------------------------------

def balance_classes(features, labels):
    """
    There are ~3x more non-seizure windows than seizure windows.
    Randomly keep only as many non-seizure windows as there are seizure windows,
    so the model doesn't just learn to say "non-seizure".
    """
    rng = np.random.default_rng(SEED)
    seizure_rows = np.where(labels == 1)[0]
    normal_rows = np.where(labels == 0)[0]
    count = min(len(seizure_rows), len(normal_rows))
    keep = np.concatenate([rng.choice(seizure_rows, count, replace=False),
                           rng.choice(normal_rows, count, replace=False)])
    return features[keep], labels[keep]


# ---------------------------------------------------------------------
# The neural network: input -> 64 -> 32 -> 1 (probability of seizure)
# ---------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def new_network(number_of_inputs, rng):
    """Random starting weights for the three layers."""
    sizes = [number_of_inputs, 64, 32, 1]
    network = {}
    for layer in range(3):
        fan_in = sizes[layer]
        network[f"W{layer}"] = rng.normal(0, np.sqrt(2 / fan_in), (sizes[layer], sizes[layer + 1]))
        network[f"b{layer}"] = np.zeros(sizes[layer + 1])
    return network


def forward(network, x):
    """Run inputs through the network. Returns every layer's output (needed for training)."""
    hidden1 = relu(x @ network["W0"] + network["b0"])
    hidden2 = relu(hidden1 @ network["W1"] + network["b1"])
    output = sigmoid(hidden2 @ network["W2"] + network["b2"])[:, 0]
    return hidden1, hidden2, output


def gradients(network, x, y):
    """
    Backpropagation: how much each weight should change to reduce the error.
    Uses the standard cross-entropy loss for a yes/no (seizure / non-seizure) output.
    """
    hidden1, hidden2, output = forward(network, x)
    batch = len(y)

    error_out = ((output - y) / batch)[:, None]              # output layer
    grads = {"W2": hidden2.T @ error_out, "b2": error_out.sum(axis=0)}

    error_h2 = (error_out @ network["W2"].T) * (hidden2 > 0)  # second hidden layer
    grads["W1"] = hidden1.T @ error_h2
    grads["b1"] = error_h2.sum(axis=0)

    error_h1 = (error_h2 @ network["W1"].T) * (hidden1 > 0)   # first hidden layer
    grads["W0"] = x.T @ error_h1
    grads["b0"] = error_h1.sum(axis=0)
    return grads


def train_model(features, labels, show_progress=True):
    """
    Train the same network as the paper: 64 -> 32, 30 epochs, batch size 32,
    learning rate 0.001, Adam optimizer (Edge Impulse's default).
    Returns (scaler, network); the scaler stores the feature mean and spread.
    """
    features, labels = balance_classes(features, labels)

    # Normalize every feature to mean 0, spread 1 (same as Edge Impulse)
    mean = features.mean(axis=0)
    spread = features.std(axis=0) + 1e-8
    scaler = (mean, spread)
    x_all = (features - mean) / spread
    y_all = labels.astype(float)

    rng = np.random.default_rng(SEED)
    network = new_network(x_all.shape[1], rng)

    # Adam keeps a running average of each gradient (m) and of its size (v)
    learning_rate, beta1, beta2 = 0.001, 0.9, 0.999
    m = {name: np.zeros_like(value) for name, value in network.items()}
    v = {name: np.zeros_like(value) for name, value in network.items()}
    step = 0

    for epoch in range(30):
        order = rng.permutation(len(y_all))
        for start in range(0, len(order), 32):
            batch_rows = order[start:start + 32]
            grads = gradients(network, x_all[batch_rows], y_all[batch_rows])
            step += 1
            for name in network:
                m[name] = beta1 * m[name] + (1 - beta1) * grads[name]
                v[name] = beta2 * v[name] + (1 - beta2) * grads[name] ** 2
                m_corrected = m[name] / (1 - beta1 ** step)
                v_corrected = v[name] / (1 - beta2 ** step)
                network[name] -= learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
        if show_progress and (epoch + 1) % 10 == 0:
            accuracy = 100 * np.mean((forward(network, x_all)[2] > 0.5) == (y_all == 1))
            print(f"  epoch {epoch + 1:>2}: training accuracy {accuracy:.1f}%")

    return scaler, network


def predict(scaler, network, features):
    """1 = seizure, 0 = non-seizure."""
    mean, spread = scaler
    probability = forward(network, (features - mean) / spread)[2]
    return (probability > 0.5).astype(int)


def evaluate(scaler, network, features, labels):
    """Return accuracy, seizure recall, and non-seizure recall (all in %)."""
    predictions = predict(scaler, network, features)
    accuracy = 100 * np.mean(predictions == labels)
    seizure_recall = 100 * np.mean(predictions[labels == 1] == 1)
    normal_recall = 100 * np.mean(predictions[labels == 0] == 0)
    return accuracy, seizure_recall, normal_recall


def print_result(name, result):
    accuracy, seizure_recall, normal_recall = result
    print(f"  {name:<34} accuracy {accuracy:5.1f}%   "
          f"seizures caught {seizure_recall:5.1f}%   normal correct {normal_recall:5.1f}%")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    random.seed(SEED)

    # Step 1: features for every window
    features, labels, subjects, segment_ids = load_or_build_dataset()
    print(f"\n{len(labels)} windows: {np.sum(labels == 1)} seizure, {np.sum(labels == 0)} non-seizure")
    print(f"{features.shape[1]} features per window\n")

    # Step 2: split off the held-out patient
    is_test_patient = subjects == TEST_PATIENT
    train_features = features[~is_test_patient]
    train_labels = labels[~is_test_patient]
    test_features = features[is_test_patient]
    test_labels = labels[is_test_patient]

    # Step 3: like Edge Impulse, keep a random 20% of training windows for validation
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(train_labels))
    validation_count = len(train_labels) // 5
    validation_rows = shuffled[:validation_count]
    fit_rows = shuffled[validation_count:]

    # Step 4: train
    print("Training 64 -> 32 network ...")
    scaler, network = train_model(train_features[fit_rows], train_labels[fit_rows])

    # Step 5: test
    print("\nRESULTS (EEG only)")
    print_result("Validation (random 20% of windows)",
                 evaluate(scaler, network, train_features[validation_rows], train_labels[validation_rows]))
    print_result(f"Held-out patient {TEST_PATIENT}",
                 evaluate(scaler, network, test_features, test_labels))
    print("\nEdge Impulse for comparison: validation ~90%, held-out sub-103 ~61%")


if __name__ == "__main__":
    main()