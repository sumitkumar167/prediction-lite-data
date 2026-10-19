#!/usr/bin/env python3
"""
Unified CHB-MIT EEG dataset preparation pipeline.

Steps:
  1. Read CHB-MIT summary files and extract fixed-length EEG windows.
  2. Label each window as seizure / non_seizure using seizure intervals.
  3. Validate and clean duplicate CSV headers.
  4. Create a reproducible balanced dataset by downsampling non-seizure windows.
  5. Write manifests and a summary report.

Dependencies:
    pip install numpy pyedflib

IMPORTANT:
- CHANNEL_INDICES are zero-based EDF signal indices inherited from the original
  script: [5, 6, 9, 10]. They are NOT guaranteed to correspond to the channel
  names used in your inference script. Check the printed EDF channel mapping
  before relying on the output.
- Output columns use CH6,CH7,CH10,CH11 to preserve the original Edge Impulse
  CSV convention. Confirm that this matches the model's training input order.
- Windows are non-overlapping by default: 512 samples at 256 Hz = 2 seconds.
- A window is labeled seizure if it overlaps any annotated seizure interval.
- Output files are written into a fresh run directory to avoid stale files.
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyedflib


# ==============================
# CONFIGURATION — EDIT THESE
# ==============================

DATASET_PATH = Path(
    "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/"
    "EEG_Classification/MIT_Scalp_EEG_Dataset/"
    "physionet.org/files/chbmit/1.0.0"
)

OUTPUT_PATH = Path(
    "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/"
    "EEG_Classification/ws_2/edge_impulse_data2"
)

#TARGET_SUBJECTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]
TARGET_SUBJECTS = ["chb01"]

# Zero-based EDF signal indices from the original script.
CHANNEL_INDICES = [5, 6, 9, 10]

# CSV headers expected by the existing Edge Impulse dataset workflow.
# These names describe 1-based EDF signal indices, not verified EEG labels.
CSV_HEADER = ["CH6", "CH7", "CH10", "CH11"]

FS_EXPECTED = 256.0
WINDOW_SIZE = 512
STEP_SIZE = 512  # non-overlapping; use 128 for 0.5-second stride if desired

# Balanced output: number of non-seizure windows per seizure window.
NON_SEIZURE_TO_SEIZURE_RATIO = 1.0
RANDOM_SEED = 42

# If True, removes/recreates only the generated raw class folders and balanced
# output folder. Do not point OUTPUT_PATH at a directory containing other work.
REBUILD_GENERATED_OUTPUTS = True

# Whether to retain the unbalanced raw windows as well as balanced dataset.
KEEP_RAW_CLASS_DATA = True

# =========================================


SEIZURE_DIR_NAME = "seizure"
NON_SEIZURE_DIR_NAME = "non_seizure"
BALANCED_DIR_NAME = "balanced"


def parse_summary(summary_file: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Parse CHB-MIT summary annotations into seconds-based seizure intervals."""
    seizure_map: Dict[str, List[Tuple[float, float]]] = {}
    current_file = None
    pending_start = None

    with summary_file.open("r", encoding="latin-1") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            file_match = re.search(r"File Name:\s*(.+)", line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1).strip()
                seizure_map.setdefault(current_file, [])
                pending_start = None
                continue

            start_match = re.search(
                r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*(\d+)\s*seconds?",
                line,
                re.IGNORECASE,
            )
            if start_match and current_file:
                pending_start = float(start_match.group(1))
                continue

            end_match = re.search(
                r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*(\d+)\s*seconds?",
                line,
                re.IGNORECASE,
            )
            if end_match and current_file and pending_start is not None:
                end = float(end_match.group(1))
                if end < pending_start:
                    raise ValueError(
                        f"Invalid seizure interval in {summary_file}: "
                        f"{current_file} {pending_start}-{end}"
                    )
                seizure_map.setdefault(current_file, []).append(
                    (pending_start, end)
                )
                pending_start = None

    return seizure_map


def is_seizure_window(
    start_sec: float,
    end_sec: float,
    seizure_ranges: List[Tuple[float, float]],
) -> bool:
    """Use half-open intervals [start, end); label any positive overlap."""
    return any(
        start_sec < seizure_end and end_sec > seizure_start
        for seizure_start, seizure_end in seizure_ranges
    )


def clear_generated_outputs() -> None:
    """Clear only generated output folders, never the source EDF dataset."""
    generated = [
        OUTPUT_PATH / SEIZURE_DIR_NAME,
        OUTPUT_PATH / NON_SEIZURE_DIR_NAME,
        OUTPUT_PATH / BALANCED_DIR_NAME,
    ]
    for path in generated:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def print_channel_mapping(edf: pyedflib.EdfReader, edf_path: Path) -> None:
    labels = [str(x).strip() for x in edf.getSignalLabels()]
    print(f"  EDF channel mapping: {edf_path.name}")
    for index, label in enumerate(labels):
        marker = "  <== SELECTED" if index in CHANNEL_INDICES else ""
        print(f"    [{index}] {label}{marker}")


def validate_configuration() -> None:
    if len(CHANNEL_INDICES) != len(CSV_HEADER):
        raise ValueError(
            f"{len(CHANNEL_INDICES)} channel indices but "
            f"{len(CSV_HEADER)} CSV header columns."
        )
    if WINDOW_SIZE <= 0 or STEP_SIZE <= 0:
        raise ValueError("WINDOW_SIZE and STEP_SIZE must be positive.")
    if NON_SEIZURE_TO_SEIZURE_RATIO < 0:
        raise ValueError("Balance ratio cannot be negative.")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")


def write_window_csv(
    output_file: Path,
    window: np.ndarray,
) -> None:
    """Write one numeric EEG window with exactly one header."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(window.tolist())


def clean_csv_headers(directory: Path) -> int:
    """
    Validate CSVs and remove repeated header rows after the first row.
    Rewrites only files that contain a repeated header. Numeric data rows are
    retained; malformed/non-numeric rows are reported and cause an error.
    """
    cleaned = 0
    expected_lower = [x.lower() for x in CSV_HEADER]

    for csv_path in sorted(directory.rglob("*.csv")):
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))

        if not rows:
            raise ValueError(f"Empty CSV: {csv_path}")

        # Normalize and enforce the one expected header.
        first = [x.strip() for x in rows[0]]
        if [x.lower() for x in first] != expected_lower:
            # Support files whose first line is numeric / headerless.
            try:
                [float(x) for x in first]
            except (ValueError, TypeError):
                raise ValueError(
                    f"Unexpected header in {csv_path}: {first}; "
                    f"expected {CSV_HEADER}"
                )
            rows.insert(0, CSV_HEADER.copy())
            cleaned += 1

        out_rows = [rows[0]]
        for row_number, row in enumerate(rows[1:], start=2):
            stripped = [x.strip() for x in row]
            if [x.lower() for x in stripped] == expected_lower:
                cleaned += 1
                continue
            if len(stripped) != len(CSV_HEADER):
                raise ValueError(
                    f"{csv_path}:{row_number}: expected "
                    f"{len(CSV_HEADER)} columns, found {len(stripped)}"
                )
            try:
                [float(x) for x in stripped]
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{row_number}: non-numeric data row: {row}"
                ) from exc
            out_rows.append(stripped)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(out_rows)

    return cleaned


def process_subject(subject_dir: Path, counters: dict) -> None:
    summary_candidates = sorted(subject_dir.glob("*summary.txt"))
    if not summary_candidates:
        print(f"WARNING: no summary file found in {subject_dir}; skipping subject")
        counters["subjects_skipped_no_summary"] += 1
        return

    seizure_map = parse_summary(summary_candidates[0])
    edf_files = sorted(
        p for p in subject_dir.glob("*.edf")
        if not p.name.startswith("._")
    )
    if not edf_files:
        print(f"WARNING: no EDF files found in {subject_dir}")
        return

    for edf_path in edf_files:
        print(f"Processing {subject_dir.name}/{edf_path.name}")
        edf = None
        try:
            edf = pyedflib.EdfReader(str(edf_path))
            labels = [str(x).strip() for x in edf.getSignalLabels()]

            if max(CHANNEL_INDICES) >= edf.signals_in_file:
                raise ValueError(
                    f"Requested channel index {max(CHANNEL_INDICES)} but "
                    f"file has {edf.signals_in_file} signals. "
                    f"Labels: {labels}"
                )

            if counters["printed_mapping_files"] < 3:
                print_channel_mapping(edf, edf_path)
                counters["printed_mapping_files"] += 1

            sample_rates = [
                float(edf.getSampleFrequency(idx))
                for idx in CHANNEL_INDICES
            ]
            if not all(np.isclose(rate, FS_EXPECTED) for rate in sample_rates):
                raise ValueError(
                    f"Expected {FS_EXPECTED} Hz on selected channels; "
                    f"got {sample_rates}"
                )
            if not all(np.isclose(rate, sample_rates[0]) for rate in sample_rates):
                raise ValueError(f"Selected channels have different rates: {sample_rates}")

            # Read one channel at a time, then stack as (samples, channels).
            signals = np.column_stack([
                edf.readSignal(idx)
                for idx in CHANNEL_INDICES
            ])

            if not np.isfinite(signals).all():
                raise ValueError("Non-finite values found in EEG signals")

            seizure_ranges = seizure_map.get(edf_path.name, [])
            counters["edf_files_processed"] += 1
            counters["annotated_seizure_intervals"] += len(seizure_ranges)

            for start_index in range(
                0, signals.shape[0] - WINDOW_SIZE + 1, STEP_SIZE
            ):
                window = signals[start_index:start_index + WINDOW_SIZE, :]
                start_sec = start_index / sample_rates[0]
                end_sec = (start_index + WINDOW_SIZE) / sample_rates[0]

                is_seizure = is_seizure_window(
                    start_sec, end_sec, seizure_ranges
                )
                class_name = SEIZURE_DIR_NAME if is_seizure else NON_SEIZURE_DIR_NAME
                out_dir = OUTPUT_PATH / class_name
                out_name = (
                    f"{subject_dir.name}_{edf_path.stem}_"
                    f"sample{start_index:09d}.csv"
                )
                out_file = out_dir / out_name
                write_window_csv(out_file, window)

                counters["windows_total"] += 1
                counters["windows_seizure" if is_seizure else "windows_non_seizure"] += 1

                # Record provenance for auditing and reproducibility.
                with (OUTPUT_PATH / "window_manifest.csv").open(
                    "a", newline="", encoding="utf-8"
                ) as manifest:
                    writer = csv.writer(manifest)
                    if manifest.tell() == 0:
                        writer.writerow([
                            "subject", "edf_file", "start_sample",
                            "start_sec", "end_sec", "label", "csv_path"
                        ])
                    writer.writerow([
                        subject_dir.name, edf_path.name, start_index,
                        f"{start_sec:.6f}", f"{end_sec:.6f}",
                        "seizure" if is_seizure else "non_seizure",
                        str(out_file.relative_to(OUTPUT_PATH))
                    ])

        except Exception as exc:
            counters["edf_files_failed"] += 1
            print(f"  ERROR: {edf_path.name}: {exc}", file=sys.stderr)
        finally:
            if edf is not None:
                edf.close()


def make_balanced_dataset() -> dict:
    raw_seizure_dir = OUTPUT_PATH / SEIZURE_DIR_NAME
    raw_non_dir = OUTPUT_PATH / NON_SEIZURE_DIR_NAME
    out_seizure = OUTPUT_PATH / BALANCED_DIR_NAME / SEIZURE_DIR_NAME
    out_non = OUTPUT_PATH / BALANCED_DIR_NAME / NON_SEIZURE_DIR_NAME

    # Always rebuild the balanced folder so stale files cannot survive a run.
    if (OUTPUT_PATH / BALANCED_DIR_NAME).exists():
        shutil.rmtree(OUTPUT_PATH / BALANCED_DIR_NAME)
    out_seizure.mkdir(parents=True, exist_ok=True)
    out_non.mkdir(parents=True, exist_ok=True)

    seizure_files = sorted(raw_seizure_dir.glob("*.csv"))
    non_files = sorted(raw_non_dir.glob("*.csv"))

    target_non = min(
        len(non_files),
        int(round(len(seizure_files) * NON_SEIZURE_TO_SEIZURE_RATIO))
    )
    rng = random.Random(RANDOM_SEED)
    sampled_non = sorted(rng.sample(non_files, target_non)) if target_non else []

    for src in seizure_files:
        shutil.copy2(src, out_seizure / src.name)
    for src in sampled_non:
        shutil.copy2(src, out_non / src.name)

    # Manifest of exact files included in balanced output.
    manifest_path = OUTPUT_PATH / "balanced_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "filename", "source_path", "random_seed"])
        for src in seizure_files:
            writer.writerow([
                "seizure", src.name, str(src.relative_to(OUTPUT_PATH)), RANDOM_SEED
            ])
        for src in sampled_non:
            writer.writerow([
                "non_seizure", src.name, str(src.relative_to(OUTPUT_PATH)), RANDOM_SEED
            ])

    return {
        "raw_seizure_files": len(seizure_files),
        "raw_non_seizure_files": len(non_files),
        "balanced_seizure_files": len(seizure_files),
        "balanced_non_seizure_files": len(sampled_non),
        "ratio_non_to_seizure": NON_SEIZURE_TO_SEIZURE_RATIO,
        "random_seed": RANDOM_SEED,
    }


def main() -> None:
    validate_configuration()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    if REBUILD_GENERATED_OUTPUTS:
        clear_generated_outputs()

    # Reinitialize manifest.
    manifest_path = OUTPUT_PATH / "window_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "subject", "edf_file", "start_sample", "start_sec",
            "end_sec", "label", "csv_path"
        ])

    counters = {
        "printed_mapping_files": 0,
        "subjects_skipped_no_summary": 0,
        "edf_files_processed": 0,
        "edf_files_failed": 0,
        "annotated_seizure_intervals": 0,
        "windows_total": 0,
        "windows_seizure": 0,
        "windows_non_seizure": 0,
    }

    print("=" * 72)
    print("CHB-MIT unified preparation pipeline")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output:  {OUTPUT_PATH}")
    print(f"Subjects: {', '.join(TARGET_SUBJECTS)}")
    print(f"Channels (zero-based EDF indices): {CHANNEL_INDICES}")
    print(f"CSV header: {CSV_HEADER}")
    print(
        f"Window: {WINDOW_SIZE} samples ({WINDOW_SIZE / FS_EXPECTED:.2f}s), "
        f"stride: {STEP_SIZE} samples ({STEP_SIZE / FS_EXPECTED:.2f}s)"
    )
    print("=" * 72)

    for subject in TARGET_SUBJECTS:
        subject_dir = DATASET_PATH / subject
        if not subject_dir.is_dir():
            print(f"WARNING: subject folder not found: {subject_dir}")
            continue
        process_subject(subject_dir, counters)

    # Clean/validate both raw class directories. This is idempotent.
    headers_cleaned = 0
    headers_cleaned += clean_csv_headers(OUTPUT_PATH / SEIZURE_DIR_NAME)
    headers_cleaned += clean_csv_headers(OUTPUT_PATH / NON_SEIZURE_DIR_NAME)

    balance_stats = make_balanced_dataset()

    report = {
        "configuration": {
            "dataset_path": str(DATASET_PATH),
            "output_path": str(OUTPUT_PATH),
            "target_subjects": TARGET_SUBJECTS,
            "channel_indices_zero_based": CHANNEL_INDICES,
            "csv_header": CSV_HEADER,
            "expected_sampling_rate_hz": FS_EXPECTED,
            "window_size_samples": WINDOW_SIZE,
            "step_size_samples": STEP_SIZE,
            "non_seizure_to_seizure_ratio": NON_SEIZURE_TO_SEIZURE_RATIO,
            "random_seed": RANDOM_SEED,
            "label_policy": "positive overlap with seizure interval [start,end)",
        },
        "processing": counters,
        "duplicate_or_missing_headers_fixed": headers_cleaned,
        "balancing": balance_stats,
    }

    report_path = OUTPUT_PATH / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print(f"EDF files processed: {counters['edf_files_processed']}")
    print(f"EDF files failed: {counters['edf_files_failed']}")
    print(f"Annotated seizure intervals: {counters['annotated_seizure_intervals']}")
    print(f"Raw seizure windows: {counters['windows_seizure']}")
    print(f"Raw non-seizure windows: {counters['windows_non_seizure']}")
    print(f"Header fixes: {headers_cleaned}")
    print(f"Balanced seizure windows: {balance_stats['balanced_seizure_files']}")
    print(f"Balanced non-seizure windows: {balance_stats['balanced_non_seizure_files']}")
    print(f"Report: {report_path}")
    print(f"Window manifest: {OUTPUT_PATH / 'window_manifest.csv'}")
    print(f"Balanced manifest: {OUTPUT_PATH / 'balanced_manifest.csv'}")
    print("=" * 72)

    if counters["edf_files_failed"]:
        print(
            "WARNING: Some EDF files failed. Inspect the error messages and "
            "dataset_report.json before using this dataset."
        )


if __name__ == "__main__":
    main()
