import os
import random

# ----------------------------------------
# CONFIG
# ----------------------------------------

DATASET_PATH = "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/EEG_Classification/ws_2/edge_impulse_data2/"

seizure_dir = os.path.join(DATASET_PATH, "seizure")
non_seizure_dir = os.path.join(DATASET_PATH, "non_seizure")

output_dir = os.path.join(DATASET_PATH, "balanced")
out_seizure = os.path.join(output_dir, "seizure")
out_non = os.path.join(output_dir, "non_seizure")

os.makedirs(out_seizure, exist_ok=True)
os.makedirs(out_non, exist_ok=True)

# ✅ Correct header
HEADER = "CH6,CH7,CH10,CH11"

# ----------------------------------------
# Helper: copy + fix header
# ----------------------------------------

def copy_with_header(src_path, dst_path):

    with open(src_path, "r") as f:
        first_line = f.readline()

    # If header already exists → copy directly
    if "CH6" in first_line:
        with open(src_path, "r") as src, open(dst_path, "w") as dst:
            dst.write(src.read())
        return

    # Otherwise → add header
    with open(src_path, "r") as src:
        content = src.read()

    with open(dst_path, "w") as dst:
        dst.write(HEADER + "\n")
        dst.write(content)

# ----------------------------------------
# Load files
# ----------------------------------------

seizure_files = [f for f in os.listdir(seizure_dir) if f.endswith(".csv")]
non_files = [f for f in os.listdir(non_seizure_dir) if f.endswith(".csv")]

print("Seizure:", len(seizure_files))
print("Non-seizure:", len(non_files))

# ----------------------------------------
# Downsample non-seizure
# ----------------------------------------

ratio = 1

target_non = min(len(non_files), len(seizure_files) * ratio)
non_sampled = random.sample(non_files, target_non)

# ----------------------------------------
# Copy + fix headers
# ----------------------------------------

print("Copying seizure files...")
for f in seizure_files:
    copy_with_header(
        os.path.join(seizure_dir, f),
        os.path.join(out_seizure, f)
    )

print("Copying non-seizure files...")
for f in non_sampled:
    copy_with_header(
        os.path.join(non_seizure_dir, f),
        os.path.join(out_non, f)
    )

# ----------------------------------------
# Done
# ----------------------------------------

print("\n✅ Balanced dataset created!")
print(f"Seizure: {len(seizure_files)}")
print(f"Non-seizure: {len(non_sampled)}")