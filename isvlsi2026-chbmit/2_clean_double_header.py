import os

DATASET_PATH = "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/EEG_Classification/ws_2/edge_impulse_data2/balanced"

def fix_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return False

    # detect unwanted second header
    if "ch1" in lines[1].lower():
        lines.pop(1)  # remove second row

        with open(file_path, "w") as f:
            f.writelines(lines)

        return True

    return False


def process_all():
    fixed = 0

    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                if fix_file(path):
                    fixed += 1

    print(f"✅ Fixed {fixed} files")


if __name__ == "__main__":
    process_all()