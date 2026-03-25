"""Download dataset from Kaggle, skipping files that already exist locally."""
import os
import subprocess
import shutil

DATASET = "benedekbrandschott/halado-adatelemzes-orvosi-kepfeldolgozas-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/" + DATASET
EXPECTED_FILES = ["jpg_1024.zip"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datas")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    local_files = set(os.listdir(DATA_DIR))
    missing = [f for f in EXPECTED_FILES if f not in local_files]

    if not missing:
        print("All files already exist locally.")
        return

    print(f"Missing files: {missing}")

    # Try automatic download
    try:
        print(f"Downloading and extracting to {DATA_DIR}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR, "--unzip"],
            check=True,
        )
        print("Done.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"\nAutomatic download failed: {e}")
        print(f"\nPlease download manually from:\n  {KAGGLE_URL}")
        print(f"and place the files in:\n  {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
