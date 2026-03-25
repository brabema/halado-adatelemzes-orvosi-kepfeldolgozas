"""Download dataset from Kaggle, skipping files that already exist locally."""
import os
import subprocess

DATASET = "benedekbrandschott/halado-adatelemzes-orvosi-kepfeldolgozas-dataset"
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
    print(f"Downloading and extracting to {DATA_DIR}...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR, "--unzip"],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
