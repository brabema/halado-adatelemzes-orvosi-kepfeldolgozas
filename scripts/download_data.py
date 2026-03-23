"""Download dataset from Kaggle."""
import os
import subprocess
import sys

DATASET = "benedekbrandschott/halado-adatelemzes-orvosi-kepfeldolgozas-dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datas")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if data already exists
    zips = [f for f in os.listdir(DATA_DIR) if f.endswith(".zip")]
    if zips:
        print(f"Data already exists in {DATA_DIR}: {zips}")
        print("Delete them first if you want to re-download.")
        return

    print(f"Downloading dataset to {DATA_DIR}...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
