import os
import subprocess

DATASET = "benedekbrandschott/halado-adatelemzes-orvosi-kepfeldolgozas-dataset"
DATA_DIR = "/app/data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if any(f.endswith(".zip") for f in os.listdir(DATA_DIR)):
        print("Dataset already downloaded.")
        return

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR],
        check=True,
    )

    print("Download complete.")


if __name__ == "__main__":
    main()