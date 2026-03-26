"""Download dataset from Kaggle, skipping files that already exist locally."""
import os
import subprocess
import threading

DATASET = "benedekbrandschott/halado-adatelemzes-orvosi-kepfeldolgozas-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/" + DATASET
EXPECTED_DATA = ["jpg_1024.zip", "jpg_1024"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datas")
TOTAL_SIZE_MB = 2693


def _monitor_download(data_dir, stop_event):
    """Print download progress every 10 seconds."""
    zip_name = DATASET.split("/")[-1] + ".zip"
    zip_path = os.path.join(data_dir, zip_name)
    while not stop_event.is_set():
        if os.path.exists(zip_path):
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            pct = size_mb / TOTAL_SIZE_MB * 100
            print(f"  {size_mb:.0f} / {TOTAL_SIZE_MB} MB ({pct:.0f}%)", flush=True)
        stop_event.wait(10)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    local_files = set(os.listdir(DATA_DIR))
    has_data = any(name in local_files for name in EXPECTED_DATA)

    found = [name for name in EXPECTED_DATA if name in local_files]
    if has_data:
        print(f"Data already exists locally: {found}")
        return

    print("No dataset found locally.")

    stop_event = threading.Event()
    zip_path = os.path.join(DATA_DIR, DATASET.split("/")[-1] + ".zip")

    try:
        # Download zip (without --unzip)
        print(f"Downloading {TOTAL_SIZE_MB} MB to {os.path.abspath(DATA_DIR)}...")
        monitor = threading.Thread(target=_monitor_download, args=(DATA_DIR, stop_event), daemon=True)
        monitor.start()

        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR],
            capture_output=True, check=True,
        )
        stop_event.set()
        print("Download complete.")

        # Extract with system unzip
        if os.path.exists(zip_path):
            # Count total files in zip
            result = subprocess.run(
                ["unzip", "-l", zip_path],
                capture_output=True, text=True,
            )
            total_files = 0
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and not line.startswith("---"):
                    total_files += 1

            print(f"Extracting {total_files} files...")
            proc = subprocess.Popen(
                ["unzip", "-o", zip_path, "-d", DATA_DIR],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            extracted = 0
            for line in proc.stdout:
                if line.strip().startswith("inflating:") or line.strip().startswith("extracting:"):
                    extracted += 1
                    if extracted % 500 == 0 or extracted == total_files:
                        pct = extracted / total_files * 100 if total_files else 0
                        print(f"  {extracted} / {total_files} files ({pct:.0f}%)", flush=True)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, "unzip")
            os.remove(zip_path)
            print("Done.")

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        stop_event.set()
        print(f"\nAutomatic download failed: {e}")
        print(f"\nPlease download manually from:\n  {KAGGLE_URL}")
        print(f"and place the files in:\n  {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
