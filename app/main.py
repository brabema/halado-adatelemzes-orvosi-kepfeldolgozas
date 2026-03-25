import os
import sys

from data_preprocessing import prepare_dataframe, split_data
from dataset import VinBigDataDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from download_data import main as download_data


def main():
    csv_path = "/app/data/train.csv"
    image_dir = "/app/data/train"

    download_data()

    print("Preparing dataframe...")
    clean_df = prepare_dataframe(csv_path)

    print("Splitting...")
    train_df, valid_df, test_df = split_data(clean_df)

    print("Creating datasets...")
    train_ds = VinBigDataDataset(train_df, image_dir)
    valid_ds = VinBigDataDataset(valid_df, image_dir)
    test_ds = VinBigDataDataset(test_df, image_dir)

    print("Dataset sizes:")
    print(len(train_ds), len(valid_ds), len(test_ds))


if __name__ == "__main__":
    main()