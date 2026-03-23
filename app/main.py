import os

from data_preprocessing import prepare_dataframe, split_data
from dataset import VinBigDataDataset

def main():
    csv_path = "/app/data/train.csv"
    image_dir = "/app/data/train"

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