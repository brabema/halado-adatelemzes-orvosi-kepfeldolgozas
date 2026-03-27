import os
import sys

from data_preprocessing import prepare_dataframe, split_data, TARGET_FINDINGS
from dataset import VinBigDataDataset
from augmentation import get_train_transforms, get_val_transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from download_data import main as download_data


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datas")
    csv_path = os.path.join(data_dir, "train.csv")
    image_dir = os.path.join(data_dir, "jpg_1024")

    download_data()

    print("Preparing dataframe...")
    clean_df = prepare_dataframe(csv_path)

    print("Splitting...")
    train_df, valid_df, test_df = split_data(clean_df)
    print(f"  Train: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}")

    print("Finding arányok halmazonként (%):")
    header = f"  {'Finding':<25} {'Train':>7} {'Valid':>7} {'Test':>7}"
    print(header)
    for f in TARGET_FINDINGS:
        t = train_df[f].mean() * 100
        v = valid_df[f].mean() * 100
        te = test_df[f].mean() * 100
        print(f"  {f:<25} {t:>6.1f}% {v:>6.1f}% {te:>6.1f}%")

    print("Creating datasets...")
    train_transforms = get_train_transforms(flip=True, rotate=True, brightness=True)
    val_transforms = get_val_transforms()

    train_ds = VinBigDataDataset(train_df, image_dir, transforms=train_transforms)
    valid_ds = VinBigDataDataset(valid_df, image_dir, transforms=val_transforms)
    test_ds = VinBigDataDataset(test_df, image_dir, transforms=val_transforms)

    print("Done.")


if __name__ == "__main__":
    main()