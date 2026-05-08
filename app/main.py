import os
import sys
from itertools import product

from data_preprocessing import prepare_dataframe, split_data, TARGET_FINDINGS
from dataset import VinBigDataDataset
from augmentation import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
import mlflow
import torch
from training import train_model
from evaluation import evaluate
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
#from download_data import main as download_data


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("vinbigdata-classification")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datas")
    csv_path = os.path.join(data_dir, "annotations_1024.csv")
    image_dir = os.path.join(data_dir, "jpg_1024")

    #download_data()

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
    train_transforms = get_train_transforms(shift=True, scale=True, rotate=True, brightness=True)
    val_transforms = get_val_transforms()

    train_ds = VinBigDataDataset(train_df, image_dir, transforms=train_transforms)
    valid_ds = VinBigDataDataset(valid_df, image_dir, transforms=val_transforms)
    test_ds = VinBigDataDataset(test_df, image_dir, transforms=val_transforms)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Starting training...")

    search_space = {
        "model": ["resnet50", "densenet121"],
        "epoch": [200],
        "dropout": [0.0, 0.1, 0.3],
        "data_fraction": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        "seed": [42],
        "batch_size": [32],
        "lr": [1e-4],
        "wd": [1e-4]
    }
    
    keys = search_space.keys()
    configs = [dict(zip(keys, values)) for values in product(*search_space.values())]

    for cfg in configs:
        with mlflow.start_run():
            model = train_model(train_ds, valid_ds, cfg, device=device)

            print("Final evaluation on test set...")
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

            test_auc = evaluate(model, test_loader, device, log_prefix="test")
            print(f"Test AUC: {test_auc:.4f}")
            
    print("Done.")

if __name__ == "__main__":
    main()