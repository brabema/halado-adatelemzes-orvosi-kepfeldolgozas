import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from data_preprocessing import TARGET_FINDINGS
from augmentation import get_val_transforms


def read_image(file_path, target_size=(512, 512)):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {file_path}")
    img = cv2.resize(img, target_size)
    img = np.stack([img, img, img], axis=-1)
    return img


class VinBigDataDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None, ext=".jpg"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.ext = ext
        self.labels = self.df[TARGET_FINDINGS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, 'image_id']
        file_path = os.path.join(self.image_dir, f"{image_id}{self.ext}")

        image = read_image(file_path)

        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = get_val_transforms()(image=image)['image']

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
