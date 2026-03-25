import os
import numpy as np
import torch
import pydicom
import cv2
from torch.utils.data import Dataset

from data_preprocessing import TARGET_FINDINGS


def read_dicom_image(file_path, target_size=(512, 512)):
    dicom = pydicom.dcmread(file_path)
    data = dicom.pixel_array

    if 'WindowCenter' in dicom and 'WindowWidth' in dicom:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        data = apply_voi_lut(dicom.pixel_array, dicom)

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / (np.max(data) + 1e-6)

    data = (data * 255).astype(np.uint8)
    data = cv2.resize(data, target_size)
    data = np.stack([data, data, data], axis=-1)

    return data


class VinBigDataDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = self.df[TARGET_FINDINGS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, 'image_id']
        file_path = os.path.join(self.image_dir, f"{image_id}.dicom")

        image = read_dicom_image(file_path)

        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label