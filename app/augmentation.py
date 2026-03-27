import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(flip=True, rotate=True, brightness=True):
    transforms = []

    if flip:
        transforms.append(A.HorizontalFlip(p=0.5))

    if rotate:
        transforms.append(A.Rotate(limit=10, p=0.5))

    if brightness:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5
        ))

    transforms.append(A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
