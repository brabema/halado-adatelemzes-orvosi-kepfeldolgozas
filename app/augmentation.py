import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(shift=True, scale=True, rotate=True, brightness=True):
    transforms = []

    if shift or scale or rotate:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5,
                border_mode=0
            )
        )

    if brightness:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            )
        )

    transforms.append(
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
