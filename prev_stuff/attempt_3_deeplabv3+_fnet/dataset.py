import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from config import HEIGHT, WIDTH, CLASS_IDS


def remap_mask(mask):
    remapped = np.full_like(mask, fill_value=-1)

    for idx, class_id in enumerate(CLASS_IDS):
        remapped[mask == class_id] = idx

    if (remapped == -1).any():
        raise ValueError("Mask contains unknown class values")

    return remapped


class OffroadDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 is_train_dataset=True,
                 perform_augmentation=False):

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

        self.mask_paths = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.endswith(".png")
        ])

        if is_train_dataset and perform_augmentation:
            self.transform = A.Compose([
                A.Resize(HEIGHT, WIDTH),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(HEIGHT, WIDTH),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        mask = mask.astype(np.int32)
        mask = remap_mask(mask)

        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), \
            torch.tensor(mask, dtype=torch.long)
