# dataset.py
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE

def make_basic_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

class ImageMaskDataset(Dataset):
    """
    Expects two lists: images and masks (same order). Masks are integer masks with values 0..C-1.
    """
    def __init__(self, image_paths, mask_paths, transforms=None):
        assert len(image_paths) == len(mask_paths)
        self.images = image_paths
        self.masks = mask_paths
        self.transforms = transforms if transforms is not None else make_basic_augs()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])[:, :, ::-1]  # BGR->RGB
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Mask not found or invalid: {self.masks[idx]}")
        augmented = self.transforms(image=img, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()
        return image, mask

def collect_pairs(img_dir, mask_dir, img_exts=(".jpg",".png",".jpeg")):
    imgs = []
    masks = []
    for fn in sorted(os.listdir(img_dir)):
        if fn.lower().endswith(img_exts):
            imgp = os.path.join(img_dir, fn)
            maskp = os.path.join(mask_dir, os.path.splitext(fn)[0] + ".png")
            if os.path.exists(maskp):
                imgs.append(imgp); masks.append(maskp)
    return imgs, masks
