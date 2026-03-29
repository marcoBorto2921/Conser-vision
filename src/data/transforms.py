"""
src/data/transforms.py
-----------------------
Image augmentation pipelines using torchvision.transforms.v2.
Designed for camera-trap images (fixed-position cameras, day/night variation).
"""

from __future__ import annotations

import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Strong augmentation pipeline for training.

    Camera-trap specific choices:
    - Vertical flip allowed (some traps can be upside down or at unusual angles)
    - ColorJitter with high brightness/contrast range (day vs night IR shots)
    - RandomGrayscale to simulate IR/night images
    - No heavy perspective distortion (camera is fixed)
    """
    return T.Compose([
        T.RandomResizedCrop(
            size=image_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.1),  # simulate IR/night camera images
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        T.ToImage(),
        T.ToDtype(T.torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for validation and test."""
    resize_size = int(image_size * (256 / 224))
    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToImage(),
        T.ToDtype(T.torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(image_size: int = 224) -> list[T.Compose]:
    """Test Time Augmentation: returns a list of deterministic transforms.

    Each transform represents one 'view' of the test image.
    Final prediction = mean of softmax probabilities across all views.
    """
    resize_size = int(image_size * (256 / 224))
    base = [
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        T.ToImage(),
        T.ToDtype(T.torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    crops = [
        T.CenterCrop(image_size),
        T.RandomCrop(image_size),  # top-left area
    ]

    tta_list = []
    for crop in crops:
        tta_list.append(T.Compose([T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC), crop] + base[1:]))
        tta_list.append(T.Compose([T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC), T.RandomHorizontalFlip(p=1.0), crop] + base[1:]))

    # original val transform
    tta_list.append(get_val_transforms(image_size))

    return tta_list
