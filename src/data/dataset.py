"""
src/data/dataset.py
--------------------
PyTorch Dataset for camera-trap image classification.
Handles train/val (with labels) and test (without labels) splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = [
    "antelope_duiker",
    "bird",
    "blank",
    "civet_genet",
    "hog",
    "leopard",
    "monkey_prosimian",
    "rodent",
]

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


class WildlifeDataset(Dataset):
    """Camera-trap image dataset for Conser-vision competition.

    Args:
        df: DataFrame with at minimum an 'id' column and optionally label columns.
        images_dir: Root directory containing image files.
        transform: Torchvision/albumentations transform to apply.
        is_test: If True, __getitem__ does not return labels.
        id_col: Name of the ID column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        transform: Optional[Callable] = None,
        is_test: bool = False,
        id_col: str = "id",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_test = is_test
        self.id_col = id_col

        # Build label array for training
        if not is_test:
            self.labels: np.ndarray = self.df[CLASS_NAMES].values.astype(np.float32)

        # Build file paths
        # DrivenData stores images with IDs matching the 'id' column
        self.image_ids: list[str] = self.df[id_col].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, idx: int) -> Image.Image:
        image_id = self.image_ids[idx]
        # DrivenData naming: images are stored as <id>.jpg
        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            # Try .JPG or .jpeg as fallback
            for ext in [".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                alt = self.images_dir / f"{image_id}{ext}"
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found for id='{image_id}'. "
                f"Tried: {self.images_dir / (str(image_id) + '.jpg')} and common variants."
            )
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> dict:
        img = self._load_image(idx)

        if self.transform is not None:
            # Support both torchvision transforms and albumentations
            if type(self.transform).__module__.startswith("albumentations"):
                # albumentations
                transformed = self.transform(image=np.array(img))
                img_tensor: torch.Tensor = transformed["image"]
            else:
                # torchvision
                img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        sample: dict = {
            "image": img_tensor,
            "id": self.image_ids[idx],
        }

        if not self.is_test:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample


def load_dataframes(
    train_features_path: str,
    train_labels_path: str,
    test_features_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge train/test DataFrames.

    Returns:
        train_df: Merged features + labels DataFrame.
        test_df: Test features DataFrame.
    """
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(test_features_path)

    train_df = train_features.merge(train_labels, on="id", how="inner")

    print(f"Train: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")
    print(f"Train label distribution:\n{train_labels[CLASS_NAMES].sum()}")

    return train_df, test_df
