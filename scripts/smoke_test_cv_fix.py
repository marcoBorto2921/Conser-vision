#!/usr/bin/env python3
"""
scripts/smoke_test_cv_fix.py
-----------------------------
CPU smoke test for the site-aware CV fix.

Exercises the new StratifiedGroupKFold split end-to-end without running any
training. Asserts that every fold is site-disjoint, prints class distributions,
and loads one real mini-batch through the actual Dataset/DataLoader path.

Usage:
    python scripts/smoke_test_cv_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASS_NAMES, WildlifeDataset, load_dataframes
from src.data.transforms import get_val_transforms


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").open())
    seed = int(cfg["general"]["seed"])
    num_classes = int(cfg["general"]["num_classes"])

    train_df, _ = load_dataframes(
        train_features_path=str(ROOT / cfg["data"]["train_features_path"]),
        train_labels_path=str(ROOT / cfg["data"]["train_labels_path"]),
        test_features_path=str(ROOT / cfg["data"]["test_features_path"]),
    )

    assert "site" in train_df.columns, (
        "train_df must have a 'site' column — check load_dataframes preserved it."
    )

    y = train_df[CLASS_NAMES].values.argmax(axis=1)
    groups = train_df["site"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.arange(len(train_df)), y, groups))
    print(f"\nBuilt {len(splits)} folds with StratifiedGroupKFold on 'site' "
          f"(seed={seed})")

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        tr_groups = set(groups[tr_idx])
        va_groups = set(groups[va_idx])
        leaked = tr_groups & va_groups
        assert len(leaked) == 0, (
            f"GROUP LEAKAGE in fold {fold_idx}: {len(leaked)} sites shared"
        )

        tr_dist = np.bincount(y[tr_idx], minlength=num_classes) / len(tr_idx) * 100
        va_dist = np.bincount(y[va_idx], minlength=num_classes) / len(va_idx) * 100
        max_diff = float(np.abs(tr_dist - va_dist).max())

        print(
            f"Fold {fold_idx}: "
            f"train rows={len(tr_idx):>6}  val rows={len(va_idx):>6} | "
            f"train sites={len(tr_groups):>4}  val sites={len(va_groups):>4}  "
            f"leaked=0 | max class-dist diff={max_diff:.2f}%"
        )
        print(
            "         val class %: "
            + ", ".join(f"{c}={p:.1f}" for c, p in zip(CLASS_NAMES, va_dist))
        )

    # --- Real Dataset/DataLoader mini-batch test (fold 0, 4 samples only) ---
    tr_idx, va_idx = splits[0]
    tiny_val_df = train_df.iloc[va_idx[:4]].reset_index(drop=True)
    tiny_ds = WildlifeDataset(
        df=tiny_val_df,
        images_dir=str(ROOT / cfg["data"]["train_images_dir"]),
        transform=get_val_transforms(cfg["baseline"]["image_size"]),
    )
    tiny_loader = DataLoader(tiny_ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(tiny_loader))

    img: torch.Tensor = batch["image"]
    label: torch.Tensor = batch["label"]
    expected_hw = int(cfg["baseline"]["image_size"])
    assert img.shape == (2, 3, expected_hw, expected_hw), (
        f"unexpected image shape: {tuple(img.shape)}"
    )
    assert label.shape == (2, num_classes), f"unexpected label shape: {tuple(label.shape)}"
    assert img.dtype == torch.float32, f"unexpected image dtype: {img.dtype}"
    print(
        f"\nMini-batch OK: images={tuple(img.shape)} dtype={img.dtype}, "
        f"labels={tuple(label.shape)}"
    )

    print("\nSMOKE TEST PASS")


if __name__ == "__main__":
    main()
