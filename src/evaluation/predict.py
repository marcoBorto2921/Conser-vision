"""
src/evaluation/predict.py
--------------------------
Generate test set predictions using trained fold checkpoints.
Supports averaging across folds and optional TTA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WildlifeDataset, CLASS_NAMES
from src.data.transforms import get_val_transforms, get_tta_transforms
from src.models.model import build_model


@torch.no_grad()
def predict_single_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Get softmax probabilities from a single model checkpoint."""
    model.eval()
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="  Predicting", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    images_dir: str,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict with Test Time Augmentation: average over multiple views."""
    tta_transforms = get_tta_transforms(image_size)
    tta_probs: list[np.ndarray] = []

    _num_workers = 0 if (device.type != "cuda" or sys.platform == "win32") else 4
    _pin_memory = device.type == "cuda"

    for i, transform in enumerate(tta_transforms):
        print(f"  TTA augmentation {i+1}/{len(tta_transforms)}")
        ds = WildlifeDataset(test_df, images_dir, transform=transform, is_test=True)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=_num_workers, pin_memory=_pin_memory,
        )
        probs = predict_single_checkpoint(model, loader, device)
        tta_probs.append(probs)

    return np.mean(tta_probs, axis=0)


def generate_submission(
    test_df: pd.DataFrame,
    images_dir: str,
    checkpoint_dir: str | Path,
    cfg: dict,
    device: torch.device,
    use_tta: bool = True,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Generate final submission by averaging predictions across all fold checkpoints.

    Args:
        test_df: Test features DataFrame.
        images_dir: Directory containing test images.
        checkpoint_dir: Directory containing fold*.pth checkpoints.
        cfg: Config dict.
        device: Torch device.
        use_tta: Whether to use TTA.
        output_path: If provided, saves the submission CSV here.

    Returns:
        Submission DataFrame with columns [id, antelope_duiker, bird, ..., rodent].
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("fold*_best.pth"))
    print(f"Found {len(checkpoints)} fold checkpoints: {[c.name for c in checkpoints]}")

    model_cfg = cfg["baseline"]
    all_fold_probs: list[np.ndarray] = []

    for ckpt_path in checkpoints:
        print(f"\nLoading {ckpt_path.name}")
        model = build_model(
            model_name=model_cfg["model_name"],
            num_classes=cfg["general"]["num_classes"],
            pretrained=False,
            dropout=model_cfg["dropout"],
            checkpoint_path=str(ckpt_path),
        ).to(device)

        if use_tta:
            probs = predict_with_tta(
                model, test_df, images_dir,
                image_size=model_cfg["image_size"],
                device=device,
                batch_size=model_cfg["batch_size"],
            )
        else:
            ds = WildlifeDataset(
                test_df, images_dir,
                transform=get_val_transforms(model_cfg["image_size"]),
                is_test=True,
            )
            _num_workers = 0 if (device.type != "cuda" or sys.platform == "win32") else 4
            loader = DataLoader(
                ds, batch_size=model_cfg["batch_size"] * 2, shuffle=False,
                num_workers=_num_workers, pin_memory=device.type == "cuda",
            )
            probs = predict_single_checkpoint(model, loader, device)

        all_fold_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    # Average across folds
    final_probs = np.mean(all_fold_probs, axis=0)

    # Build submission DataFrame
    sub_df = pd.DataFrame({"id": test_df["id"].values})
    for i, cls in enumerate(CLASS_NAMES):
        sub_df[cls] = final_probs[:, i]

    # Verify probabilities sum to ~1
    row_sums = sub_df[CLASS_NAMES].sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        print(f"\n✓ Submission saved → {output_path}")
        print(f"  Shape: {sub_df.shape}")
        print(f"  Preview:\n{sub_df.head()}")

    return sub_df
