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
from src.evaluation.eval import calibrate_temperature
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
def predict_logits_single_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Get raw logits (pre-softmax) from a single model checkpoint."""
    model.eval()
    all_logits: list[np.ndarray] = []

    for batch in tqdm(loader, desc="  Predicting logits", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu().numpy().astype(np.float32))

    return np.concatenate(all_logits)


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    images_dir: str,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
    temperature: float | None = None,
) -> np.ndarray:
    """Predict with Test Time Augmentation: average over multiple views.

    When temperature is provided, collects raw logits per TTA view, averages
    them in logit space, then applies softmax(avg_logits / temperature).
    Otherwise averages softmax probabilities directly (legacy behaviour, T=1.0).
    """
    tta_transforms = get_tta_transforms(image_size)

    _num_workers = 0 if (device.type != "cuda" or sys.platform == "win32") else 4
    _pin_memory = device.type == "cuda"

    if temperature is not None:
        # Collect logits per TTA view, average in logit space, then apply softmax with T
        tta_logits: list[np.ndarray] = []
        for i, transform in enumerate(tta_transforms):
            print(f"  TTA augmentation {i+1}/{len(tta_transforms)}")
            ds = WildlifeDataset(test_df, images_dir, transform=transform, is_test=True)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=_num_workers, pin_memory=_pin_memory,
            )
            logits = predict_logits_single_checkpoint(model, loader, device)
            tta_logits.append(logits)

        avg_logits = np.mean(tta_logits, axis=0)
        scaled = torch.from_numpy((avg_logits / temperature).astype(np.float32))
        return F.softmax(scaled, dim=1).numpy()
    else:
        # Legacy: average probabilities across TTA views
        tta_probs: list[np.ndarray] = []
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

    For each checkpoint, loads saved val_logits/val_labels (if present) to calibrate
    temperature T via NLL minimisation. Applies softmax(logits / T) for test predictions.
    Falls back to T=1.0 with a warning for old checkpoints without calibration data.

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
    if not checkpoints:
        raise FileNotFoundError(
            f"No fold checkpoints found in {checkpoint_dir}. "
            "Run training first."
        )

    model_cfg = cfg["baseline"]
    all_fold_probs: list[np.ndarray] = []

    for ckpt_path in checkpoints:
        print(f"\nLoading {ckpt_path.name}")

        # Load checkpoint dict to retrieve val_logits / val_labels for calibration
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        if "val_logits" in ckpt_dict and "val_labels" in ckpt_dict:
            val_logits: np.ndarray = ckpt_dict["val_logits"]
            val_labels: np.ndarray = ckpt_dict["val_labels"]
            temperature = calibrate_temperature(val_logits, val_labels)
        else:
            print(
                f"  WARNING: checkpoint {ckpt_path.name} has no val_logits/val_labels. "
                "Falling back to T=1.0 (no calibration)."
            )
            temperature = 1.0

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
                temperature=temperature,
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
            raw_logits = predict_logits_single_checkpoint(model, loader, device)
            scaled = torch.from_numpy((raw_logits / temperature).astype(np.float32))
            probs = F.softmax(scaled, dim=1).numpy()

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
        print(f"\nSubmission saved -> {output_path}")
        print(f"  Shape: {sub_df.shape}")
        print(f"  Preview:\n{sub_df.head()}")

    return sub_df
