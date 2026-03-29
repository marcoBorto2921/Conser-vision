#!/usr/bin/env python3
"""
scripts/train_advanced.py
--------------------------
Advanced training: ConvNeXt-Base at 384px + ensemble with EfficientNet-B3.
Reads checkpoint_dir to find existing fold checkpoints from baseline
and generates ensemble OOF + test predictions.

Usage:
    python scripts/train_advanced.py
    python scripts/train_advanced.py --model convnext_base --image-size 384
    python scripts/train_advanced.py --model efficientnetv2_m --image-size 384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.seed import set_global_seed
from src.data.dataset import load_dataframes, CLASS_NAMES, NUM_CLASSES
from src.training.train import run_cv
from src.evaluation.predict import generate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model_name")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ensemble-with", type=str, default=None,
                        help="Path to OOF CSV from another model to ensemble with")
    return parser.parse_args()


def ensemble_oof(
    oof_a_path: str,
    oof_b_path: str,
    train_df: pd.DataFrame,
    id_col: str = "id",
) -> float:
    """Compute ensemble log-loss from two OOF prediction files."""
    oof_a = pd.read_csv(oof_a_path).set_index(id_col)[CLASS_NAMES].values
    oof_b = pd.read_csv(oof_b_path).set_index(id_col)[CLASS_NAMES].values

    oof_avg = (oof_a + oof_b) / 2.0
    true_classes = train_df[CLASS_NAMES].values.argmax(axis=1)
    ll = log_loss(true_classes, oof_avg, labels=list(range(NUM_CLASSES)))
    print(f"\nEnsemble OOF log-loss (avg A+B): {ll:.4f}")
    return ll


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    if args.model:
        cfg["advanced"]["model_name"] = args.model
    if args.image_size:
        cfg["advanced"]["image_size"] = args.image_size
        cfg["data"]["image_size_large"] = args.image_size
    if args.folds:
        cfg["cross_validation"]["n_splits"] = args.folds

    # Use 'advanced' block instead of 'baseline'
    cfg["baseline"] = cfg["advanced"]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Model: {cfg['baseline']['model_name']} @ {cfg['baseline']['image_size']}px")

    set_global_seed(cfg["general"]["seed"])

    train_df, test_df = load_dataframes(
        train_features_path=cfg["data"]["train_features_path"],
        train_labels_path=cfg["data"]["train_labels_path"],
        test_features_path=cfg["data"]["test_features_path"],
    )

    model_name = cfg["baseline"]["model_name"]
    output_dir = Path(cfg["paths"]["model_dir"]) / model_name

    oof_preds, fold_scores = run_cv(
        train_df=train_df,
        images_dir=cfg["data"]["train_images_dir"],
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )

    # Save this model's OOF
    oof_path = Path(cfg["paths"]["oof_dir"]) / f"oof_{model_name}.csv"
    oof_df = train_df[["id"]].copy()
    for i, cls in enumerate(CLASS_NAMES):
        oof_df[cls] = oof_preds[:, i]
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved → {oof_path}")

    # If ensemble partner provided, compute ensemble score
    if args.ensemble_with and Path(args.ensemble_with).exists():
        ensemble_oof(args.ensemble_with, str(oof_path), train_df)

    # Generate submission
    generate_submission(
        test_df=test_df,
        images_dir=cfg["data"]["test_images_dir"],
        checkpoint_dir=str(output_dir),
        cfg=cfg,
        device=device,
        use_tta=True,
        output_path=f"{cfg['paths']['submission_dir']}/submission_{model_name}_TTA.csv",
    )


if __name__ == "__main__":
    main()
