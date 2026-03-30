#!/usr/bin/env python3
"""
scripts/train_baseline.py
--------------------------
Entry point: train EfficientNet-B3 baseline with 5-fold CV.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --config configs/config.yaml
    python scripts/train_baseline.py --folds 1  # single fold for quick test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.seed import set_global_seed
from src.data.dataset import load_dataframes
from src.training.train import run_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline model (EfficientNet-B3)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--folds", type=int, default=None, help="Override n_splits (e.g. 1 for quick debug)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs (e.g. 2 for quick test)")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override folds if specified
    if args.folds is not None:
        cfg["cross_validation"]["n_splits"] = args.folds

    # Override epochs if specified
    if args.epochs is not None:
        cfg["baseline"]["num_epochs"] = args.epochs

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected. Training on CPU will be very slow.")
    print(f"Device: {device}")

    # Seed
    set_global_seed(cfg["general"]["seed"])

    # Load data
    print("\nLoading data...")
    train_df, test_df = load_dataframes(
        train_features_path=cfg["data"]["train_features_path"],
        train_labels_path=cfg["data"]["train_labels_path"],
        test_features_path=cfg["data"]["test_features_path"],
    )

    # Run CV training
    print("\nStarting cross-validation training...")
    oof_preds, fold_scores = run_cv(
        train_df=train_df,
        images_dir=cfg["data"]["train_images_dir"],
        cfg=cfg,
        device=device,
        output_dir=Path(cfg["paths"]["model_dir"]),
    )

    print("\n✓ Training complete.")
    print(f"  Mean CV log-loss: {sum(fold_scores)/len(fold_scores):.4f}")


if __name__ == "__main__":
    main()
