#!/usr/bin/env python3
"""
scripts/predict.py
-------------------
Generate submission CSV from trained fold checkpoints.

Usage:
    python scripts/predict.py
    python scripts/predict.py --no-tta
    python scripts/predict.py --checkpoint-dir models/weights --output submissions/sub_v1.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_dataframes
from src.evaluation.predict import generate_submission
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate competition submission")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    set_global_seed(cfg["general"]["seed"])

    _, test_df = load_dataframes(
        train_features_path=cfg["data"]["train_features_path"],
        train_labels_path=cfg["data"]["train_labels_path"],
        test_features_path=cfg["data"]["test_features_path"],
    )

    checkpoint_dir = args.checkpoint_dir or cfg["paths"]["model_dir"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tta_tag = "noTTA" if args.no_tta else "TTA"
    output_path = args.output or f"{cfg['paths']['submission_dir']}/submission_{tta_tag}_{timestamp}.csv"

    generate_submission(
        test_df=test_df,
        images_dir=cfg["data"]["test_images_dir"],
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        device=device,
        use_tta=not args.no_tta,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
