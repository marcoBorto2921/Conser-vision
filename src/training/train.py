"""
src/training/train.py
----------------------
Training loop with StratifiedKFold cross-validation.
Supports mixed-precision, early stopping, and checkpointing.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WildlifeDataset, CLASS_NAMES, NUM_CLASSES
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model import build_model
from utils.seed import set_global_seed


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch.

    Samples a mixing coefficient lam ~ Beta(alpha, alpha) and returns
    a linearly interpolated image and the two original label tensors.

    Args:
        x: Image batch tensor, shape (B, C, H, W).
        y: Label tensor (one-hot or class indices), shape (B, ...).
        alpha: Beta distribution concentration parameter.

    Returns:
        Tuple of (mixed_x, y_a, y_b, lam) where lam is the scalar mixing coefficient.
    """
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    gradient_clip: float = 1.0,
    mixup_alpha: float = 0.4,
    mixup_prob: float = 0.0,
) -> dict[str, float]:
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=scaler.is_enabled()):
            if random.random() < mixup_prob:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                logits = model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return {"loss": total_loss / n_samples}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validation epoch. Returns loss and OOF predictions."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="  Val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        raw_logits = logits.cpu().numpy().astype(np.float32)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_logits.append(raw_logits)
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    all_probs_np = np.concatenate(all_probs)
    all_logits_np = np.concatenate(all_logits)
    all_labels_np = np.concatenate(all_labels)
    # Convert one-hot to class indices for log_loss
    true_classes = all_labels_np.argmax(axis=1)
    ll = log_loss(true_classes, all_probs_np, labels=list(range(NUM_CLASSES)))
    label_indices = all_labels_np.argmax(axis=1)  # shape (n,) integer class indices

    return {
        "loss": total_loss / n_samples,
        "log_loss": ll,
        "probs": all_probs_np,
        "logits": all_logits_np,
        "labels": all_labels_np,       # one-hot, shape (n, 8) — kept for backward compat
        "label_indices": label_indices, # integer class indices, shape (n,)
    }


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    images_dir: str,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    resume_checkpoint: Optional[Path] = None,
) -> tuple[float, np.ndarray]:
    """Train a single fold and return (best_val_log_loss, oof_predictions)."""
    print(f"\n{'='*60}", flush=True)
    print(f"  FOLD {fold+1}", flush=True)
    print(f"{'='*60}", flush=True)

    set_global_seed(cfg["general"]["seed"] + fold)
    model_cfg = cfg["baseline"]

    # Datasets & loaders
    train_ds = WildlifeDataset(
        train_df, images_dir,
        transform=get_train_transforms(model_cfg["image_size"])
    )
    val_ds = WildlifeDataset(
        val_df, images_dir,
        transform=get_val_transforms(model_cfg["image_size"])
    )
    # num_workers=0 on CPU/Windows (multiprocessing overhead is worse than single-process)
    _num_workers = 0 if device.type != "cuda" else 2
    train_loader = DataLoader(
        train_ds, batch_size=model_cfg["batch_size"],
        shuffle=True, num_workers=_num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=model_cfg["batch_size"] * 2,
        shuffle=False, num_workers=_num_workers, pin_memory=device.type == "cuda"
    )

    # Model
    model = build_model(
        model_name=model_cfg["model_name"],
        num_classes=cfg["general"]["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    ).to(device)

    # Optimizer with differential LR
    param_groups = model.get_optimizer_param_groups(
        lr=model_cfg["learning_rate"],
        lr_backbone_multiplier=0.1,
    )
    optimizer = AdamW(
        param_groups,
        weight_decay=model_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=model_cfg["num_epochs"],
        eta_min=1e-6,
    )

    # Loss: soft cross-entropy with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=model_cfg["label_smoothing"])
    scaler = GradScaler("cuda", enabled=model_cfg["mixed_precision"] and device.type == "cuda")

    best_log_loss = float("inf")
    best_oof: Optional[np.ndarray] = None
    patience_counter = 0
    patience = model_cfg["early_stopping_patience"]
    start_epoch = 0

    if resume_checkpoint is not None and resume_checkpoint.exists():
        print(f"  Resuming from {resume_checkpoint}", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_log_loss = ckpt["best_score"]
        patience_counter = ckpt.get("patience_counter", 0)
        print(
            f"  Resumed: starting at epoch {start_epoch + 1}, "
            f"best log-loss so far: {best_log_loss:.4f}, "
            f"patience: {patience_counter}/{patience}",
            flush=True,
        )
        # Recover best_oof from the best checkpoint so OOF predictions are
        # always from the best model, not from the (potentially worse) last state.
        best_ckpt_path = output_dir / f"fold{fold+1}_best.pth"
        print("  Running initial validation to recover OOF predictions...", flush=True)
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            best_oof = validate(model, val_loader, criterion, device)["probs"]
            # Restore the last (training) model state for continued training
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            best_oof = validate(model, val_loader, criterion, device)["probs"]
        if start_epoch >= model_cfg["num_epochs"]:
            print("  All epochs already completed — skipping fold.", flush=True)
            return best_log_loss, best_oof

    for epoch in range(start_epoch, model_cfg["num_epochs"]):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            gradient_clip=model_cfg["gradient_clip"],
            mixup_alpha=model_cfg.get("mixup_alpha", 0.4),
            mixup_prob=model_cfg.get("mixup_prob", 0.0),
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch+1:3d}/{model_cfg['num_epochs']} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val log-loss: {val_metrics['log_loss']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s",
            flush=True,
        )

        if val_metrics["log_loss"] < best_log_loss:
            best_log_loss = val_metrics["log_loss"]
            best_oof = val_metrics["probs"]
            patience_counter = 0
            checkpoint_path = output_dir / f"fold{fold+1}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_score": best_log_loss,
                    "val_log_loss": best_log_loss,
                    "patience_counter": patience_counter,
                    "fold": fold,
                    "val_logits": val_metrics["logits"],
                    "val_labels": val_metrics["label_indices"],
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best checkpoint → {checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}", flush=True)
                # Save last checkpoint before exiting so resume works
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_score": best_log_loss,
                        "patience_counter": patience_counter,
                        "fold": fold,
                    },
                    output_dir / f"fold{fold+1}_last.pth",
                )
                break

        # Save last checkpoint every epoch for resume support
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_log_loss,
                "patience_counter": patience_counter,
                "fold": fold,
            },
            output_dir / f"fold{fold+1}_last.pth",
        )

    print(f"\n  Fold {fold+1} best log-loss: {best_log_loss:.4f}", flush=True)
    return best_log_loss, best_oof


def run_cv(
    train_df: pd.DataFrame,
    images_dir: str,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    resume: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Run full StratifiedKFold cross-validation.

    Returns:
        oof_preds: OOF predictions array of shape (n_train, num_classes).
        fold_scores: List of per-fold log-loss values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_splits = cfg["cross_validation"]["n_splits"]

    # Stratify on the class with highest probability (one-hot)
    target_series = train_df[CLASS_NAMES].idxmax(axis=1)

    oof_preds = np.zeros((len(train_df), cfg["general"]["num_classes"]), dtype=np.float32)
    fold_scores: list[float] = []
    all_val_idx: np.ndarray = np.array([], dtype=np.intp)

    # n_splits=1 → single 80/20 stratified holdout (quick debug mode)
    if n_splits == 1:
        all_idx = np.arange(len(train_df))
        train_idx, val_idx = train_test_split(
            all_idx, test_size=0.2, stratify=target_series,
            random_state=cfg["general"]["seed"],
        )
        splits = [(train_idx, val_idx)]
    else:
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=cfg["cross_validation"]["shuffle"],
            random_state=cfg["general"]["seed"],
        )
        splits = list(skf.split(train_df, target_series))

    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        resume_checkpoint: Optional[Path] = None
        if resume:
            last_ckpt = output_dir / f"fold{fold+1}_last.pth"
            if last_ckpt.exists():
                resume_checkpoint = last_ckpt
            else:
                print(f"  No resume checkpoint found for fold {fold+1}, starting from scratch.", flush=True)

        fold_log_loss, fold_oof = train_fold(
            fold=fold,
            train_df=fold_train_df,
            val_df=fold_val_df,
            images_dir=images_dir,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            resume_checkpoint=resume_checkpoint,
        )

        oof_preds[val_idx] = fold_oof
        fold_scores.append(fold_log_loss)
        all_val_idx = np.concatenate([all_val_idx, val_idx])

    overall_ll = log_loss(
        train_df[CLASS_NAMES].values.argmax(axis=1)[all_val_idx],
        oof_preds[all_val_idx],
        labels=list(range(cfg["general"]["num_classes"])),
    )
    print(f"\n{'='*60}", flush=True)
    print("  CV Results:", flush=True)
    for i, s in enumerate(fold_scores):
        print(f"    Fold {i+1}: {s:.4f}", flush=True)
    print(f"  Mean fold log-loss: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}", flush=True)
    print(f"  OOF log-loss:       {overall_ll:.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    # Save OOF predictions
    oof_df = train_df[["id"]].copy()
    for i, cls in enumerate(CLASS_NAMES):
        oof_df[cls] = oof_preds[:, i]
    oof_path = Path(cfg["paths"]["oof_dir"]) / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF predictions saved → {oof_path}", flush=True)

    return oof_preds, fold_scores
