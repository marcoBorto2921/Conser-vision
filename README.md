# Conser-vision: Wildlife Image Classification

> [DrivenData competition link](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)

---

## Overview

| Field | Details |
|-------|---------|
| **Task** | Multi-class image classification (8 classes) |
| **Metric** | Log-loss (lower is better) |
| **Data** | Camera-trap images, Taï National Park |
| **Deadline** | ~1 year remaining |
| **Status** | 🟢 Active |

### Classes
`antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`

### Problem Description
Classify wildlife species appearing in camera-trap images from conservation research at Taï National Park (Wild Chimpanzee Foundation + Max Planck Institute). Images include day/night (IR) shots, varying animal positions and distances.

---

## Dataset

| Split | Notes |
|-------|-------|
| Train | 16,488 images + one-hot labels across 8 classes |
| Test  | 4,464 images only, predict class probabilities |

**Key challenges**:
- Class imbalance (blank images likely dominant)
- Day vs. night infrared images (grayscale-ish)
- Small/distant animals vs. close-up shots

---

## Approach

1. **Baseline**: EfficientNet-B3 (timm) + StratifiedKFold(5) + label smoothing
2. **Advanced**: ConvNeXt-Base ensemble + TTA (5 views) + probability averaging across folds
3. **Augmentation**: ColorJitter (day/night), RandomGrayscale (IR simulation), flips, rotation

---

## Results

| Stage | Log-loss | Notes |
|-------|----------|-------|
| Fold 1 local CV | 0.8454 | EfficientNet-B3, epoch 29 |
| Public LB | 1.8334 | Single fold, no hyperparameter tuning — rank 333/2,060 |
| Private LB | — | — |

---

## Setup & Run
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data from DrivenData (after joining competition)
# Unzip into data/raw/:
#   train_features.csv
#   train_labels.csv
#   test_features.csv
#   train_features/   ← folder with training images
#   test_features/    ← folder with test images

# 4. Train baseline (5-fold CV)
python scripts/train_baseline.py

# 5. Quick debug (1 fold)
python scripts/train_baseline.py --folds 1

# 6. Generate submission (with TTA)
python scripts/predict.py

# 7. Generate submission (without TTA, faster)
python scripts/predict.py --no-tta
```

---

## Key Learnings

- Copying data from Google Drive to Colab RAM before training gives ~10x speed improvement
- `subprocess.Popen` with `PYTHONUNBUFFERED=1` is required for real-time output streaming in Colab
- Gap between local CV (0.8454) and public LB (1.8334) suggests the model is overconfident on test — TTA and multi-fold ensemble expected to close this gap significantly
