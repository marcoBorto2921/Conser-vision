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
| Train | Images + one-hot labels across 8 classes |
| Test  | Images only, predict class probabilities |

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
| Baseline (local CV) | — | EfficientNet-B3 defaults |
| Best local CV | — | — |
| Public LB | — | — |
| Private LB | — | — |

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data from DrivenData (after joining competition)
# Unzip into data/raw/:
#   train_features.csv
#   train_labels.csv
#   test_features.csv
#   train_features/   ← folder with training images
#   test_features/    ← folder with test images

# 3. Train baseline (5-fold CV)
python scripts/train_baseline.py

# 4. Quick debug (1 fold)
python scripts/train_baseline.py --folds 1

# 5. Generate submission (with TTA)
python scripts/predict.py

# 6. Generate submission (without TTA, faster)
python scripts/predict.py --no-tta
```

---

## Key Learnings

_To be filled in during/after competition._
