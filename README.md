# Conser-vision: Wildlife Image Classification

**Competition**: [DrivenData #87 — Conser-vision Practice Area: Image Classification](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)
**Task**: Multi-class image classification across 8 wildlife species
**Metric**: Logarithmic loss (lower is better)
**Status**: Active — iterating toward top 10%

---

## Problem Statement

Camera traps are motion-triggered devices deployed by conservation researchers to passively record wildlife without human interference. They generate far more footage than can be manually reviewed. This competition, built around data from the Wild Chimpanzee Foundation and the Max Planck Institute for Evolutionary Anthropology at Taï National Park (Ivory Coast), asks participants to automate species identification from these images.

The task is 8-class image classification. Each image must be assigned a probability distribution over the following classes:

`antelope_duiker` · `bird` · `blank` · `civet_genet` · `hog` · `leopard` · `monkey_prosimian` · `rodent`

The evaluation metric is log-loss, which penalises confident wrong predictions sharply. A model that is accurate but poorly calibrated will score worse than a model that is slightly less accurate but well-calibrated. This makes probability calibration a first-class concern alongside raw classification accuracy.

---

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| Train | 16,488 | One-hot across 8 classes |
| Test  | 4,464  | Unlabelled — predict class probabilities |

Images vary significantly in visual characteristics:

- **Day vs. night**: daytime images are full-colour; nighttime images captured via infrared are near-grayscale
- **Animal size and distance**: close-up animals vs. distant silhouettes at the edge of the frame
- **Blank images**: a large proportion of images contain no animal — background motion triggers the camera
- **Occlusion and motion blur**: animals partially hidden by vegetation, or blurred from movement

The class distribution is imbalanced, with `blank` being the most frequent class by a significant margin.

---

## Approach

### Pipeline overview

The pipeline is fully reproducible: all hyperparameters are stored in `configs/config.yaml`, training uses a fixed seed, and model weights are checkpointed to Google Drive for persistence across Colab sessions.

```
raw images → augmentation → EfficientNet backbone → dropout → linear head → softmax
                                                                          ↓
                                                            temperature scaling (post-hoc)
                                                                          ↓
                                                              submission probabilities
```

### Model

**Current**: EfficientNetV2-M pretrained on ImageNet-21k, loaded via `timm` (`tf_efficientnetv2_m`). 54M parameters, ImageNet-1k top-1 85.2%. Fine-tuned end-to-end with a differential learning rate (backbone at lr×0.1, head at full lr).

**Head architecture**: `BatchNorm1d → Dropout(0.3) → Linear(num_features, 8)`

The BatchNorm before the classifier normalises feature magnitudes across the batch, making training less sensitive to the initial learning rate. Dropout at 0.3 is a conservative regulariser appropriate for a dataset of ~16k images where overfitting is a real risk.

**Next step (planned)**: ConvNeXt-Base ensemble at 384px resolution.

### Training setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Correct weight decay decoupling vs. Adam |
| Learning rate | 1e-4 (head), 1e-5 (backbone) | Differential LR: backbone already converged on ImageNet |
| Scheduler | CosineAnnealingLR | Smooth convergence, no manual step tuning |
| Loss | CrossEntropyLoss + label smoothing 0.1 | Reduces overconfidence, improves log-loss directly |
| Epochs | 60 (early stopping patience 10) | Full convergence; early stopping prevents overfitting |
| Batch size | 32 | Stable on T4 16GB at 224px |
| Image size | 224px | Baseline; 384px planned for next experiment |
| Mixed precision | torch.cuda.amp | ~40% memory reduction, same convergence |
| CV strategy | Single fold (80/20 stratified) | Fast iteration; 5-fold reserved for final ensemble |

### Augmentation

**Training**:
- `RandomResizedCrop(224, scale=(0.6, 1.0))` — handles animals at varying distances
- `HorizontalFlip(p=0.5)` — valid; animals traverse the frame in both directions
- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)` — simulates varying daylight conditions
- `RandomGrayscale(p=0.1)` — simulates nighttime infrared images
- `GaussianBlur(p=0.2)` — simulates motion blur from moving animals
- `MixUp(alpha=0.4, p=0.5)` — soft label mixing, reduces overconfidence
- ImageNet normalisation

**Validation / Test**:
- `Resize(256) → CenterCrop(224)`
- ImageNet normalisation

`RandomErasing` was deliberately excluded: it risks erasing the only animal in the frame, which is particularly harmful on small or distant subjects.

### Post-hoc calibration: temperature scaling

Log-loss is sensitive to probability calibration. A model trained with cross-entropy is well-calibrated on its training set but typically overconfident on unseen data. Temperature scaling addresses this with a single learnable parameter T:

```
p_calibrated = softmax(logits / T)
```

T is optimised on the held-out validation set by minimising NLL using `scipy.optimize.minimize_scalar`. Typical optimal values for overconfident networks fall in the range T ∈ [1.5, 2.5]. The same T is then applied at inference time before generating submission probabilities.

### Test Time Augmentation (TTA)

At inference, each test image is passed through 5 augmented views: center crop, two random crops, and horizontal flips of each. The softmax probabilities are averaged across views to reduce prediction variance.

```
p_final = (1/5) * Σ softmax(f(aug_i(x)) / T)
```

---

## Results

| Submission | Model | Epochs | Image size | LB log-loss | Notes |
|------------|-------|--------|------------|-------------|-------|
| v1 | EfficientNet-B3 | 29 | 224px | 1.8334 | Single fold, no temperature scaling, no TTA |
| v2 | EfficientNet-B3 | 60 | 224px | — | Temperature scaling + MixUp — in progress |
| v3 | EfficientNetV2-M | — | 384px | — | Planned |

**Local CV (fold 1)**: 0.8454 log-loss at epoch 29.

The gap between local CV (0.8454) and public LB (1.8334) is primarily caused by overconfidence on the test distribution. Temperature scaling is expected to close a significant portion of this gap without any retraining.

---

## Repository Structure

```
conser-vision/
├── configs/
│   └── config.yaml            # All hyperparameters — never hardcoded in scripts
├── data/
│   ├── raw/                   # Downloaded from DrivenData (gitignored)
│   └── processed/             # Preprocessed splits
├── notebooks/
│   ├── 01_eda.ipynb           # Class distribution, day/night analysis, sample images
│   └── colab_runner.ipynb     # Colab entry point: clone repo, mount Drive, train
├── scripts/
│   ├── train_baseline.py      # Entry point: single-fold training
│   ├── train_advanced.py      # Entry point: ensemble training
│   └── predict.py             # Generates submission CSV with TTA + temperature scaling
├── src/
│   ├── data/dataset.py        # PyTorch Dataset, transforms, DataLoader factory
│   ├── models/model.py        # EfficientNet/ConvNeXt wrapper via timm
│   ├── training/train.py      # Training loop with AMP, MixUp, early stopping
│   └── evaluation/eval.py     # Log-loss, temperature calibration, OOF diagnostics
├── submissions/               # Output CSVs (gitignored)
├── CLAUDE.md                  # Claude Code instructions for autonomous scaffolding
├── TECHNICAL_CHOICES.md       # Justification of every major technical decision
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (training on CPU is possible but slow)
- DrivenData account to download competition data

### Local setup

```bash
git clone https://github.com/marcoBorto2921/Conser-vision.git
cd Conser-vision

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Verify setup
which python                    # must point to .venv/bin/python
python -c "import torch, timm; print(torch.__version__)"
```

### Data download

After registering for the competition on DrivenData:

```bash
pip install drivendata-download
drd download 87 --output data/raw/
```

Or download manually and place files in `data/raw/`:

```
data/raw/
├── train_features.csv
├── train_labels.csv
├── test_features.csv
├── train_features/             # directory of training images
└── test_features/              # directory of test images
```

### Training

```bash
# Single fold (default, recommended for iteration)
python scripts/train_baseline.py

# Quick smoke test — 1 epoch on CPU
python scripts/train_baseline.py --epochs 1 --device cpu

# Generate submission with TTA and temperature scaling
python scripts/predict.py
```

### Google Colab

Open `notebooks/colab_runner.ipynb` directly from GitHub in Colab. The notebook:

1. Clones this repository
2. Mounts Google Drive for data and checkpoint persistence
3. Copies data from Drive to Colab RAM (required for ~10x training speed)
4. Installs dependencies and runs the training script with real-time output streaming

All checkpoints and submission CSVs are saved back to Drive automatically.

---

## Engineering Notes

**Data locality on Colab**: Loading images directly from Google Drive during training is the single largest bottleneck on Colab. Copying the entire dataset to `/content/ram/` before training begins yields approximately a 10x throughput improvement. This step is automated in the Colab runner notebook.

**Real-time output streaming**: `subprocess.Popen` with `stdout=PIPE, stderr=STDOUT, bufsize=1` and `PYTHONUNBUFFERED=1` is required to stream training logs in real time from a subprocess in Colab. `subprocess.run` blocks the output until the process terminates.

**AMP deprecation**: PyTorch 2.x deprecates `torch.cuda.amp.autocast` in favour of `torch.amp.autocast('cuda')`. All training code uses the updated API.

**Config patching**: Symlinks between Colab and Drive are unreliable. The Colab runner directly patches the YAML config keys for data paths at runtime, rather than relying on symlinks.

---

## Limitations and Future Work

**Temperature scaling scope**: The current calibration is global — a single T applied to all classes equally. Per-class or per-confidence-bin calibration (histogram binning, isotonic regression) may yield additional improvements, particularly for rare classes like `leopard`.

**Single-fold validation**: The reported local CV score is from a single 80/20 split. This estimate has higher variance than a 5-fold score. The gap between local CV and LB may partially reflect this, not only overconfidence.

**No object detection pre-processing**: Top solutions on similar wildlife classification datasets use a two-stage pipeline: MegaDetector to crop the animal bounding box, then a classifier on the crop. This would significantly reduce background noise in the input and is the highest-impact architectural change not yet implemented.

**External data**: Not permitted by competition rules. Models pretrained on iNaturalist or LILA BC would likely give a strong boost but are disqualified.

---

## References

| Paper | Year | Relevance |
|-------|------|-----------|
| [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) | 2019 | Backbone architecture |
| [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) | 2021 | Advanced backbone |
| [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | 2022 | ConvNeXt architecture |
| [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) | 2019 | Optimizer |
| [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) | 2019 | Label smoothing analysis |
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | 2017 | Temperature scaling foundation |
| [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) | 2018 | MixUp augmentation |

---

*Built as part of a personal ML engineering portfolio targeting applied ML roles in Europe. Competition pipeline designed for reproducibility and reuse across future DrivenData competitions.*