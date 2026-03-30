# CLAUDE.md — Conser-vision: Wildlife Image Classification

## Competition
- **URL**: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/
- **Task**: Multi-class image classification (8 classes)
- **Metric**: Log-loss (lower is better)
- **Classes**: `antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`

## Project Layout
```
conser-vision/
├── data/
│   ├── raw/               # Coloca qui i dati scaricati da DrivenData (gitignored)
│   │   ├── train_features/  # CSV con path immagini train
│   │   ├── test_features/   # CSV con path immagini test
│   │   ├── train_labels.csv
│   │   └── train_features.csv
│   └── processed/         # Dataset preprocessati
├── src/
│   ├── data/dataset.py    # PyTorch Dataset + transforms
│   ├── models/model.py    # EfficientNet/ConvNeXt wrapper
│   ├── training/train.py  # Training loop con CV
│   └── evaluation/eval.py # Metriche e diagnostica
├── configs/
│   └── config.yaml        # Tutti gli iperparametri
├── notebooks/             # EDA e sperimentazione
├── submissions/           # Output CSV (gitignored)
└── scripts/
    ├── download_data.py   # Helper per scaricare i dati
    ├── train_baseline.py  # Entry point training baseline
    ├── train_advanced.py  # Entry point training avanzato
    └── predict.py         # Genera submission
```

## How to Run (Claude Code Instructions)

### Step 0 — Setup environment (ALWAYS use venv)
```bash
cd conser-vision

# Crea il venv se non esiste ancora
python -m venv .venv

## Git Workflow
After each significant change, commit with a descriptive message following the convention:
`feat: ...`, `fix: ...`, `exp: ...`, `docs: ...`

# Attiva il venv (Linux/macOS)
source .venv/bin/activate
# Attiva il venv (Windows)
# .venv\Scripts\activate

# Installa dipendenze nel venv
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ Claude Code: esegui SEMPRE i comandi python e pip con il venv attivo.
> Verifica con `which python` che punti a `.venv/bin/python` prima di procedere.

### Step 1 — Download data
Scarica manualmente da DrivenData dopo esserti registrato alla competizione:
- `train_features.csv` → `data/raw/train_features.csv`
- `train_labels.csv` → `data/raw/train_labels.csv`
- `test_features.csv` → `data/raw/test_features.csv`
- Cartella immagini train → `data/raw/train_features/`
- Cartella immagini test → `data/raw/test_features/`

Oppure usa il DrivenData CLI:
```bash
pip install drivendata-download
drd download 87 --output data/raw/
```

### Step 2 — EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Step 3 — Train baseline (EfficientNet-B3, 5-fold CV)
```bash
python scripts/train_baseline.py
```

### Step 4 — Train advanced (ConvNeXt ensemble + TTA)
```bash
python scripts/train_advanced.py --config configs/config.yaml
```

### Step 5 — Generate submission
```bash
python scripts/predict.py --checkpoint models/weights/best_model.pth --output submissions/
```

## Technical Strategy

### Model
- **Baseline**: EfficientNet-B3 pretrained on ImageNet (timm)
- **Advanced**: ConvNeXt-Base + EfficientNet-V2-M ensemble
- **Head**: Linear classifier with dropout

### Augmentations (train)
- RandomResizedCrop(224)
- HorizontalFlip, VerticalFlip (camera traps = fixed position, ma utile)
- ColorJitter (brightness/contrast per variazioni giorno/notte)
- RandomRotation(15°)
- Normalize ImageNet stats

### Augmentations (val/test)
- Resize(256) → CenterCrop(224)
- Normalize

### Training
- AdamW + CosineAnnealingLR
- Label smoothing 0.1
- StratifiedKFold(n_splits=5)
- Mixed precision (torch.cuda.amp)
- Early stopping patience=5

### TTA (Test Time Augmentation)
- 5 crops × flips → average probabilities

### Submission
- OOF predictions per validation CV score locale
- Average probabilities across folds per test set
