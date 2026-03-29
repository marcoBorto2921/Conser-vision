# %% [markdown]
# # 01 — Exploratory Data Analysis
# ## Conser-vision: Wildlife Image Classification
# Competition: https://www.drivendata.org/competitions/87/
#
# **Goals:**
# 1. Understand class distribution and imbalance
# 2. Visualise sample images per class (day vs. night)
# 3. Analyse image properties (size, aspect ratio, channel stats)
# 4. Check train/test distribution alignment
# 5. Identify any data quality issues

# %% Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# %% Load data
TRAIN_FEATURES = "../data/raw/train_features.csv"
TRAIN_LABELS   = "../data/raw/train_labels.csv"
TEST_FEATURES  = "../data/raw/test_features.csv"
TRAIN_IMG_DIR  = "../data/raw/train_features"
TEST_IMG_DIR   = "../data/raw/test_features"

CLASS_NAMES = [
    "antelope_duiker", "bird", "blank", "civet_genet",
    "hog", "leopard", "monkey_prosimian", "rodent",
]

train_features = pd.read_csv(TRAIN_FEATURES)
train_labels   = pd.read_csv(TRAIN_LABELS)
test_features  = pd.read_csv(TEST_FEATURES)

train_df = train_features.merge(train_labels, on="id", how="left")

print(f"Train: {len(train_df):,} | Test: {len(test_features):,}")
print(f"Train columns: {list(train_df.columns)}")
print(train_df.head(3))

# %% Class distribution
class_counts = train_labels[CLASS_NAMES].sum().sort_values(ascending=False)
print("\nClass counts:\n", class_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Absolute counts
axes[0].bar(class_counts.index, class_counts.values, color=sns.color_palette("muted", 8))
axes[0].set_title("Class Distribution (absolute)")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

# Percentage
class_pct = class_counts / class_counts.sum() * 100
axes[1].bar(class_pct.index, class_pct.values, color=sns.color_palette("muted", 8))
axes[1].set_title("Class Distribution (%)")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("% of train set")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("../data/processed/class_distribution.png", bbox_inches="tight")
plt.show()

# %% Check for multi-label samples
label_sums = train_labels[CLASS_NAMES].sum(axis=1)
print(f"\nMulti-label samples: {(label_sums > 1).sum()}")
print(f"No-label samples:    {(label_sums == 0).sum()}")
print(f"Single-label:        {(label_sums == 1).sum()}")

# %% Sample images per class
def show_samples(class_name: str, n: int = 6):
    ids = train_labels[train_labels[class_name] == 1]["id"].sample(min(n, train_labels[class_name].sum()), random_state=42)
    fig, axes = plt.subplots(1, len(ids), figsize=(3 * len(ids), 3))
    fig.suptitle(f"Class: {class_name}", fontsize=12)
    for ax, img_id in zip(axes, ids):
        img_path = Path(TRAIN_IMG_DIR) / f"{img_id}.jpg"
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(str(img_id)[:8], fontsize=6)
    plt.tight_layout()
    plt.savefig(f"../data/processed/samples_{class_name}.png", bbox_inches="tight")
    plt.show()

for cls in CLASS_NAMES:
    show_samples(cls)

# %% Image size analysis (sample 200 images)
print("\nAnalysing image sizes (sample)...")
sizes = []
sample_ids = train_df["id"].sample(min(200, len(train_df)), random_state=42)

for img_id in tqdm(sample_ids):
    img_path = Path(TRAIN_IMG_DIR) / f"{img_id}.jpg"
    if img_path.exists():
        img = Image.open(img_path)
        sizes.append({"width": img.width, "height": img.height, "mode": img.mode})

sizes_df = pd.DataFrame(sizes)
print(sizes_df.describe())
print(f"\nImage modes: {sizes_df['mode'].value_counts().to_dict()}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(sizes_df["width"], bins=30, color="steelblue")
axes[0].set_title("Width distribution")
axes[1].hist(sizes_df["height"], bins=30, color="coral")
axes[1].set_title("Height distribution")
axes[2].scatter(sizes_df["width"], sizes_df["height"], alpha=0.3, s=10)
axes[2].set_title("Width vs Height")
plt.tight_layout()
plt.savefig("../data/processed/image_sizes.png", bbox_inches="tight")
plt.show()

# %% Channel statistics (brightness proxy for day/night)
print("\nComputing brightness stats per class...")
brightness_data = []

for cls in CLASS_NAMES:
    cls_ids = train_labels[train_labels[cls] == 1]["id"].sample(min(30, int(train_labels[cls].sum())), random_state=42)
    for img_id in cls_ids:
        img_path = Path(TRAIN_IMG_DIR) / f"{img_id}.jpg"
        if img_path.exists():
            img = np.array(Image.open(img_path).convert("L"))  # grayscale
            brightness_data.append({"class": cls, "mean_brightness": img.mean(), "std_brightness": img.std()})

bright_df = pd.DataFrame(brightness_data)
fig, ax = plt.subplots(figsize=(10, 5))
bright_df.boxplot(column="mean_brightness", by="class", ax=ax)
ax.set_title("Mean brightness by class (proxy for day/night)")
ax.set_xlabel("Class")
plt.suptitle("")
plt.tight_layout()
plt.savefig("../data/processed/brightness_by_class.png", bbox_inches="tight")
plt.show()

# %% Summary
print("\n" + "="*50)
print("EDA Summary")
print("="*50)
print(f"Total train images: {len(train_df):,}")
print(f"Total test images:  {len(test_features):,}")
print(f"Classes: {CLASS_NAMES}")
print(f"\nMost frequent class: {class_counts.idxmax()} ({class_counts.max():,} samples, {class_counts.max()/len(train_df)*100:.1f}%)")
print(f"Rarest class:        {class_counts.idxmin()} ({class_counts.min():,} samples, {class_counts.min()/len(train_df)*100:.1f}%)")
print(f"Imbalance ratio:     {class_counts.max()/class_counts.min():.1f}x")
