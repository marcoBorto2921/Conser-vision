# =============================================================================
# Makefile — Conser-vision: Wildlife Image Classification
# =============================================================================

.PHONY: help setup train train-fast predict eda clean lint test

COMP = conser-vision
CONFIG = configs/config.yaml
PYTHON = python

help:
	@echo "Available targets:"
	@echo "  setup        Install dependencies"
	@echo "  eda          Run EDA notebook"
	@echo "  train        Train baseline (5-fold CV)"
	@echo "  train-fast   Train 1-fold only (debug)"
	@echo "  train-adv    Train advanced model (ConvNeXt)"
	@echo "  predict      Generate submission with TTA"
	@echo "  predict-fast Generate submission without TTA"
	@echo "  lint         Run ruff linter"
	@echo "  clean        Remove checkpoints and submissions"

setup:
	pip install -r requirements.txt

eda:
	$(PYTHON) -c "exec(open('notebooks/01_eda.py').read())"

train:
	$(PYTHON) scripts/train_baseline.py --config $(CONFIG)

train-fast:
	$(PYTHON) scripts/train_baseline.py --config $(CONFIG) --folds 1

train-adv:
	$(PYTHON) scripts/train_advanced.py --config $(CONFIG)

predict:
	$(PYTHON) scripts/predict.py --config $(CONFIG)

predict-fast:
	$(PYTHON) scripts/predict.py --config $(CONFIG) --no-tta

lint:
	ruff check src/ scripts/ utils/

test:
	pytest tests/ -v

clean:
	rm -f models/weights/*.pth models/checkpoints/*.pth
	rm -f submissions/*.csv
	@echo "Cleaned checkpoints and submissions."
