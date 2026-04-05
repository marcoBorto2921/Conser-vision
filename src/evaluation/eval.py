"""
src/evaluation/eval.py
-----------------------
Model calibration utilities.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def calibrate_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Find optimal temperature T to calibrate model confidence.

    Uses scipy.optimize.minimize_scalar with method='bounded' on [0.5, 5.0]
    to minimise NLL (negative log-likelihood) on the validation set.

    Args:
        val_logits: Raw logits, shape (n, num_classes).
        val_labels: Integer class indices, shape (n,).

    Returns:
        Optimal temperature scalar T.
    """

    def nll(temperature: float) -> float:
        scaled_logits = val_logits / temperature
        probs = softmax(scaled_logits, axis=1)
        # Clip for numerical stability
        probs = np.clip(probs, 1e-12, 1.0)
        # Mean negative log-likelihood for true classes
        true_probs = probs[np.arange(len(val_labels)), val_labels]
        return -np.mean(np.log(true_probs))

    result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    if not result.success:
        warnings.warn(
            f"Temperature calibration did not converge: {result.message}. "
            f"Using T={result.x:.4f} anyway.",
            RuntimeWarning,
            stacklevel=2,
        )
    T: float = float(result.x)
    print(f"Optimal temperature T = {T:.4f}")
    return T
