"""Evaluation metrics for NDVI forecasts."""

import numpy as np


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Calculate MAE, RMSE, and bias."""
    observed = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    error = forecast - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }
