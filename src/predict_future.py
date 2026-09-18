"""Future NDVI prediction utilities."""

from pathlib import Path

import numpy as np


def recursive_forecast(model, seed_sequence: np.ndarray, steps: int) -> np.ndarray:
    """Predict future values by feeding each prediction back into the sequence."""
    if steps < 1:
        raise ValueError("steps must be positive")
    sequence = np.asarray(seed_sequence, dtype=np.float32).copy()
    predictions = []
    for _ in range(steps):
        next_value = np.asarray(model.predict(sequence[None, ...], verbose=0))[0]
        predictions.append(next_value)
        sequence = np.concatenate([sequence[1:], next_value.reshape(1, *next_value.shape)], axis=0)
    return np.stack(predictions)


def save_predictions(predictions: np.ndarray, path: str | Path) -> Path:
    """Save predictions as a NumPy array."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, predictions)
    return destination
