"""Utilities for preparing NDVI sequences for modeling."""

from pathlib import Path
from typing import Tuple

import numpy as np


def normalize_ndvi(ndvi: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """Clip NDVI values to the physical range and scale to [0, 1]."""
    values = np.asarray(ndvi, dtype=np.float32)
    values = np.clip(values, lower, upper)
    return (values - lower) / (upper - lower)


def make_sequences(array: np.ndarray, sequence_length: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window inputs and future targets from time-first data."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim < 1 or sequence_length < 1 or horizon < 1:
        raise ValueError("array must be time-first and sequence_length/horizon must be positive")
    if len(values) < sequence_length + horizon:
        raise ValueError("array is shorter than sequence_length + horizon")
    inputs = []
    targets = []
    for start in range(len(values) - sequence_length - horizon + 1):
        inputs.append(values[start : start + sequence_length])
        targets.append(values[start + sequence_length : start + sequence_length + horizon])
    return np.stack(inputs), np.stack(targets)


def ensure_directory(path: str | Path) -> Path:
    """Create and return an output directory."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
