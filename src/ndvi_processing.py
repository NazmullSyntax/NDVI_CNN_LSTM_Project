"""NDVI calculation and raster processing helpers."""

from pathlib import Path
from typing import Optional

import numpy as np


def calculate_ndvi(nir: np.ndarray, red: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Calculate NDVI from aligned NIR and red reflectance arrays."""
    nir_values = np.asarray(nir, dtype=np.float32)
    red_values = np.asarray(red, dtype=np.float32)
    denominator = nir_values + red_values
    return np.divide(nir_values - red_values, denominator + epsilon, where=np.isfinite(denominator), out=np.zeros_like(denominator))


def load_array(path: str | Path) -> np.ndarray:
    """Load a NumPy array from .npy or .npz storage."""
    source = Path(path)
    if source.suffix == ".npz":
        archive = np.load(source)
        return archive[archive.files[0]]
    return np.load(source)


def summarize_ndvi(ndvi: np.ndarray) -> dict[str, float]:
    """Return common descriptive statistics for an NDVI array."""
    values = np.asarray(ndvi, dtype=np.float32)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "missing_fraction": float(np.mean(~np.isfinite(values))),
    }
