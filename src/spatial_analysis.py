"""Spatial analysis helpers with optional rasterio/geopandas integration."""

import numpy as np


def zonal_mean(ndvi: np.ndarray, zones: np.ndarray) -> dict[int, float]:
    """Calculate mean NDVI for each integer zone label."""
    values = np.asarray(ndvi, dtype=float)
    labels = np.asarray(zones)
    if values.shape != labels.shape:
        raise ValueError("ndvi and zones must have the same shape")
    return {
        int(label): float(np.nanmean(values[labels == label]))
        for label in np.unique(labels)
    }
