"""Plotting helpers for NDVI outputs."""

from pathlib import Path

import numpy as np


def save_ndvi_map(ndvi: np.ndarray, path: str | Path, title: str = "NDVI") -> Path:
    """Save a simple NDVI image using matplotlib."""
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="NDVI")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination
