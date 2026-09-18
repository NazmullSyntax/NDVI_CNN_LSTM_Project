"""
ndvi_processing.py
==================
NDVI calculation from raw Landsat / Sentinel-2 bands, cloud masking,
and yearly compositing.

NDVI = (NIR - RED) / (NIR + RED)

Band mapping by sensor:
- Landsat 5 TM  : RED = B3, NIR = B4, QA = BQA
- Landsat 7 ETM+: RED = B3, NIR = B4, QA = BQA
- Landsat 8 OLI : RED = B4, NIR = B5, QA = BQA
- Landsat 9 OLI2: RED = B4, NIR = B5, QA = BQA
- Sentinel-2 MSI: RED = B4, NIR = B8, QA = SCL
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio is required. Install: pip install rasterio")
    sys.exit(1)


# Sensor band mapping
SENSOR_BANDS = {
    "landsat5": {"red": 3, "nir": 4, "qa": 1, "scale": 0.0000275, "offset": -0.2},
    "landsat7": {"red": 3, "nir": 4, "qa": 1, "scale": 0.0000275, "offset": -0.2},
    "landsat8": {"red": 4, "nir": 5, "qa": 1, "scale": 0.0000275, "offset": -0.2},
    "landsat9": {"red": 4, "nir": 5, "qa": 1, "scale": 0.0000275, "offset": -0.2},
    "sentinel2": {"red": 4, "nir": 8, "qa": 1, "scale": 0.0001, "offset": 0.0},
}


def calculate_ndvi(
    red: np.ndarray, nir: np.ndarray, clip: bool = True
) -> np.ndarray:
    """
    Compute NDVI from RED and NIR band arrays.

    Parameters
    ----------
    red, nir : np.ndarray
        Reflectance bands (float32), same shape.
    clip : bool
        If True, clip NDVI to [-1, 1].

    Returns
    -------
    ndvi : np.ndarray (float32)
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    denom = nir + red
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)

    ndvi = (nir - red) / denom
    if clip:
        ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi.astype(np.float32)


def cloud_mask_landsat(qa: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of *valid* (non-cloud) pixels from a Landsat
    Collection-2 QA_PIXEL band.

    Bit 0: Fill
    Bit 1: Dilated cloud
    Bit 2: Cirrus
    Bit 3: Cloud
    Bit 4: Cloud shadow
    Bit 5: Snow
    """
    qa = qa.astype(np.uint16)
    fill = (qa >> 0) & 1
    dilated_cloud = (qa >> 1) & 1
    cirrus = (qa >> 2) & 1
    cloud = (qa >> 3) & 1
    shadow = (qa >> 4) & 1
    snow = (qa >> 5) & 1

    invalid = (fill | dilated_cloud | cirrus | cloud | shadow | snow).astype(bool)
    return ~invalid


def cloud_mask_sentinel2(scl: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of *valid* pixels from a Sentinel-2 SCL band.
    Valid classes: 2 (dark veg), 4 (veg), 5 (bare), 6 (water), 7 (unclassified)
    Invalid: 1 (saturated), 3 (shadow), 8/9 (cloud), 10 (cirrus), 11 (snow)
    """
    scl = scl.astype(np.uint8)
    valid_classes = {2, 4, 5, 6, 7}
    mask = np.isin(scl, list(valid_classes))
    return mask


def yearly_composite(ndvi_stack: list) -> np.ndarray:
    """
    Composite multiple NDVI scenes for a year into one image using
    the per-pixel maximum (max-NDVI compositing reduces cloud artifacts).
    """
    if not ndvi_stack:
        raise ValueError("ERROR: ndvi_stack is empty.")
    arr = np.stack(ndvi_stack, axis=0)
    with np.errstate(all="ignore"):
        composite = np.nanmax(arr, axis=0)
    return composite


def process_scene(
    scene_path: Path,
    sensor: str,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Process a single Landsat/Sentinel scene: read bands, apply cloud mask,
    compute NDVI.
    """
    sensor = sensor.lower()
    if sensor not in SENSOR_BANDS:
        raise ValueError(
            f"ERROR: Unsupported sensor '{sensor}'. "
            f"Choose from {list(SENSOR_BANDS.keys())}."
        )

    bands = SENSOR_BANDS[sensor]
    scene_path = Path(scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"ERROR: Scene not found: {scene_path}")

    with rasterio.open(scene_path) as src:
        red = src.read(bands["red"]).astype(np.float32)
        nir = src.read(bands["nir"]).astype(np.float32)

        # Apply reflectance scaling (Landsat C2)
        red = red * bands["scale"] + bands["offset"]
        nir = nir * bands["scale"] + bands["offset"]

        # Cloud masking
        try:
            qa = src.read(bands["qa"])
            if sensor.startswith("landsat"):
                valid = cloud_mask_landsat(qa)
            else:
                valid = cloud_mask_sentinel2(qa)
        except Exception:
            print(f"  WARNING: QA band not usable in {scene_path.name}, skipping mask.")
            valid = np.ones_like(red, dtype=bool)

        ndvi = calculate_ndvi(red, nir)
        ndvi[~valid] = np.nan

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(np.nan_to_num(ndvi, nan=-9999.0), 1)

    return ndvi


def main():
    print("=" * 60)
    print(" NDVI Processing Utility")
    print("=" * 60)
    print("Supported sensors and their band mappings:")
    for sensor, b in SENSOR_BANDS.items():
        print(f"  {sensor:12s}  RED=B{b['red']}  NIR=B{b['nir']}  QA=B{b['qa']}")
    print("\nUse process_scene() from Python to process a scene.")


if __name__ == "__main__":
    main()