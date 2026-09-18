"""
spatial_analysis.py
===================
Green-space classification from NDVI, area statistics, and change
maps between two years.

IMPORTANT: NDVI thresholds used here are *project parameters*, not
universally validated scientific boundaries. Adjust in config.yaml.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import load_config, denormalize_ndvi


CLASS_LABELS = {
    0: "Water / non-vegetation",
    1: "Bare soil / built-up / low veg",
    2: "Low vegetation",
    3: "Moderate vegetation",
    4: "High vegetation",
}


def classify_ndvi(ndvi: np.ndarray, thresholds: dict) -> np.ndarray:
    """
    Assign integer classes based on NDVI thresholds.
    Class 0: < water_max
    Class 1: water_max..bare_max
    Class 2: bare_max..low_veg_max
    Class 3: low_veg_max..moderate_veg_max
    Class 4: >= moderate_veg_max
    """
    w = thresholds["water_max"]
    b = thresholds["bare_max"]
    lv = thresholds["low_veg_max"]
    mv = thresholds["moderate_veg_max"]

    classes = np.full(ndvi.shape, -1, dtype=np.int8)
    classes[(ndvi >= -1) & (ndvi < w)] = 0
    classes[(ndvi >= w) & (ndvi < b)] = 1
    classes[(ndvi >= b) & (ndvi < lv)] = 2
    classes[(ndvi >= lv) & (ndvi < mv)] = 3
    classes[ndvi >= mv] = 4
    classes[np.isnan(ndvi)] = -1
    return classes


def pixel_area_m2(transform, crs) -> float:
    """
    Compute pixel area in m² from the affine transform.
    Assumes projected CRS (metres). If geographic (degrees), converts
    via latitude-dependent scaling.
    """
    px_w = abs(transform.a)  # pixel width in CRS units
    px_h = abs(transform.e)  # pixel height in CRS units
    if crs is not None and crs.is_geographic:
        # Rough conversion: 1 degree ≈ 111,320 m at equator
        # For Dhaka (~23.7°N) this is approximate.
        return (px_w * 111_320) * (px_h * 110_570)
    return px_w * px_h


def green_space_statistics(
    ndvi: np.ndarray, transform, crs, thresholds: dict
):
    """Return a dict of areas (km²) and percentages per class."""
    classes = classify_ndvi(ndvi, thresholds)
    area_px = pixel_area_m2(transform, crs) / 1e6  # km² per pixel

    total_valid = np.sum(classes >= 0)
    if total_valid == 0:
        return {}

    stats = {}
    total_green = 0.0
    for cls, label in CLASS_LABELS.items():
        count = int(np.sum(classes == cls))
        area = count * area_px
        stats[label] = {
            "pixels": count,
            "area_km2": float(area),
            "percent": float(100.0 * count / total_valid),
        }
        if cls >= 2:  # low + moderate + high = "green"
            total_green += area

    stats["Total green area_km2"] = float(total_green)
    stats["Total green percent"] = float(100.0 * (total_green / (area_px * total_valid)))
    return stats


def change_map(ndvi_a: np.ndarray, ndvi_b: np.ndarray, threshold: float = 0.05):
    """
    Return (change, category_map) where category_map is:
        0 = Stable
        1 = Vegetation Increase
        2 = Vegetation Decrease
    """
    diff = ndvi_b - ndvi_a
    cat = np.zeros_like(diff, dtype=np.int8)
    cat[diff > threshold] = 1
    cat[diff < -threshold] = 2
    cat[np.isnan(diff)] = -1
    return diff, cat


def plot_change_map(cat: np.ndarray, title: str, out_path: Path):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#d9d9d9", "#2ca02c", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cat, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Stable", "Increase", "Decrease"])
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(config_path="config.yaml"):
    config = load_config(config_path)
    root = Path(__file__).resolve().parent.parent
    processed = root / config["paths"]["processed_data"]
    csv_dir = root / config["paths"]["csv"]
    maps_dir = root / config["paths"]["maps"]

    npz = np.load(processed / "ndvi_stacked.npz", allow_pickle=True)
    years = [int(y) for y in npz["years"]]
    stacked = npz["data"]

    # Reference raster for transform/CRS
    ref_candidates = list((root / config["paths"]["demo_data"]).glob("ndvi_*.tif"))
    if not ref_candidates:
        ref_candidates = list((root / config["paths"]["raw_data"]).glob("ndvi_*.tif"))
    if not ref_candidates:
        print("ERROR: No reference raster found for spatial analysis.")
        return
    ref = sorted(ref_candidates)[0]
    with rasterio.open(ref) as src:
        transform = src.transform
        crs = src.crs

    thresholds = config["green_thresholds"]
    change_thr = config["change_threshold"]

    # ---- Per-year statistics ----
    rows = []
    for i, yr in enumerate(years):
        ndvi = denormalize_ndvi(stacked[i][..., 0], config["ndvi_min"], config["ndvi_max"])
        ndvi[stacked[i][..., 0] == 0.0] = np.nan  # restore NaN
        stats = green_space_statistics(ndvi, transform, crs, thresholds)
        row = {"Year": yr, "Mean_NDVI": float(np.nanmean(ndvi))}
        for k, v in stats.items():
            if isinstance(v, dict):
                row[k] = v["area_km2"]
        row["Green_percent"] = stats.get("Total green percent", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_dir / "green_space_statistics.csv", index=False)
    print(f"[spatial_analysis] Green space stats -> {csv_dir / 'green_space_statistics.csv'}")
    print(df.head())

    # ---- Change map: first year -> last historical year ----
    first = denormalize_ndvi(stacked[0][..., 0], config["ndvi_min"], config["ndvi_max"])
    last = denormalize_ndvi(stacked[-1][..., 0], config["ndvi_min"], config["ndvi_max"])
    _, cat = change_map(first, last, change_thr)
    plot_change_map(cat, f"NDVI Change {years[0]}-{years[-1]}",
                    maps_dir / f"change_{years[0]}_{years[-1]}.png")
    print(f"[spatial_analysis] Change map {years[0]}-{years[-1]} saved.")

    # ---- Predicted change map if predictions exist ----
    pred_dir = maps_dir
    predicted_files = sorted(pred_dir.glob("predicted_ndvi_*.tif"))
    if predicted_files:
        with rasterio.open(predicted_files[0]) as src:
            pred_arr = src.read(1)
        _, cat_pred = change_map(last, pred_arr, change_thr)
        plot_change_map(
            cat_pred,
            f"NDVI Change {years[-1]}-{predicted_files[0].stem.split('_')[-1]}",
            maps_dir / f"change_{years[-1]}_predicted.png",
        )
        print("[spatial_analysis] Predicted change map saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)