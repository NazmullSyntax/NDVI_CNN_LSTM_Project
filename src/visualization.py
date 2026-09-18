"""
visualization.py
================
Publication-quality maps and graphs:
- NDVI maps for milestone years
- Mean NDVI over time
- Green-space % over time
- All figures saved to outputs/figures/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import load_config, denormalize_ndvi


def plot_ndvi_maps(years_to_plot, ndvi_arrays, out_dir: Path):
    """Grid of NDVI maps for selected years."""
    n = len(years_to_plot)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-0.2, vmax=0.8)

    for ax, yr, arr in zip(axes, years_to_plot, ndvi_arrays):
        im = ax.imshow(arr, cmap=cmap, norm=norm)
        ax.set_title(f"NDVI {yr}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), orientation="horizontal",
                 fraction=0.04, pad=0.05, label="NDVI")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ndvi_maps.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualization] NDVI maps -> {out}")


def plot_mean_ndvi(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(df["Year"], df["Mean_NDVI"], "o-", color="seagreen")
    plt.xlabel("Year")
    plt.ylabel("Mean NDVI")
    plt.title("Mean NDVI over Time (Dhaka)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "mean_ndvi_timeseries.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[visualization] Mean NDVI timeseries -> {out}")


def plot_green_percent(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(df["Year"], df["Green_percent"], "s-", color="forestgreen")
    plt.xlabel("Year")
    plt.ylabel("Green Area (%)")
    plt.title("Green-Space Percentage over Time (Dhaka)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "green_space_percent.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[visualization] Green % timeseries -> {out}")


def main(config_path="config.yaml"):
    config = load_config(config_path)
    root = Path(__file__).resolve().parent.parent
    processed = root / config["paths"]["processed_data"]
    csv_dir = root / config["paths"]["csv"]
    fig_dir = root / config["paths"]["figures"]

    # ---- NDVI maps for milestone years ----
    npz = np.load(processed / "ndvi_stacked.npz", allow_pickle=True)
    years = [int(y) for y in npz["years"]]
    stacked = npz["data"]

    milestone = [y for y in [2000, 2005, 2010, 2015, 2020, 2025] if y in years]
    idxs = [years.index(y) for y in milestone]
    arrays = []
    for i in idxs:
        arr = denormalize_ndvi(stacked[i][..., 0], config["ndvi_min"], config["ndvi_max"])
        arr = np.where(stacked[i][..., 0] == 0.0, np.nan, arr)
        arrays.append(arr)
    plot_ndvi_maps(milestone, arrays, fig_dir)

    # ---- Timeseries from green-space stats CSV ----
    stats_csv = csv_dir / "green_space_statistics.csv"
    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        plot_mean_ndvi(df, fig_dir)
        plot_green_percent(df, fig_dir)
    else:
        print("[visualization] Run spatial_analysis.py first for time series plots.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)