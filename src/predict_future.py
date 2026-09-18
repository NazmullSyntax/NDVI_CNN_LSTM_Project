"""
predict_future.py
=================
Generates future NDVI predictions by feeding the model a rolling
sequence of the last known yearly NDVI images.

Usage:
    python src/predict_future.py --years 5
    python src/predict_future.py --years 10
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from tensorflow.keras.models import load_model

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import (
    load_config,
    denormalize_ndvi,
)
from src.train import masked_mse  # need for custom loss on load


def predict_future(config_path="config.yaml", n_future=5):
    config = load_config(config_path)
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / config["paths"]["processed_data"]
    model_dir = root / config["paths"]["models"]
    csv_dir = root / config["paths"]["csv"]
    fig_dir = root / config["paths"]["figures"]

    # ---- Load stacked historical data ----
    stacked_path = processed_dir / "ndvi_stacked.npz"
    if not stacked_path.exists():
        raise FileNotFoundError(
            f"ERROR: {stacked_path} not found. Run data preprocessing first."
        )
    npz = np.load(stacked_path, allow_pickle=True)
    years = [int(y) for y in npz["years"]]
    stacked = npz["data"]  # (n_years, H, W, 1)

    # ---- Load model ----
    model_file = model_dir / "final_cnn_lstm.h5"
    if not model_file.exists():
        raise FileNotFoundError(
            f"ERROR: {model_file} not found. Run train.py first."
        )
    model = load_model(model_file, custom_objects={"masked_mse": masked_mse})

    seq_len = config["sequence_length"]
    if len(years) < seq_len:
        raise ValueError(
            f"ERROR: Need at least {seq_len} historical years, got {len(years)}."
        )

    # ---- Rolling forecast ----
    # Start with the last `seq_len` years as the first input window
    window = stacked[-seq_len:].copy()  # (seq_len, H, W, 1)
    future_years = []
    future_images = []

    last_year = years[-1]
    for i in range(1, n_future + 1):
        x_in = window[np.newaxis, ...]  # (1, seq_len, H, W, 1)
        pred = model.predict(x_in, verbose=0)[0]  # (H, W, 1)

        future_years.append(last_year + i)
        future_images.append(pred)

        # Roll the window forward
        window = np.concatenate([window[1:], pred[np.newaxis, ...]], axis=0)

    future_images = np.stack(future_images, axis=0)
    print(f"[predict_future] Predicted years: {future_years}")

    # ---- Save predictions as GeoTIFFs (using last historical raster as template) ----
    maps_dir = root / config["paths"]["maps"]
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Find a reference raster (either from raw or demo)
    ref_candidates = list((root / config["paths"]["raw_data"]).glob("ndvi_*.tif")) + \
                     list((root / config["paths"]["demo_data"]).glob("ndvi_*.tif"))
    if not ref_candidates:
        print("[predict_future] WARNING: No reference raster found; skipping GeoTIFF export.")
    else:
        ref = sorted(ref_candidates)[-1]
        with rasterio.open(ref) as src:
            profile = src.profile.copy()
            profile.update(dtype="float32", count=1, nodata=-9999.0)
        for yr, img in zip(future_years, future_images):
            arr = denormalize_ndvi(img[..., 0], config["ndvi_min"], config["ndvi_max"])
            out = maps_dir / f"predicted_ndvi_{yr}.tif"
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            print(f"  Saved {out}")

    # ---- Mean NDVI table ----
    # Historical means
    hist_means = []
    for i, yr in enumerate(years):
        arr = denormalize_ndvi(stacked[i][..., 0], config["ndvi_min"], config["ndvi_max"])
        hist_means.append(float(np.mean(arr)))

    future_means = []
    for img in future_images:
        arr = denormalize_ndvi(img[..., 0], config["ndvi_min"], config["ndvi_max"])
        future_means.append(float(np.mean(arr)))

    df = pd.DataFrame(
        {
            "Year": years + future_years,
            "Mean_NDVI": hist_means + future_means,
            "Type": ["Historical"] * len(years) + ["Predicted"] * len(future_years),
        }
    )
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "future_predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"[predict_future] Table -> {out_csv}")

    # ---- Forecast plot ----
    plt.figure(figsize=(9, 4))
    plt.plot(years, hist_means, "o-", label="Historical (actual)")
    plt.plot(
        [years[-1]] + future_years,
        [hist_means[-1]] + future_means,
        "s--",
        color="crimson",
        label="Predicted",
    )
    plt.axvline(years[-1], color="gray", linestyle=":", alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("Mean NDVI")
    plt.title(f"NDVI Forecast to {future_years[-1]}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = fig_dir / f"forecast_to_{future_years[-1]}.png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[predict_future] Forecast plot -> {out_fig}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5,
                        help="Number of future years to predict")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    predict_future(args.config, args.years)