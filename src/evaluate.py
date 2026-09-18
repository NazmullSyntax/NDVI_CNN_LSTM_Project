"""
evaluate.py
===========
Computes MAE, RMSE, and R² on the held-out test set, and produces
diagnostic plots (training curves, actual vs predicted, residuals).
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import load_config, denormalize_ndvi


def compute_metrics(y_true, y_pred, ndvi_min=-1.0, ndvi_max=1.0):
    """Compute MAE, RMSE, R² in real NDVI units (masking zeros)."""
    mask = y_true != 0.0
    yt = y_true[mask]
    yp = y_pred[mask]

    # Denormalize from [0,1] back to real NDVI
    yt = denormalize_ndvi(yt, ndvi_min, ndvi_max)
    yp = denormalize_ndvi(yp, ndvi_min, ndvi_max)

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def plot_training_history(history_path: Path, out_path: Path):
    with open(history_path) as f:
        hist = json.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["loss"], label="Train")
    ax[0].plot(hist["val_loss"], label="Validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Masked MSE")
    ax[0].legend()

    ax[1].plot(hist["mae"], label="Train")
    ax[1].plot(hist["val_mae"], label="Validation")
    ax[1].set_title("MAE")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("MAE")
    ax[1].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[evaluate] Training curves -> {out_path}")


def plot_actual_vs_predicted(y_true, y_pred, years, out_path: Path,
                             ndvi_min=-1.0, ndvi_max=1.0):
    """Mean NDVI per year: actual vs predicted."""
    actual_means, pred_means = [], []
    for i in range(len(years)):
        m = y_true[i] != 0.0
        actual_means.append(
            denormalize_ndvi(y_true[i][m].mean(), ndvi_min, ndvi_max)
        )
        pred_means.append(
            denormalize_ndvi(y_pred[i][m].mean(), ndvi_min, ndvi_max)
        )

    plt.figure(figsize=(8, 4))
    plt.plot(years, actual_means, "o-", label="Actual")
    plt.plot(years, pred_means, "s--", label="Predicted")
    plt.xlabel("Year")
    plt.ylabel("Mean NDVI")
    plt.title("Actual vs Predicted Mean NDVI (Test Set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[evaluate] Actual vs predicted -> {out_path}")

    return {"years": list(years), "actual": actual_means, "predicted": pred_means}


def plot_residuals(y_true, y_pred, out_path: Path, ndvi_min=-1.0, ndvi_max=1.0):
    mask = y_true != 0.0
    yt = denormalize_ndvi(y_true[mask], ndvi_min, ndvi_max)
    yp = denormalize_ndvi(y_pred[mask], ndvi_min, ndvi_max)
    residuals = yt - yp

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(residuals, bins=60, color="steelblue", edgecolor="black")
    ax[0].set_title("Residual Distribution")
    ax[0].set_xlabel("Actual - Predicted NDVI")
    ax[0].set_ylabel("Frequency")

    ax[1].scatter(yp, residuals, s=1, alpha=0.3, color="darkorange")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_title("Residuals vs Predicted")
    ax[1].set_xlabel("Predicted NDVI")
    ax[1].set_ylabel("Residual")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[evaluate] Residuals -> {out_path}")


def main(config_path="config.yaml"):
    config = load_config(config_path)
    root = Path(__file__).resolve().parent.parent
    model_dir = root / config["paths"]["models"]
    fig_dir = root / config["paths"]["figures"]
    csv_dir = root / config["paths"]["csv"]

    # Load predictions
    pred_file = model_dir / "test_predictions.npz"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"ERROR: {pred_file} not found. Run train.py first."
        )
    npz = np.load(pred_file, allow_pickle=True)
    y_true = npz["y_true"]
    y_pred = npz["y_pred"]
    years = [int(y) for y in npz["years"]]

    # Metrics
    metrics = compute_metrics(
        y_true, y_pred, config["ndvi_min"], config["ndvi_max"]
    )
    print("\n" + "=" * 50)
    print(" TEST SET METRICS (real NDVI units)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    csv_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([metrics]).to_csv(csv_dir / "model_metrics.csv", index=False)
    print(f"[evaluate] Metrics -> {csv_dir / 'model_metrics.csv'}")

    # Plots
    plot_training_history(model_dir / "training_history.json",
                          fig_dir / "training_curves.png")
    yearly = plot_actual_vs_predicted(
        y_true, y_pred, years, fig_dir / "actual_vs_predicted.png",
        config["ndvi_min"], config["ndvi_max"]
    )
    plot_residuals(y_true, y_pred, fig_dir / "residuals.png",
                   config["ndvi_min"], config["ndvi_max"])

    pd.DataFrame(yearly).to_csv(csv_dir / "test_yearly_ndvi.csv", index=False)
    print(f"[evaluate] Test yearly NDVI -> {csv_dir / 'test_yearly_ndvi.csv'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)