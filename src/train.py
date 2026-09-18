"""
train.py
========
Trains the CNN-LSTM model on the stacked yearly NDVI dataset using
a *chronological* train/validation/test split.

Why chronological splitting?
----------------------------
Time-series data has temporal ordering. Random shuffling would let
the model "see" future years during training, inflating test metrics
and producing unrealistically good results. Chronological splitting
mimics real forecasting: train on the past, validate/test on the
future.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import callbacks, optimizers, losses

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.cnn_lstm_model import build_cnn_lstm_model
from src.data_preprocessing import (
    load_config,
    load_boundary,
    build_stacked_dataset,
    find_ndvi_files,
    create_sequences,
)


# ----------------------------------------------------------------------
def set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------
def chronological_split(X, y, years, train_end, val_end):
    """Split arrays by year into train / val / test."""
    def _mask(yrs, lo, hi):
        return np.array([(lo <= yr <= hi) for yr in yrs])

    train_mask = _mask(years, min(years), train_end)
    val_mask = _mask(years, train_end + 1, val_end)
    test_mask = _mask(years, val_end + 1, max(years))

    return (
        X[train_mask], y[train_mask], [y for y, m in zip(years, train_mask) if m],
        X[val_mask],   y[val_mask],   [y for y, m in zip(years, val_mask) if m],
        X[test_mask],  y[test_mask],  [y for y, m in zip(years, test_mask) if m],
    )


# ----------------------------------------------------------------------
def masked_mse(y_true, y_pred):
    """MSE that ignores pixels where y_true is exactly 0 (our NaN proxy)."""
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    squared = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(squared) / (tf.reduce_sum(mask) + 1e-8)


# ----------------------------------------------------------------------
def train(config_path="config.yaml"):
    config = load_config(config_path)
    set_seeds(config["random_seed"])

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / config["paths"]["processed_data"]
    model_dir = root / config["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load or build stacked dataset ----
    stacked_path = processed_dir / "ndvi_stacked.npz"
    if not stacked_path.exists():
        print(f"[train] {stacked_path} not found. Building from raw data...")
        data_dir = root / config["paths"]["raw_data"]
        boundary_path = root / config["paths"]["boundary"] / "dhaka_boundary.shp"
        files = find_ndvi_files(data_dir, config["start_year"], config["end_year"])
        boundary = load_boundary(boundary_path)
        years, stacked, _ = build_stacked_dataset(
            files,
            boundary,
            image_size=config["image_size"],
            ndvi_min=config["ndvi_min"],
            ndvi_max=config["ndvi_max"],
            output_npz=stacked_path,
        )
    else:
        npz = np.load(stacked_path, allow_pickle=True)
        years = [int(y) for y in npz["years"]]
        stacked = npz["data"]
        print(f"[train] Loaded {stacked_path} shape={stacked.shape}")

    # ---- Create sequences ----
    seq_len = config["sequence_length"]
    X, y, target_years = create_sequences(stacked, years, seq_len)
    print(f"[train] Sequences: X={X.shape}, y={y.shape}")

    # ---- Chronological split ----
    X_tr, y_tr, yr_tr, X_va, y_va, yr_va, X_te, y_te, yr_te = chronological_split(
        X, y, target_years, config["train_end_year"], config["validation_end_year"]
    )
    print(f"[train] Train years: {yr_tr}")
    print(f"[train] Val   years: {yr_va}")
    print(f"[train] Test  years: {yr_te}")

    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
        raise ValueError(
            "ERROR: Empty split. Check train_end_year / validation_end_year "
            "in config.yaml vs your data range."
        )

    # ---- Build model ----
    model = build_cnn_lstm_model(
        sequence_length=seq_len,
        image_size=config["image_size"],
        feature_dim=128,
        lstm_units=128,
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=masked_mse,
        metrics=["mae"],
    )
    model.summary()

    # ---- Callbacks ----
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_cnn_lstm.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ---- Train ----
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=cbs,
        verbose=1,
    )

    # ---- Save final model & history ----
    model.save(model_dir / "final_cnn_lstm.h5")
    hist_path = model_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # ---- Save test predictions for evaluation ----
    preds = model.predict(X_te)
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        y_true=y_te,
        y_pred=preds,
        years=np.array(yr_te),
    )
    print(f"[train] Saved model, history, and test predictions to {model_dir}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)