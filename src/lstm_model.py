"""
lstm_model.py
=============
Standalone LSTM for temporal modeling of pre-extracted CNN feature
vectors. Kept separate from the combined model for clarity and
ablation experiments.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_model(
    sequence_length=5,
    feature_dim=128,
    lstm_units=128,
    output_dim=64 * 64,  # flattened NDVI image
    name="lstm_model",
) -> Model:
    """
    LSTM that reads a sequence of spatial feature vectors and predicts
    a flattened NDVI image.
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="feature_seq")

    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_dim, activation="sigmoid", name="ndvi_flat")(x)

    return Model(inputs, outputs, name=name)


if __name__ == "__main__":
    m = build_lstm_model()
    m.summary()