"""
cnn_model.py
============
CNN spatial feature extractor. The CNN is NOT used as a classifier
here — it produces a compact feature vector that summarizes the
spatial pattern of a single NDVI image.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_feature_extractor(
    input_shape=(64, 64, 1), feature_dim=128, name="cnn_extractor"
) -> Model:
    """
    Build a CNN that maps an NDVI image to a 1D feature vector.

    Architecture:
        Conv2D(32) -> ReLU -> MaxPool
        Conv2D(64) -> ReLU -> MaxPool
        Conv2D(128) -> ReLU -> GlobalAveragePooling
        Dense(feature_dim)
    """
    inputs = layers.Input(shape=input_shape, name="ndvi_image")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(feature_dim, activation="relu", name="spatial_features")(x)

    return Model(inputs, x, name=name)


if __name__ == "__main__":
    model = build_cnn_feature_extractor()
    model.summary()