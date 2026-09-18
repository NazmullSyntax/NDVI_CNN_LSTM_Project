"""
cnn_lstm_model.py
=================
End-to-end CNN-LSTM model.

Input : sequence of NDVI images
        shape (seq_len, H, W, 1)
Output: predicted NDVI image for the next year
        shape (H, W, 1)

Uses TimeDistributed(CNN) to extract per-frame spatial features,
then LSTM to model temporal dynamics, then a Dense head that
reconstructs the NDVI image.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from .cnn_model import build_cnn_feature_extractor


def build_cnn_lstm_model(
    sequence_length=5,
    image_size=64,
    feature_dim=128,
    lstm_units=128,
    name="cnn_lstm",
) -> Model:
    """
    CNN-LSTM model for spatio-temporal NDVI prediction.
    """
    # 1. CNN branch (shared across time via TimeDistributed)
    cnn = build_cnn_feature_extractor(
        input_shape=(image_size, image_size, 1), feature_dim=feature_dim
    )

    # 2. Input: sequence of images
    inputs = layers.Input(
        shape=(sequence_length, image_size, image_size, 1), name="ndvi_sequence"
    )

    # 3. Apply CNN to each time step
    x = layers.TimeDistributed(cnn, name="time_distributed_cnn")(inputs)
    # x shape: (batch, seq_len, feature_dim)

    # 4. Temporal modeling
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.2)(x)

    # 5. Reconstruction head
    x = layers.Dense(image_size * image_size, activation="relu")(x)
    outputs = layers.Reshape((image_size, image_size, 1), name="ndvi_prediction")(x)

    # Note: NDVI in [0,1] normalized range → use sigmoid on final activation
    # We apply sigmoid here to keep output in [0,1].
    outputs = layers.Activation("sigmoid", name="ndvi_sigmoid")(outputs)

    model = Model(inputs, outputs, name=name)
    return model


if __name__ == "__main__":
    m = build_cnn_lstm_model()
    m.summary()