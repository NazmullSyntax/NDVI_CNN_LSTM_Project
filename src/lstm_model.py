"""Temporal LSTM model components."""


def build_lstm(sequence_shape: tuple[int, int], units: int = 64, horizon: int = 1):
    """Build an LSTM regressor for feature sequences."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense

    inputs = Input(shape=sequence_shape, name="feature_sequence")
    hidden = LSTM(units, dropout=0.1, name="temporal_encoder")(inputs)
    outputs = Dense(horizon, name="ndvi_forecast")(hidden)
    return Model(inputs, outputs, name="ndvi_lstm")
