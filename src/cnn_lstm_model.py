"""Combined CNN-LSTM forecasting model."""


def build_cnn_lstm(input_shape: tuple[int, int, int, int], filters: int = 32, units: int = 64, horizon: int = 1):
    """Build a model for sequences of spatial NDVI patches."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import TimeDistributed, Conv2D, GlobalAveragePooling2D, LSTM, Dense

    inputs = Input(shape=input_shape, name="ndvi_patch_sequence")
    features = TimeDistributed(Conv2D(filters, 3, padding="same", activation="relu"))(inputs)
    features = TimeDistributed(GlobalAveragePooling2D())(features)
    temporal = LSTM(units, dropout=0.1, name="temporal_encoder")(features)
    outputs = Dense(horizon, name="forecast")(temporal)
    return Model(inputs, outputs, name="ndvi_cnn_lstm")
