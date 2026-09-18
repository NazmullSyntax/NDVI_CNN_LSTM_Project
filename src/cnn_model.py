"""Convolutional feature extractor for spatial NDVI patches."""


def build_cnn(input_shape: tuple[int, int, int], filters: int = 32):
    """Build a compact CNN encoder; TensorFlow is imported only when called."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D

    inputs = Input(shape=input_shape, name="ndvi_patch")
    features = Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    features = Conv2D(filters * 2, 3, padding="same", activation="relu")(features)
    outputs = GlobalAveragePooling2D(name="spatial_features")(features)
    return Model(inputs, outputs, name="ndvi_cnn_encoder")
