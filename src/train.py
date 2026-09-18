"""Training entry points for NDVI forecasting models."""

from pathlib import Path
from typing import Any


def train_model(model: Any, x_train, y_train, x_validation=None, y_validation=None, epochs: int = 20, batch_size: int = 16):
    """Compile and train a Keras model, returning its History object."""
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    validation_data = (x_validation, y_validation) if x_validation is not None and y_validation is not None else None
    return model.fit(x_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size)


def save_model(model: Any, path: str | Path) -> Path:
    """Save a trained Keras model and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    return destination
