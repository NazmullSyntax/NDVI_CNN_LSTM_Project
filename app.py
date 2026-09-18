"""Small Streamlit dashboard for inspecting NDVI arrays and forecasts."""

from pathlib import Path

import numpy as np


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="NDVI CNN-LSTM", layout="wide")
    st.title("NDVI CNN-LSTM Project")
    st.caption("Upload a NumPy array to inspect its spatial or temporal NDVI values.")
    uploaded = st.file_uploader("NDVI array (.npy or .npz)", type=["npy", "npz"])
    if uploaded is None:
        st.info("Place a demo array in data/demo or upload one to begin.")
        return
    values = np.load(uploaded)
    if hasattr(values, "files"):
        values = values[values.files[0]]
    st.metric("Mean NDVI", f"{float(np.nanmean(values)):.3f}")
    st.write({"shape": values.shape, "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values))})
    if values.ndim >= 2:
        st.image(values[-1] if values.ndim > 2 else values, caption="Latest NDVI view", clamp=True)


if __name__ == "__main__":
    main()
