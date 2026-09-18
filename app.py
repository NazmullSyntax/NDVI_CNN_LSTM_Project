"""Streamlit web interface for the Dhaka City CNN-LSTM NDVI system."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data_preprocessing import (
    build_stacked_dataset,
    find_ndvi_files,
    generate_demo_boundary,
    generate_demo_ndvi,
    load_boundary,
    load_config,
)
from src.evaluate import main as evaluate_model
from src.predict_future import predict_future
from src.spatial_analysis import main as spatial_analysis
from src.train import train as train_model
from src.visualization import main as visualize


st.set_page_config(page_title="NDVI Green Space Prediction", layout="wide")
st.title("NDVI Green Space Prediction System")
st.caption("CNN-LSTM Spatio-Temporal Prediction | Dhaka City, Bangladesh")
config = load_config(str(ROOT / "config.yaml"))

with st.sidebar:
    st.header("Configuration")
    st.number_input("Sequence length (years)", 3, 15, config["sequence_length"])
    future_years = st.number_input(
        "Future years to predict", 1, 20, config["future_years"]
    )
    st.divider()
    st.markdown(f"**Study area:** {config['study_area']}")
    st.markdown(f"**Historical range:** {config['start_year']}-{config['end_year']}")
    st.markdown(f"**Train end:** {config['train_end_year']}")
    st.markdown(f"**Validation end:** {config['validation_end_year']}")

tab_data, tab_train, tab_eval, tab_future, tab_maps = st.tabs(
    ["Data", "Train", "Evaluate", "Future", "Maps & Graphs"]
)

with tab_data:
    st.subheader("1. Prepare Data")
    st.write(
        "Upload yearly NDVI GeoTIFF files (for example, `ndvi_2000.tif`) and a "
        "Dhaka boundary shapefile/GeoJSON, or generate synthetic demo data."
    )
    generate_col, upload_col = st.columns(2)
    with generate_col:
        if st.button("Generate Synthetic Demo Dataset", type="primary"):
            with st.spinner("Generating synthetic NDVI rasters..."):
                generate_demo_ndvi(
                    config["start_year"],
                    config["end_year"],
                    output_dir=ROOT / config["paths"]["demo_data"],
                    seed=config["random_seed"],
                )
                boundary_path = generate_demo_boundary(
                    ROOT / config["paths"]["boundary"] / "dhaka_boundary.geojson"
                )
                boundary = load_boundary(boundary_path)
                files = find_ndvi_files(
                    ROOT / config["paths"]["demo_data"],
                    config["start_year"],
                    config["end_year"],
                )
                build_stacked_dataset(
                    files,
                    boundary,
                    image_size=config["image_size"],
                    ndvi_min=config["ndvi_min"],
                    ndvi_max=config["ndvi_max"],
                    output_npz=ROOT / config["paths"]["processed_data"] / "ndvi_stacked.npz",
                )
            st.success("Demo dataset ready.")
    with upload_col:
        st.write("Upload real files")
        uploaded_ndvi = st.file_uploader(
            "NDVI GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True
        )
        uploaded_boundary = st.file_uploader(
            "Dhaka boundary (.geojson or zipped shapefile)", type=["zip", "geojson"]
        )
        if uploaded_ndvi and st.button("Process Uploaded Files"):
            raw_dir = ROOT / config["paths"]["raw_data"]
            raw_dir.mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_ndvi:
                (raw_dir / uploaded_file.name).write_bytes(uploaded_file.getvalue())
            if uploaded_boundary and uploaded_boundary.name.endswith(".geojson"):
                boundary_dir = ROOT / config["paths"]["boundary"]
                boundary_dir.mkdir(parents=True, exist_ok=True)
                (boundary_dir / uploaded_boundary.name).write_bytes(uploaded_boundary.getvalue())
            st.success(f"Saved {len(uploaded_ndvi)} raster files to {raw_dir}")

with tab_train:
    st.subheader("2. Train the CNN-LSTM Model")
    st.write(
        f"Chronological split: Train <= {config['train_end_year']}, "
        f"Validation {config['train_end_year'] + 1}-{config['validation_end_year']}, "
        f"Test {config['validation_end_year'] + 1}+"
    )
    if st.button("Start Training"):
        with st.spinner("Training... this may take a few minutes."):
            try:
                train_model(str(ROOT / "config.yaml"))
                st.success("Training complete.")
            except Exception as error:
                st.error(f"Training failed: {error}")

with tab_eval:
    st.subheader("3. Evaluate on Test Set")
    if st.button("Compute Metrics"):
        with st.spinner("Evaluating..."):
            try:
                evaluate_model(str(ROOT / "config.yaml"))
                metrics_csv = ROOT / config["paths"]["csv"] / "model_metrics.csv"
                if metrics_csv.exists():
                    st.dataframe(pd.read_csv(metrics_csv), use_container_width=True)
                for filename, caption in [("actual_vs_predicted.png", "Actual vs Predicted"), ("training_curves.png", "Training Curves")]:
                    figure = ROOT / config["paths"]["figures"] / filename
                    if figure.exists():
                        st.image(str(figure), caption=caption)
            except Exception as error:
                st.error(f"Evaluation failed: {error}")

with tab_future:
    st.subheader(f"4. Predict Next {future_years} Years")
    if st.button("Generate Forecast"):
        with st.spinner("Forecasting..."):
            try:
                forecast = predict_future(str(ROOT / "config.yaml"), int(future_years))
                st.dataframe(forecast.tail(10), use_container_width=True)
                figure = ROOT / config["paths"]["figures"] / f"forecast_to_{forecast['Year'].max()}.png"
                if figure.exists():
                    st.image(str(figure), caption="Forecast")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

with tab_maps:
    st.subheader("5. Maps & Visualizations")
    spatial_col, figure_col = st.columns(2)
    with spatial_col:
        if st.button("Compute Spatial Statistics"):
            try:
                spatial_analysis(str(ROOT / "config.yaml"))
                st.success("Spatial analysis complete.")
            except Exception as error:
                st.error(f"Spatial analysis failed: {error}")
    with figure_col:
        if st.button("Generate Figures"):
            try:
                visualize(str(ROOT / "config.yaml"))
                st.success("Figures generated.")
            except Exception as error:
                st.error(f"Visualization failed: {error}")

    figure_dir = ROOT / config["paths"]["figures"]
    for filename in ["ndvi_maps.png", "mean_ndvi_timeseries.png", "green_space_percent.png"]:
        figure = figure_dir / filename
        if figure.exists():
            st.image(str(figure), caption=filename)

    csv_dir = ROOT / config["paths"]["csv"]
    for filename in ["model_metrics.csv", "green_space_statistics.csv", "future_predictions.csv"]:
        csv_file = csv_dir / filename
        if csv_file.exists():
            st.download_button(
                f"Download {filename}", csv_file.read_bytes(), file_name=filename, mime="text/csv"
            )
