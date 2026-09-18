# NDVI CNN-LSTM Project

A starter research workflow for calculating Normalized Difference Vegetation Index (NDVI), extracting spatial features with a CNN, and forecasting temporal patterns with an LSTM.

## Project structure

- `data/`: raw, processed, boundary, and demo inputs
- `models/`: trained model artifacts
- `notebooks/`: ordered exploration and modeling workflow
- `outputs/`: figures, maps, CSV exports, and reports
- `src/`: reusable preprocessing, modeling, evaluation, prediction, and visualization code

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the dashboard with:

```powershell
streamlit run app.py
```

Start with the notebooks in numerical order. The scaffold uses `.npy` and `.npz` arrays for a simple demo path; raster and vector workflows can be added through `rasterio` and `geopandas`.
