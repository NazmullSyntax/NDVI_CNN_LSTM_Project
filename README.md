# 🌿 CNN-LSTM Based Spatio-Temporal Prediction of Urban Green Space Dynamics Using NDVI

**Study Area:** Dhaka City, Bangladesh  
**Historical Period:** 2000 – 2025  
**Future Prediction:** Configurable (default 2026 – 2030)

---

## 1. Introduction

This project predicts future urban green-space dynamics in Dhaka City using
satellite-derived NDVI (Normalized Difference Vegetation Index) time series.

The model combines:
- **CNN (Convolutional Neural Network)** → extracts spatial patterns from each NDVI image
- **LSTM (Long Short-Term Memory)** → learns temporal dynamics across years
- **CNN-LSTM (TimeDistributed)** → end-to-end spatio-temporal forecasting

---

## 2. Research Objectives

1. Analyze vegetation change in Dhaka from 2000 to 2025.
2. Build a CNN-LSTM model that learns spatio-temporal NDVI patterns.
3. Predict future NDVI (2026 onward).
4. Quantify green-space area and percentage change.
5. Validate the model with **chronological back-testing** (not random splits).

---

## 3. System Architecture

---

## 4. Dataset Requirements

- One NDVI GeoTIFF per year: `ndvi_2000.tif` through `ndvi_2025.tif`
- All rasters must share the same CRS, extent, resolution, and grid.
- A Dhaka boundary shapefile or GeoJSON, such as `data/boundary/dhaka_boundary.shp`.

Supported raw sensors are Landsat 5/7/8/9 and Sentinel-2.

| Sensor | RED | NIR | QA |
| --- | --- | --- | --- |
| Landsat 5/7 | B3 | B4 | BQA |
| Landsat 8/9 | B4 | B5 | BQA |
| Sentinel-2 | B4 | B8 | SCL |

Convert sensor imagery to aligned annual NDVI GeoTIFFs before running the
preprocessing pipeline.

---

## 5. Installation

Create and activate a virtual environment, then install the pinned project
dependencies:

```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```
