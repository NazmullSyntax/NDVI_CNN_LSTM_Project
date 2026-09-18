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
