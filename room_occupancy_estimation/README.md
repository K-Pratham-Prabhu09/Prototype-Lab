# 🏠 Room Occupancy Estimation from Multi-Sensor Fusion

## Problem Statement

> **Can a deep learning model accurately estimate the number of people in a room using only non-invasive environmental sensors — and can it be compressed enough to run on a microcontroller?**

Modern smart buildings rely on occupancy information for HVAC optimization, energy management, lighting control, and emergency evacuation planning. Camera-based systems raise **privacy concerns**. Wi-Fi/Bluetooth tracking requires people to carry devices. The alternative: **passive environmental sensing** — temperature, light, sound, CO₂, and motion (PIR) sensors that detect human presence without identifying anyone.

This project builds an **end-to-end deep learning pipeline** that:

1. **Trains a neural network** on multi-sensor time-series data to predict room occupancy count (0–3 people)
2. **Quantizes the model** (INT8 / dynamic quantization) to measure accuracy-vs-size tradeoffs
3. **Exports to ONNX** for deployment on edge hardware (Raspberry Pi, Arduino, ESP32, etc.)

The core question isn't just _"can you classify occupancy?"_ — it's _"how much model precision can you sacrifice before a building's HVAC system starts wasting energy on an empty room?"_

---

## Dataset

**Source:** [UCI Machine Learning Repository — Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)

| Property | Value |
|---|---|
| **Samples** | 10,129 |
| **Features** | 16 sensor channels + timestamp |
| **Target** | `Room_Occupancy_Count` (0, 1, 2, or 3 people) |
| **Date Range** | 2017-12-22 to 2018-01-11 (7 days) |
| **Missing Values** | None |
| **Format** | Single CSV, fully processed, ready to load |

### Sensor Layout (7 non-invasive sensors)

| Sensor ID | Type | Count | What It Measures |
|---|---|---|---|
| S1–S4 | **Temperature** | 4 | Ambient temperature at 4 locations in the room (°C) |
| S1–S4 | **Light** | 4 | Illuminance at 4 locations (lux) |
| S1–S4 | **Sound** | 4 | Sound level at 4 locations (voltage) |
| S5 | **CO₂** | 1 | Carbon dioxide concentration (ppm) + slope |
| S6–S7 | **PIR (Passive Infrared)** | 2 | Motion detection (binary) at 2 locations |

### Class Distribution

```
0 people  →  8,228 samples  (81.2%)  ████████████████████████████████░░░░
1 person  →    459 samples  ( 4.5%)  ██░
2 people  →    748 samples  ( 7.4%)  ███░
3 people  →    694 samples  ( 6.9%)  ███░
```

> ⚠️ **Heavy class imbalance** — the room is empty 81% of the time. Your model _will_ cheat by predicting "0" unless you address this.

---

## What Makes This Interesting

### 🧠 Modelling Challenges
- **Class imbalance:** 81% of samples are class 0. Accuracy is a misleading metric — a dummy model scores 81%. You need to think about weighted loss, stratified sampling, F1-macro, or SMOTE.
- **Multi-modal sensor fusion:** Temperature, light, sound, CO₂, and PIR are fundamentally different physical quantities at different scales. How you normalize, combine, and weight them matters.
- **Spatial redundancy:** 4 temperature sensors, 4 light sensors, 4 sound sensors — do you need all of them? Feature importance and dimensionality reduction become real questions.
- **Temporal patterns:** Occupancy changes over time. A single row is a snapshot, but the _trajectory_ of CO₂ slope and PIR triggers tells a richer story. Can you leverage windowed sequences?

### ⚡ Quantization & Deployment Challenges
- **Regression-like classification:** The target is ordinal (0 < 1 < 2 < 3). Quantization error might cause off-by-one misclassifications. Is predicting "1 person" when there are "2" acceptable for an HVAC system? What about for emergency evacuation?
- **Sensor noise resilience:** Real sensors drift and produce noise. A quantized model with reduced precision needs to be robust to this — or it's useless in production.
- **ONNX edge deployment:** The final ONNX model should be small enough for a microcontroller. Profile the model size, inference latency, and memory footprint.

---

## Suggested Pipeline

```
CSV Data
  │
  ├── EDA & Feature Engineering
  │     ├── Handle class imbalance
  │     ├── Normalize sensor scales
  │     └── Optional: window-based features (CO₂ slope, rolling stats)
  │
  ├── Model Training (PyTorch)
  │     ├── Baseline: Simple MLP / 1D-CNN
  │     ├── Evaluate: F1-macro, confusion matrix, per-class recall
  │     └── Iterate: architecture, loss function, augmentation
  │
  ├── Quantization
  │     ├── Dynamic quantization (INT8)
  │     ├── Compare: model size, inference time, accuracy drop
  │     └── Ask: is the accuracy drop acceptable for the use case?
  │
  └── ONNX Export
        ├── Export quantized model to ONNX
        ├── Validate with ONNX Runtime
        └── Profile: latency, memory, compatibility
```

---

## Success Criteria

| Metric | Target |
|---|---|
| **F1-Macro** (full precision) | > 0.85 |
| **F1-Macro** (quantized INT8) | < 3% drop from full precision |
| **ONNX model size** | < 500 KB |
| **ONNX inference latency** | < 5 ms per sample |
| **Per-class recall** (class 1, 2, 3) | > 0.75 each |

---

## Project Structure

```
room_occupancy_estimation/
├── data/
│   └── Occupancy_Estimation.csv    # Raw dataset (10,129 × 19)
├── src/                            # Source code
│   ├── train.py                    # Model training
│   ├── quantize.py                 # Post-training quantization
│   └── export_onnx.py             # ONNX export & validation
├── models/                         # Saved models (.pt, .onnx)
├── notebooks/                      # EDA & experimentation
└── README.md                       # ← You are here
```

---

## References

- Adarsh Pal Singh, Vivek Jain, Sachin Chaudhari, Frank Alexander Kraemer, Stefan Werner, and Vishal Garg. 2018. _Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes._ IEEE Sensors Journal.
- [UCI Dataset Page](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)
