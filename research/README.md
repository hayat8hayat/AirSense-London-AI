# 🔬 Research & Model Development

This directory documents the data science methodology, experimentation, and model architecture used in the AirSense project.

## 🎯 Objective
The primary goal (Model A) is to predict air quality levels for **5 key pollutants** (NO2, O3, PM10, PM2.5, SO2) in London (Bloomsbury Station) **3 hours in advance**, utilizing the past **6 hours** of historical context.

*(Note: The notebook also experiments with longer horizons, including 12-hour forecasts using 24-hour and 7-day historical windows.)*

---

## 📊 Data Strategy

### 1. Source & Selection
* **Data Source:** OpenAQ API (London Bloomsbury)
* **Time Resolution:** Hourly measurements
* **Features Selected:** `no2`, `o3`, `pm10`, `pm25`, `so2`

### 2. Preprocessing & Cleaning
* **Gap Filling:** Real-world sensor data often has missing hours. We used **Linear Interpolation** to fill gaps up to **24 hours** to maintain time-series continuity while discarding larger outages.
* **Normalization:** Neural networks require inputs on a similar scale. We applied **Min-Max Scaling** to transform all pollutant values into a range of `[0, 1]`.
    * *Artifact:* The fitted scaler determines the transformation logic for the production pipeline.

---

## 🧠 Model Architecture

We selected an **Encoder-Decoder LSTM** architecture. This structure is designed for sequence-to-sequence prediction, allowing the model to encode the historical context into a fixed internal state before decoding it into the future forecast sequence.

### Network Structure (Model A)
* **Input Layer:** `(Batch_Size, 6, 5)`
    * *Sequence Length:* 6 hours
    * *Features:* 5 pollutants
* **Encoder (LSTM):** **200 units**, activation `relu`. Compresses the input sequence.
* **Bridge (RepeatVector):** Repeats the internal state **3 times** (for the 3 output hours).
* **Decoder (LSTM):** **200 units**, activation `relu`, returning sequences. Unfolds the state into future predictions.
* **Output Layer (TimeDistributed Dense):** `5` units applied to each time step independently.

### Training Configuration
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Epochs:** 50 (with Model Checkpointing)
* **Batch Size:** 32

---

## 📉 Results & Performance

The models were evaluated on a held-out test set (last 20% of data).

* **Performance:** Model A (6h input $\to$ 3h output) achieved an $R^2$ score of **~0.93**, demonstrating high predictive accuracy for short-term forecasting.
* **Trend Capture:** The Encoder-Decoder structure successfully captures the rising and falling trends of pollutants during peak urban activity.

---

## 📓 Notebooks

* **`LSTM_Air_Quality_Forecasting.ipynb`**: The primary research notebook. It contains:
    1.  **Data Acquisition:** Fetching raw data from the OpenAQ API.
    2.  **Exploratory Data Analysis (EDA):** Visualizing correlations and missing data.
    3.  **Feature Engineering:** Resampling to hourly intervals and creating sliding window datasets.
    4.  **Model Training:** Building and fitting Keras LSTM models for multiple time horizons (A, B, and C).
    5.  **Evaluation:** Generating accuracy reports and plots of True vs. Predicted values.
