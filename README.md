# 🏙️ AirSense London AI

A real-time Air Quality forecasting pipeline using **Python**, **InfluxDB**, **Docker**, and **Deep Learning (LSTM)**.

This system continuously monitors air pollution levels in London (Bloomsbury) and uses an AI model to predict air quality trends for the next 3 hours.

---

## 🚀 Key Features

- **ETL Pipeline:** Automates data fetching from the OpenAQ API (NO2, O3, PM10, PM2.5, SO2)
- **Time-Series Database:** Stores historical and forecast data in **InfluxDB v2** via Docker
- **AI Forecasting:** Uses a trained LSTM model to predict future pollution levels
- **Data Integrity:** Implements nuclear-cleaning logic and raw integer timestamps to prevent duplication

---

## 🛠️ Tech Stack

- **Language:** Python 3.12 (Pandas, TensorFlow, InfluxDB Client)
- **Database:** InfluxDB v2
- **Containerization:** Docker & Docker Compose
- **Model:** Keras / TensorFlow (LSTM)

---

## 📂 Project Structure

```text
AirSense-London-AI/
├── notebooks/                  
│   ├── LSTM_Air_Quality.ipynb   # Model training & experimentation
│   └── README.md               # Research & methodology documentation
├── docker-compose.yml          # InfluxDB configuration
├── live_forecaster.py          # Main production pipeline
├── Model_A_6h_final.keras      # Trained LSTM model
├── scaler.pkl                  # Feature scaler
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```
## 📦 Setup & Installation

### 1. Clone the Repository

    git clone https://github.com/hayat8hayat/AirSense-London-AI.git
    cd AirSense-London-AI

### 2. Configure Environment Variables

    cp .env.example .env

Edit the `.env` file and add:
- Your OpenAQ API Key
- InfluxDB credentials

### 3. Start InfluxDB (Docker)

    docker-compose up -d

InfluxDB Dashboard:
- URL: http://localhost:8086
- Username: admin_airsense
- Password: AirSense2025!

### 4. Install Python Dependencies

    python -m venv venv

Activate the virtual environment:

Windows:

    .\venv\Scripts\activate

macOS / Linux:

    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

### 5. Run the Forecasting Pipeline

    python live_forecaster.py

