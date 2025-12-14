import os
import sys
import time
import logging
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from influxdb_client.domain.write_precision import WritePrecision

# InfluxDB and TensorFlow libraries
from influxdb_client import InfluxDBClient, Point, WriteOptions
from tensorflow.keras.models import load_model

# ==============================================================================
# ⚙️ CONFIGURATION & SETUP
# ==============================================================================

# 1. Load environment variables
load_dotenv()

# Database Settings
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
ORG = os.getenv("INFLUX_ORG")
BUCKET = os.getenv("INFLUX_BUCKET")

# API Settings
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
STATION_ID = int(os.getenv("STATION_ID", 148))

# Constants
LOG_FILE = os.getenv("LOG_FILE", "system_backlog.log")
SENSORS = {"no2": 238, "o3": 229, "pm10": 244, "pm25": 232, "so2": 4933}
N_INPUT_HOURS = 6   # Number of past hours needed for prediction
N_FORECAST_HOURS = 3 # Number of future hours to predict

# 2. Configure Logging
# Using UTF-8 to ensure emojis print correctly on Windows
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, console_handler]
)

# ==============================================================================
# 🚀 INITIALIZATION
# ==============================================================================

def initialize_system():
    """
    Initializes database connections and loads ML models.
    Returns: client, write_api, model, scaler
    """
    try:
        logging.info("🚀 System Startup: Initializing resources...")

        # Initialize InfluxDB Client
        logging.info("🔌 Connecting to InfluxDB...")
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
        write_api = client.write_api(write_options=WriteOptions(batch_size=1))
        
        # Load ML Artifacts
        logging.info(f"🧠 Loading AI Model: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        
        logging.info(f"⚖️  Loading Scaler: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)

        logging.info("✅ Initialization complete. System is Healthy.")
        return client, write_api, model, scaler

    except Exception as e:
        logging.critical(f"❌ CRITICAL ERROR during Startup: {e}")
        sys.exit(1)

# Global resources
client, write_api, model, scaler = initialize_system()

# ==============================================================================
# 🧠 CORE LOGIC
# ==============================================================================

def fetch_sensor_data(start_time_str):
    """
    Fetches raw sensor data from OpenAQ API for all configured sensors.
    """
    raw_data = []
    
    for name, sensor_id in SENSORS.items():
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
        headers = {"X-API-Key": OPENAQ_API_KEY}
        params = {"limit": 100, "datetime_from": start_time_str}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for entry in results:
                    raw_data.append({
                        "datetime": entry['period']['datetimeTo']['utc'],
                        "pollutant": name,
                        "value": entry['value']
                    })
            else:
                logging.warning(f"⚠️ API Warning for {name}: Status {response.status_code}")
        except Exception as e:
            logging.error(f"❌ Connection failed for sensor {name}: {e}")
            
    return raw_data

def process_data(raw_data):
    """
    Cleans, pivots, and resamples raw data to hourly intervals.
    Returns a DataFrame containing exactly the last N_INPUT_HOURS rows.
    """
    df = pd.DataFrame(raw_data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Floor timestamps to the nearest hour (e.g., 13:00:10 -> 13:00:00)
    df['datetime'] = df['datetime'].dt.floor('h')
    logging.info(f"🧹 Data cleaned and timestamps floored to nearest hour.")
    
    # Pivot to wide format (Columns: Pollutants, Index: Time)
    df_wide = df.pivot_table(index='datetime', columns='pollutant', values='value', aggfunc='mean')
    
    # Resample to ensure continuous hourly index, filling small gaps
    df_clean = df_wide.resample('1h').mean().interpolate(method='time', limit=2).dropna()
    logging.info(f"📊 Data pivoted and resampled. Total hours available: {len(df_clean)}")
    
    # Verify we have enough data history
    if len(df_clean) < N_INPUT_HOURS:
        logging.warning(f"⚠️ Insufficient history. Found {len(df_clean)} hours, required {N_INPUT_HOURS}. Waiting...")
        return None
        
    # Return only the relevant tail (last 6 hours)
    logging.info(f"✅ Sufficient data available. Proceeding with last {N_INPUT_HOURS} hours.")
    return df_clean.tail(N_INPUT_HOURS)

def store_historical_data(df):
    try:
        # 1. DELETE SAFETY: Define window to clean
        # Convert to string just for the delete query (API requires string here)
        start_time = df.index[0].strftime('%Y-%m-%dT%H:00:00Z')
        stop_time = (df.index[-1] + timedelta(hours=1)).strftime('%Y-%m-%dT%H:00:00Z')
        
        logging.info(f"🧹 Nuclear Cleaning Window: {start_time} to {stop_time}")
        
        delete_api = client.delete_api()
        delete_api.delete(
            start=start_time, 
            stop=stop_time, 
            predicate='_measurement="air_quality" AND data_type="real"', 
            bucket=BUCKET, 
            org=ORG
        )

        logging.info(f"📤 Uploading {len(df)} rows of clean data...")
        
        points = []
        for timestamp, row in df.iterrows():
            # 🔨 THE SILVER BULLET: Use Raw Integer (Nanoseconds)
            # timestamp.value returns the exact nanoseconds since 1970
            nano_time = int(timestamp.value)
            
            point = Point("air_quality") \
                .tag("station_id", str(STATION_ID)) \
                .tag("data_type", "real") \
                .time(nano_time, WritePrecision.NS)  # <--- FORCE NANOSECONDS
            
            for col in df.columns:
                point.field(col, row[col])
            points.append(point)
            
        write_api.write(bucket=BUCKET, org=ORG, record=points)
        write_api.flush()
        
    except Exception as e:
        logging.error(f"❌ Failed to store historical data: {e}")
        raise e
    

def generate_and_store_forecast(input_df):
    try:
        # Prepare and Predict
        input_scaled = scaler.transform(input_df)
        X_input = input_scaled.reshape(1, N_INPUT_HOURS, 5)
        pred_scaled = model.predict(X_input, verbose=0)
        pred_values = scaler.inverse_transform(pred_scaled.reshape(N_FORECAST_HOURS, 5))
        
        # Calculate Future Times
        last_real_time = input_df.index[-1]
        future_dates = [last_real_time + timedelta(hours=i+1) for i in range(N_FORECAST_HOURS)]
        pollutants = input_df.columns.tolist()
        
        # 1. Clean Old Forecasts
        start_pred = future_dates[0].strftime('%Y-%m-%dT%H:00:00Z')
        stop_pred = (future_dates[-1] + timedelta(hours=1)).strftime('%Y-%m-%dT%H:00:00Z')
        
        logging.info(f"🧹 Nuclear Cleaning Forecast Window: {start_pred} to {stop_pred}")
        
        delete_api = client.delete_api()
        delete_api.delete(
            start=start_pred, 
            stop=stop_pred, 
            predicate='_measurement="air_quality" AND data_type="forecast"', 
            bucket=BUCKET, 
            org=ORG
        )

        # 2. Write New Forecasts
        points = []
        for i in range(N_FORECAST_HOURS):
            # 🔨 THE SILVER BULLET: Use Raw Integer (Nanoseconds)
            nano_time = int(future_dates[i].value)
            
            vals = pred_values[i]
            point = Point("air_quality") \
                .tag("station_id", str(STATION_ID)) \
                .tag("data_type", "forecast") \
                .time(nano_time, WritePrecision.NS) # <--- FORCE NANOSECONDS
            
            for j, pol in enumerate(pollutants):
                point.field(pol, vals[j])
            points.append(point)
            
        write_api.write(bucket=BUCKET, org=ORG, record=points)
        write_api.flush()
        logging.info("🔮 Forecast generated and stored.")

    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        raise e


def clear_bucket():
    """
    Deletes all data in the bucket to start fresh.
    """
    try:
        logging.info("🧹 Clearing old data from the database...")
        delete_api = client.delete_api()
        
        # Delete data from year 1970 to year 2100 (basically everything)
        start = "1970-01-01T00:00:00Z"
        stop = "2100-01-01T00:00:00Z"
        
        delete_api.delete(start, stop, '_measurement="air_quality"', bucket=BUCKET, org=ORG)
        logging.info("✨ Database is clean!")
        
    except Exception as e:
        logging.error(f"❌ Failed to clear database: {e}")


def run_cycle():
    """
    Orchestrates the main execution cycle.
    """
    now_utc = datetime.now(timezone.utc)
    logging.info(f"🔄 Starting execution cycle at {now_utc.strftime('%H:%M:%S')} UTC")
    
    # 1. Fetch Data
    fetch_start = (now_utc - timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
    raw_data = fetch_sensor_data(fetch_start)
    
    if not raw_data:
        logging.error("❌ No data retrieved. Aborting cycle.")
        return

    # 2. Process Data
    clean_df = process_data(raw_data)
    if clean_df is None:
        return
    
    logging.info(f"✅ Data processing complete. Proceeding to storage and forecasting.")
    # 3. Store Historical Data (Sync)
    store_historical_data(clean_df)

    # 4. Generate & Store Forecast
    generate_and_store_forecast(clean_df)
    
    logging.info("✅ Cycle completed successfully.")

# ==============================================================================
# ⏱️ MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Run immediately on startup
    clear_bucket()
    run_cycle()
    
    # Schedule hourly execution
    while True:
        print("\n💤 Sleeping for 60 minutes...")
        time.sleep(3600)
        run_cycle()