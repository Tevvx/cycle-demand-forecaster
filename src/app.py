import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# 1. SETUP PATHS (Robust Fix)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define Root Dir (Go up one level from src)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add 'src' to system path so we can import prep.py
sys.path.append(current_dir)
from prep import prepare_data

# DEFINE ASSET PATHS
MODEL_PATH = os.path.join(root_dir, 'models', 'bike_rental_pipeline.joblib')
TRANSFORMER_PATH = os.path.join(root_dir, 'models', 'yeo_johnson_hum_transformer.joblib')

# Failsafe: Check if files exist, if not try looking in current folder
if not os.path.exists(MODEL_PATH):
    # Fallback to current directory if not found in models/
    MODEL_PATH = os.path.join(current_dir, 'bike_rental_pipeline.joblib')

if not os.path.exists(TRANSFORMER_PATH):
    # Fallback to current directory
    TRANSFORMER_PATH = os.path.join(current_dir, 'yeo_johnson_hum_transformer.joblib')

# 2. Page Configuration
st.set_page_config(page_title="🚲 Bike Rental Forecaster", layout="wide")

# 3. Load Assets (Cached for speed)
@st.cache_resource
def load_assets():
    try:
        print(f"Loading model from: {MODEL_PATH}")
        
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"File not found: {MODEL_PATH}. Please check your folder structure.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_assets()

# 4. Sidebar Inputs
st.sidebar.header("User Input Features")

def user_input_features():
    # A. Date Components
    date_val = st.sidebar.date_input("Select Date", value=pd.to_datetime("2012-01-01"))
    
    # B. Weather Conditions
    weathersit = st.sidebar.selectbox("Weather Condition", 
        (1, 2, 3, 4), 
        format_func=lambda x: {
            1: "1: Clear / Few Clouds",
            2: "2: Mist / Cloudy",
            3: "3: Light Snow / Rain",
            4: "4: Heavy Rain / Ice"
        }[x])
    
    # C. Numeric Inputs
    temp = st.sidebar.slider("Temperature (°C)", -10.0, 40.0, 15.0)
    hum = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider("Windspeed (km/h)", 0.0, 60.0, 10.0)
    
    # D. Categorical Checks
    holiday = st.sidebar.checkbox("Is it a Holiday?", value=False)
    workingday = st.sidebar.checkbox("Is it a Working Day?", value=True)
    
    # Map inputs to DataFrame
    data = {
        'dteday': date_val,
        'mnth': date_val.month,
        'weekday': date_val.weekday(),
        'weathersit': weathersit,
        'temp': temp,
        'hum': hum,
        'windspeed': windspeed,
        'holiday': 1 if holiday else 0,
        'workingday': 1 if workingday else 0,
        'season': (date_val.month % 12 + 3) // 3 # Simple season approx
    }
    return pd.DataFrame([data])

# 5. Main App Logic
st.title("🚲 Intelligent Bike Demand Forecasting")
st.markdown("""
This app predicts the daily demand for bike rentals based on weather and calendar data.
**Adjust the sidebar** to test different scenarios.
""")

# Get raw input
input_df = user_input_features()

with st.expander("View Input Data"):
    st.write(input_df)

if st.button("Predict Demand"):
    if pipeline:
        try:
            # Apply Preprocessing (Creates new features, keeps old ones)
            processed_df = prepare_data(input_df, pt_hum_path=TRANSFORMER_PATH)
            
            # (Matches the list from Task 1 and test_model.py)
            model_features = [
                'temp', 
                'hum_transformed', 
                'windspeed_log', 
                'day_of_year', 
                'season', 
                'holiday', 
                'workingday', 
                'weathersit', 
                'month_sin', 
                'month_cos', 
                'weekday_sin', 
                'weekday_cos'
            ]
            
            # Check if all columns exist before predicting
            X_pred = processed_df[model_features]
            
            prediction = pipeline.predict(X_pred)
            
            pred_value = int(prediction[0])
            st.success(f"Predicted Rentals: {pred_value} bikes")
            
            # Delta metric
            avg_rentals = 3405 
            delta = int(pred_value - avg_rentals)
            st.metric(label="Forecast vs Average", value=f"{pred_value}", delta=f"{delta}")
            
        except KeyError as e:
            st.error(f"Missing feature error: The model expects {e}. Check prep.py logic.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Pipeline not loaded. Check model path.")