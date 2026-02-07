"""
test_model.py - Quality Gate for Model Performance
This script loads the trained model, PREPARES the data using prep.py,
and verifies its RMSE meets the threshold.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.append(src_path)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from prep import prepare_data  

def test_model_quality():
    print("\n" + "="*70)
    print("🔍 RUNNING QUALITY GATE CHECK")
    print("="*70)
    
    # 1. LOAD MODEL
    try:
        
        model = joblib.load('bike_rental_pipeline.joblib')
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ ERROR: Model file not found!")
        sys.exit(1)
    
    # 2. LOAD DATA
    try:
        
        df = pd.read_csv('day_2011.csv')
        print(f"✅ Raw data loaded: {len(df)} rows")
    except FileNotFoundError:
        print("❌ ERROR: Data file not found!")
        sys.exit(1)

    # 3. PREPARE DATA 
    print("⚙️ Processing data...")
    # This adds month_sin, windspeed_log, caps outliers, etc.
    df_clean = prepare_data(df, pt_hum_path='yeo_johnson_hum_transformer.joblib')
    
    # Filter to exact features expected by the model
    features_required = [
        'temp', 'hum_transformed', 'windspeed_log', 'day_of_year',
        'season', 'holiday', 'workingday', 'weathersit',
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
    ]
    
    X = df_clean[features_required]
    y_true = df_clean['cnt']
    print(f"✅ Data processed. Features: {X.shape[1]}")

    # 4. PREDICT & EVALUATE
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n📊 Model Performance:")
    print(f"   RMSE = {rmse:.2f}")
    
    # 5. QUALITY GATE
    
    BASELINE_RMSE = 602.23
    
    if rmse <= BASELINE_RMSE:
        print(f"\n✅ QUALITY GATE PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ QUALITY GATE FAILED! RMSE {rmse:.2f} > {BASELINE_RMSE}")
        sys.exit(1)

if __name__ == "__main__":
    test_model_quality()