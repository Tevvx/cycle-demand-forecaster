"""
test_model.py - Quality Gate for Model Performance
This script loads the trained model, PREPARES the data using prep.py,
and verifies its RMSE meets the threshold.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Get the Src folder (For imports)
src_dir = os.path.join(root_dir, 'src')
sys.path.append(src_dir)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from prep import prepare_data  

MODEL_PATH = os.path.join(root_dir, 'bike_rental_pipeline.joblib')
DATA_PATH = os.path.join(root_dir, 'day_2011.csv')
TRANSFORMER_PATH = os.path.join(root_dir, 'yeo_johnson_hum_transformer.joblib')

def test_model_quality():
    print("\n" + "="*70)
    print("🔍 RUNNING QUALITY GATE CHECK")
    print("="*70)
    
    # 1. LOAD MODEL
    try:
        print(f"📂 Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    
    # 2. LOAD DATA
    try:
        print(f"📂 Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Raw data loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)

    # 3. PREPARE DATA 
    print("⚙️ Processing data...")
    # This adds month_sin, windspeed_log, caps outliers, etc.
    try:
        # Pass the absolute path to the transformer
        df_clean = prepare_data(df, pt_hum_path=TRANSFORMER_PATH)
    except Exception as e:
        print(f"❌ ERROR during data preparation: {e}")
        sys.exit(1)
    
    # Filter to exact features expected by the model
    features_required = [
        'temp', 'hum_transformed', 'windspeed_log', 'day_of_year',
        'season', 'holiday', 'workingday', 'weathersit',
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
    ]

    missing_cols = [col for col in features_required if col not in df_clean.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing columns after prep: {missing_cols}")
        sys.exit(1)
    
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