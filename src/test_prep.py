import pytest
import pandas as pd
import numpy as np
from prep import cap_outliers, get_cyclical_features, prepare_data

# --- TEST 1: Logic Check (Does it actually catch outliers?) ---
def test_cap_outliers_iqr():
    # Create a dataframe with a massive outlier (1000)
    df = pd.DataFrame({'value': [1, 2, 3, 2, 2, 3, 1000]})
    
    # Run the function
    df_clean = cap_outliers(df, 'value', method='iqr')
    
    # Assert: The max value should be much smaller than 1000 now
    # (For this dataset, IQR is small, so 1000 will definitely be capped)
    assert df_clean['value'].max() < 1000
    print("✅ Outlier capping logic works")

# --- TEST 2: Math Check (Are Sin/Cos values valid?) ---
def test_cyclical_features_range():
    # Create dummy data with edge cases (Month 1, Month 12)
    df = pd.DataFrame({
        'mnth': [1, 6, 12],
        'weekday': [0, 3, 6]
    })
    
    df_processed = get_cyclical_features(df)
    
    # Assert: Trigonometric values must ALWAYS be between -1.0 and 1.0
    cols_to_check = ['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
    
    for col in cols_to_check:
        assert col in df_processed.columns
        assert df_processed[col].min() >= -1.0
        assert df_processed[col].max() <= 1.0
    print("✅ Cyclical features are within mathematical bounds")

# --- TEST 3: Integration Check (Does the master function run?) ---
def test_prepare_data_execution():
    """
    This tests the whole pipeline 'prepare_data'.
    It verifies that the code doesn't crash even if the joblib file is missing
    (which happens in GitHub Actions).
    """
    # Create a raw dataframe mimicking the 2012 dataset
    raw_df = pd.DataFrame({
        'dteday': ['01-01-2012', '02-01-2012'],
        'mnth': [1, 1],
        'weekday': [0, 1],
        'windspeed': [10.5, 12.0],
        'hum': [60.0, 55.0],
        'temp': [0.3, 0.35]
    })
    
    # Run the master function
    # Note: This will print the "Warning: joblib not found" but SHOULD NOT CRASH
    clean_df = prepare_data(raw_df)
    
    # Assert: Check if key columns were created
    assert 'day_of_year' in clean_df.columns
    assert 'windspeed_log' in clean_df.columns
    assert 'hum_transformed' in clean_df.columns
    
    # Check shape (Should have same number of rows)
    assert len(clean_df) == 2
    print("✅ Master pipeline runs successfully")