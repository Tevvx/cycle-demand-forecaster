import pandas as pd
import numpy as np
import joblib

def cap_outliers(df, col, method='iqr', threshold=1.5):
    """
    Caps outliers using IQR or Z-Score method.
    Used in Unit Tests.
    """
    df_clean = df.copy()
    if method == 'iqr':
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - (threshold * IQR)
        upper = Q3 + (threshold * IQR)
        df_clean[col] = np.where(df_clean[col] < lower, lower,
                        np.where(df_clean[col] > upper, upper, df_clean[col]))
    elif method == 'z-score':
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        lower = mean - (threshold * std)
        upper = mean + (threshold * std)
        df_clean[col] = np.where(df_clean[col] < lower, lower,
                        np.where(df_clean[col] > upper, upper, df_clean[col]))
    return df_clean

def get_cyclical_features(df):
    """
    Generates sin/cos encoding for month and weekday.
    Used in Unit Tests.
    """
    df = df.copy()
    
    # Ensure columns exist before processing
    if 'mnth' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
    
    if 'weekday' in df.columns:
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
    return df

def prepare_data(df, pt_hum_path='yeo_johnson_hum_transformer.joblib'):
    """
    Master function to prepare new data (2012 or User Input) 
    using the exact logic from Task 1.
    """
    df = df.copy()
    
    # 1. Date Features
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)
        df['day_of_year'] = df['dteday'].dt.dayofyear
    
    # 2. Cyclical Encoding (Calls the function above)
    df = get_cyclical_features(df)
    
    # 3. Outlier Capping (Calls the function above)
    if 'windspeed' in df.columns:
        df = cap_outliers(df, 'windspeed', method='iqr')
    if 'hum' in df.columns:
        df = cap_outliers(df, 'hum', method='iqr')
    if 'temp' in df.columns:
        df = cap_outliers(df, 'temp', method='z-score', threshold=3)
    
    # 4. Transformations
    if 'windspeed' in df.columns:
        df['windspeed_log'] = np.log1p(df['windspeed'])
    
    # 5. Yeo-Johnson (Load the fitted transformer)
    # Only try to load if humidity exists and we are not in a CI/CD test environment
    if 'hum' in df.columns:
        try:
            pt_hum = joblib.load(pt_hum_path)
            df['hum_transformed'] = pt_hum.transform(df[['hum']])
        except FileNotFoundError:
            # Fallback for CI/CD or when file isn't ready yet
            print(f"Warning: {pt_hum_path} not found. Skipping Yeo-Johnson transform.")
            df['hum_transformed'] = df['hum'] # Placeholder to prevent crash
            
    return df