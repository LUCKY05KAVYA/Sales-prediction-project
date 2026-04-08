import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_and_preprocess_data(data_path='data/Walmart_Sales.csv'):
    """
    Load and perform initial preprocessing on Walmart Sales dataset
    """
    # Load data
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Sort by Store and Date (important for time series)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    print(f"Data Loaded: {df.shape}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Number of Stores: {df['Store'].nunique()}")
    
    # Basic cleaning
    df['Holiday_Flag'] = df['Holiday_Flag'].astype(int)
    
    # Check missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Found {missing} missing values. Filling with forward fill.")
        df = df.fillna(method='ffill')
    else:
        print("No missing values found.")
    
    return df


def prepare_data_for_lstm(df, target='Weekly_Sales', seq_length=12):
    """
    Final preparation: scaling + sequence creation for LSTM
    """
    # We will use features.py for feature engineering
    from src.features import preprocess_for_lstm
    
    print("🔧 Performing Feature Engineering...")
    X_scaled, y_scaled, scaler_features, scaler_target, feature_cols = preprocess_for_lstm(df, target)
    
    print(f"Feature Engineering Completed. Total Features: {X_scaled.shape[1]}")
    
    # Create sequences
    from src.utils import create_sequences
    
    X_seq, y_seq = create_sequences(X_scaled, seq_length)
    y_seq = y_scaled[seq_length:]   # Align targets
    
    print(f"Sequences Created → X: {X_seq.shape}, y: {y_seq.shape}")
    
    return X_seq, y_seq, scaler_features, scaler_target, feature_cols


def save_preprocessing_objects(scaler_features, scaler_target, feature_cols):
    """Save scalers and feature list for future inference"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler_features, 'models/scaler_features.pkl')
    joblib.dump(scaler_target, 'models/scaler_target.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print("Scalers and feature list saved in 'models/' folder")


# Optional: Quick test function
if __name__ == "__main__":
    df = load_and_preprocess_data()
    print("\nSample Data:")
    print(df.head())
