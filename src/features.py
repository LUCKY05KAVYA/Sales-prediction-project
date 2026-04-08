


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_date_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    return df

def add_lag_features(df, target='Weekly_Sales', lags=[1, 2, 3, 4, 8, 12]):
    for lag in lags:
        df[f'Lag_{lag}'] = df.groupby('Store')[target].shift(lag)
    return df

def add_rolling_features(df, target='Weekly_Sales', windows=[4, 8, 12]):
    for w in windows:
        df[f'Roll_Mean_{w}'] = df.groupby('Store')[target].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean())
        df[f'Roll_Std_{w}'] = df.groupby('Store')[target].transform(
            lambda x: x.rolling(window=w, min_periods=1).std())
    return df

def preprocess_for_lstm(df, target='Weekly_Sales'):
    external_cols = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    
    df = add_date_features(df)
    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = external_cols + [col for col in df.columns 
                                  if col.startswith(('Lag_', 'Roll_', 'Year', 'Month', 'Week', 'Day', 'IsWeekend'))]
    
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    X_scaled = scaler_features.fit_transform(df[feature_cols])
    y_scaled = scaler_target.fit_transform(df[[target]])
    
    return X_scaled, y_scaled, scaler_features, scaler_target, feature_cols
