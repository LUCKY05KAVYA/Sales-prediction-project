



import pandas as pd
import numpy as np
import joblib

def load_data(path='data/Walmart_Sales.csv'):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df

def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)
