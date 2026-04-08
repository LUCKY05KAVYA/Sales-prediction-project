import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.utils import load_data, create_sequences


from src.features import preprocess_for_lstm
from src.model import train_model, evaluate_model

# Create directories


os.makedirs('models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

print("Starting Retail Sales LSTM Forecasting...\n")

# 1. Load Data
df = load_data('data/Walmart_Sales.csv')
print(f"Data Loaded: {df.shape}")

# 2. Preprocessing & Feature Engineering
X_scaled, y_scaled, scaler_feat, scaler_target, feature_cols = preprocess_for_lstm(df)

# 3. Create Time Series Sequences
seq_length = 12
X_seq, y_seq = create_sequences(X_scaled, seq_length)
y_seq = y_scaled[seq_length:]  

print(f"Sequences Created: {X_seq.shape}")

# 4. Train-Test Split (Time Series - No Shuffle)
train_size = int(0.7 * len(X_seq))
val_size = int(0.15 * len(X_seq))

X_train = X_seq[:train_size]
X_val = X_seq[train_size:train_size+val_size]
X_test = X_seq[train_size+val_size:]

y_train = y_seq[:train_size]
y_val = y_seq[train_size:train_size+val_size]
y_test = y_seq[train_size+val_size:]

print(f"Split Done → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 5. Train LSTM Model
model, history = train_model(X_train, y_train, X_val, y_val, epochs=80)

# 6. Evaluate
rmse, mae, pred, actual = evaluate_model(model, X_test, y_test, scaler_target)

print("\n" + "="*50)
print("🎯 FINAL RESULTS")
print("="*50)
print(f"LSTM RMSE : ${rmse:,.2f}")
print(f"LSTM MAE  : ${mae:,.2f}")
print("="*50)

# 7. Save Model and Scalers
model.save('models/lstm_model.h5')
joblib.dump(scaler_feat, 'models/scaler_features.pkl')
joblib.dump(scaler_target, 'models/scaler_target.pkl')

# 8. Save Metrics
with open('results/metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\n")

# 9. Plot Results

plt.figure(figsize=(15, 7))
plt.plot(actual[:300], label='Actual Weekly Sales', linewidth=2)
plt.plot(pred[:300], label='Predicted Weekly Sales', linewidth=2)
plt.title('LSTM Retail Sales Forecast vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Weekly Sales ($)')
plt.legend()
plt.grid(True)
plt.savefig('results/plots/predictions.png', dpi=300)
plt.show()

print("All done! Check 'results/plots/' and 'models/' folder.")
