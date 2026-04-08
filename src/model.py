from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def build_lstm_model(input_shape, units=64, dropout=0.3):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=80, batch_size=32):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                             restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

def evaluate_model(model, X_test, y_test, scaler_target):
    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler_target.inverse_transform(pred_scaled)
    actual = scaler_target.inverse_transform(y_test)
    
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    
    return rmse, mae, pred, actual
