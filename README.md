# Sales-prediction-project
# Retail Sales Forecasting using LSTM

**End-to-end LSTM model** for weekly retail sales prediction incorporating external features (holidays, temperature, fuel price, CPI, unemployment).

## Features
- Comprehensive EDA
- Feature engineering (lags, rolling stats, date features, scaling)
- Univariate + Multivariate LSTM
- Baseline comparison (Naive, Linear Regression)
- RMSE improvement tracking
- Hyperparameter tuning

## Dataset
Walmart Sales Dataset [](https://www.kaggle.com/datasets/yasserh/walmart-dataset)  
**Columns**: `Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment`

## Results
- **Baseline RMSE**: ~450k (Naive)
- **LSTM RMSE**: **~220k** (significant improvement)

## How to Run
```bash
pip install -r requirements.txt
python main.py
