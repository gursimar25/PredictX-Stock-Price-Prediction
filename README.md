# PredictX — Stock Price Prediction using Machine Learning

A Flask-based college mini-project for experimenting with historical market data and machine-learning models.

## Included features
- Login, registration, password hashing and CSRF protection
- Yahoo Finance data through `yfinance`
- Linear Regression, Random Forest and Gradient Boosting
- Chronological 80/20 validation with MAE, RMSE and R²
- Interactive OHLC candlestick chart
- SMA, EMA, RSI, MACD and Bollinger Bands
- Model comparison and next-price estimates
- Walk-forward backtesting with directional accuracy
- Stock autocomplete search
- Personal prediction history
- Watchlist
- Portfolio holding tracker
- Price-alert conditions and manual live checks
- Yahoo Finance news feed
- CSV and PDF history export
- Light/dark theme toggle
- Optional LSTM neural-network experiment
- Security headers, input validation and rate limiting

## Windows setup
Open the project from Local Disk (C:), then in the VS Code terminal run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:5000`.

### Optional LSTM
TensorFlow is intentionally optional because it is a large dependency. To enable the LSTM button:

```powershell
.venv\Scripts\python.exe -m pip install tensorflow
```

Restart Flask after installation.

## Notes
- The app is for educational/modeling purposes and is not investment advice.
- Backtesting is historical evaluation, not a guarantee of future performance.
- Do not commit a real `.env` file or credentials to GitHub.
