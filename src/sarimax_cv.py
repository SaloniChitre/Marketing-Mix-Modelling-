import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# -------------------------------
# Load Data
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "marketing_data.csv")

df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# -------------------------------
# Feature Engineering (same as MMM)
# -------------------------------
df["time_index"] = np.arange(len(df))
df["sin_weekly"] = np.sin(2 * np.pi * df["time_index"] / 7)
df["cos_weekly"] = np.cos(2 * np.pi * df["time_index"] / 7)

# Exogenous features
exog_features = ["tv_spend", "digital_spend", "social_spend", "price", "promotion",
                 "sin_weekly", "cos_weekly", "time_index"]

# -------------------------------
# Time Series Cross Validation
# -------------------------------
initial_train_size = int(len(df) * 0.6)
step_size = 30   # move window forward
forecast_horizon = 30

mae_scores = []

for start in range(initial_train_size, len(df) - forecast_horizon, step_size):
    
    train = df.iloc[:start]
    test = df.iloc[start:start + forecast_horizon]

    y_train = train["sales"]
    y_test = test["sales"]

    X_train = train[exog_features]
    X_test = test[exog_features]

    # -------------------------------
    # SARIMAX Model
    # -------------------------------
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=forecast_horizon, exog=X_test)

    mae = mean_absolute_error(y_test, forecast)
    mae_scores.append(mae)

    print(f"Fold MAE: {mae:.2f}")

# -------------------------------
# Final Results
# -------------------------------
print("\n📊 CROSS-VALIDATION RESULTS:\n")
print(f"Average MAE: {np.mean(mae_scores):.2f}")
print(f"Std Dev MAE: {np.std(mae_scores):.2f}")