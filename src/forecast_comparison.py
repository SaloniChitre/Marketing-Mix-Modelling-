import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
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
# Train-Test Split
# -------------------------------
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# -------------------------------
# NAIVE MODEL
# -------------------------------
naive_forecast = np.repeat(train["sales"].iloc[-1], len(test))

# -------------------------------
# MEAN MODEL
# -------------------------------
mean_value = train["sales"].mean()
mean_forecast = np.repeat(mean_value, len(test))

# -------------------------------
# ARIMA
# -------------------------------
train_arima = train.set_index("date")

arima_model = ARIMA(train_arima["sales"], order=(1,1,1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))

# -------------------------------
# SARIMA (adds seasonality)
# -------------------------------
sarima_model = SARIMAX(train_arima["sales"],
                       order=(1,1,1),
                       seasonal_order=(1,1,1,7))  # weekly seasonality
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=len(test))

# -------------------------------
# PROPHET
# -------------------------------
prophet_train = train[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})

prophet_model = Prophet()
prophet_model.fit(prophet_train)

future = prophet_model.make_future_dataframe(periods=len(test))
forecast_prophet = prophet_model.predict(future)

prophet_forecast = forecast_prophet["yhat"].iloc[-len(test):].values

# -------------------------------
# EVALUATION (MAE)
# -------------------------------
y_true = test["sales"].values

mae_naive = mean_absolute_error(y_true, naive_forecast)
mae_mean = mean_absolute_error(y_true, mean_forecast)
mae_arima = mean_absolute_error(y_true, arima_forecast)
mae_sarima = mean_absolute_error(y_true, sarima_forecast)
mae_prophet = mean_absolute_error(y_true, prophet_forecast)

print("\n📊 MODEL COMPARISON (MAE):\n")
print(f"Naive MAE: {mae_naive:.2f}")
print(f"Mean MAE: {mae_mean:.2f}")
print(f"ARIMA MAE: {mae_arima:.2f}")
print(f"SARIMA MAE: {mae_sarima:.2f}")
print(f"Prophet MAE: {mae_prophet:.2f}")

# -------------------------------
# PLOTTING
# -------------------------------
plt.figure()

# Actual
plt.plot(test["date"], y_true, label="Actual")

# Forecasts
plt.plot(test["date"], naive_forecast, label="Naive")
plt.plot(test["date"], mean_forecast, label="Mean")
plt.plot(test["date"], arima_forecast, label="ARIMA")
plt.plot(test["date"], sarima_forecast, label="SARIMA")
plt.plot(test["date"], prophet_forecast, label="Prophet")

plt.title("Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()

plt.show()