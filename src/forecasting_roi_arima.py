import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# -------------------------------
# Load Data
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "marketing_data.csv")

df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Set index
df.set_index("date", inplace=True)

# -------------------------------
# Check Stationarity (ADF Test)
# -------------------------------
result = adfuller(df["sales"])
print("\n📊 ADF Test:")
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")

# -------------------------------
# Train ARIMA Model
# -------------------------------
model = ARIMA(df["sales"], order=(1,1,1))
model_fit = model.fit()

print("\n📈 ARIMA MODEL SUMMARY:\n")
print(model_fit.summary())

# -------------------------------
# Forecast Future Sales
# -------------------------------
forecast_steps = 90
forecast = model_fit.forecast(steps=forecast_steps)

forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:]
forecast_df = pd.DataFrame({
    "date": forecast_index,
    "forecasted_sales": forecast.values
})

# -------------------------------
# ROI Calculation (Temporary)
# -------------------------------
total_tv = df["tv_spend"].sum()
total_digital = df["digital_spend"].sum()
total_social = df["social_spend"].sum()

total_sales = df["sales"].sum()

# Placeholder contribution split (will fix next step)
tv_contribution = total_sales * 0.3
digital_contribution = total_sales * 0.4
social_contribution = total_sales * 0.3

roi_tv = (tv_contribution - total_tv) / total_tv
roi_digital = (digital_contribution - total_digital) / total_digital
roi_social = (social_contribution - total_social) / total_social

# -------------------------------
# Output
# -------------------------------
print("\n📊 FORECAST SAMPLE:\n")
print(forecast_df.head())

print("\n💰 ROI BY CHANNEL:\n")
print(f"TV ROI: {roi_tv:.2f}")
print(f"Digital ROI: {roi_digital:.2f}")
print(f"Social ROI: {roi_social:.2f}")