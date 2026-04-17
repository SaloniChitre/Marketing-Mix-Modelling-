import pandas as pd
import numpy as np
import os
from prophet import Prophet

# -------------------------------
# Load Data
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "marketing_data.csv")

df = pd.read_csv(data_path)

# Prophet requires specific column names
prophet_df = df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

# -------------------------------
# Train Forecasting Model
# -------------------------------
model = Prophet()
model.fit(prophet_df)

# Create future dates
future = model.make_future_dataframe(periods=90)  # next 3 months

forecast = model.predict(future)

# -------------------------------
# ROI Calculation
# -------------------------------

# Total spend per channel
total_tv = df["tv_spend"].sum()
total_digital = df["digital_spend"].sum()
total_social = df["social_spend"].sum()

total_sales = df["sales"].sum()

# Approx contribution (simple proportional method)
tv_contribution = total_sales * 0.3
digital_contribution = total_sales * 0.4
social_contribution = total_sales * 0.3

# ROI calculation
roi_tv = (tv_contribution - total_tv) / total_tv
roi_digital = (digital_contribution - total_digital) / total_digital
roi_social = (social_contribution - total_social) / total_social

# -------------------------------
# Output
# -------------------------------
print("\n📊 FORECAST (Next 5 rows):\n")
print(forecast[["ds", "yhat"]].tail())

print("\n💰 ROI BY CHANNEL:\n")
print(f"TV ROI: {roi_tv:.2f}")
print(f"Digital ROI: {roi_digital:.2f}")
print(f"Social ROI: {roi_social:.2f}")