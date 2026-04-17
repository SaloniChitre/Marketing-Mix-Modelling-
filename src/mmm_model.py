import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# -------------------------------
# Load Data
# -------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "marketing_data.csv")

df = pd.read_csv(data_path)

# -------------------------------
# Adstock Function
# -------------------------------
def adstock(series, decay=0.5):
    result = []
    prev = 0
    for val in series:
        new_val = val + decay * prev
        result.append(new_val)
        prev = new_val
    return np.array(result)

# Apply adstock
df["tv_adstock"] = adstock(df["tv_spend"])
df["digital_adstock"] = adstock(df["digital_spend"])
df["social_adstock"] = adstock(df["social_spend"])

# -------------------------------
# Feature Engineering
# -------------------------------
df["time_index"] = np.arange(len(df))
df["sin_weekly"] = np.sin(2 * np.pi * df["time_index"] / 7)
df["cos_weekly"] = np.cos(2 * np.pi * df["time_index"] / 7)

# -------------------------------
# Log Transformations
# -------------------------------
df["log_sales"] = np.log(df["sales"] + 1)
df["log_tv"] = np.log(df["tv_adstock"] + 1)
df["log_digital"] = np.log(df["digital_adstock"] + 1)
df["log_social"] = np.log(df["social_adstock"] + 1)
df["log_price"] = np.log(df["price"] + 1)

# -------------------------------
# Model Features (EXOGENOUS ADDED)
# -------------------------------
X = df[[
    "log_tv",
    "log_digital",
    "log_social",
    "log_price",
    "promotion",
    "sin_weekly",
    "cos_weekly",
    "time_index"
]]

y = df["log_sales"]

# Add intercept
X = sm.add_constant(X)

# -------------------------------
# Train Model
# -------------------------------
model = sm.OLS(y, X).fit()

# -------------------------------
# Output
# -------------------------------
print("\n📊 MODEL SUMMARY:\n")
print(model.summary())

print("\n📈 INTERPRETATION:\n")
coeffs = model.params

for col in X.columns:
    if col != "const":
        print(f"{col}: {coeffs[col]:.4f}")