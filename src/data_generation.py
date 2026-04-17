import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_data(n_days=365):
    dates = pd.date_range(start="2023-01-01", periods=n_days)

    # -------------------------------
    # Marketing spends
    # -------------------------------
    tv = np.random.uniform(1000, 5000, n_days)
    digital = np.random.uniform(500, 3000, n_days)
    social = np.random.uniform(300, 2000, n_days)

    # -------------------------------
    # External factors (EXOGENOUS)
    # -------------------------------
    price = np.random.uniform(10, 20, n_days)
    promotion = np.random.choice([0, 1], size=n_days, p=[0.7, 0.3])

    # -------------------------------
    # Trend + Seasonality
    # -------------------------------
    trend = np.linspace(10000, 20000, n_days)
    yearly_seasonality = 2000 * np.sin(2 * np.pi * dates.dayofyear / 365)
    weekly_seasonality = 1500 * np.sin(2 * np.pi * np.arange(n_days) / 7)

    # -------------------------------
    # True coefficients (hidden)
    # -------------------------------
    beta_tv = 0.05
    beta_digital = 0.08
    beta_social = 0.1

    # -------------------------------
    # Noise
    # -------------------------------
    noise = np.random.normal(0, 1000, n_days)

    # -------------------------------
    # Final Sales Equation
    # -------------------------------
    sales = (
        trend
        + yearly_seasonality
        + weekly_seasonality
        + beta_tv * tv
        + beta_digital * digital
        + beta_social * social
        - 200 * price
        + 3000 * promotion
        + noise
    )

    df = pd.DataFrame({
        "date": dates,
        "tv_spend": tv,
        "digital_spend": digital,
        "social_spend": social,
        "price": price,
        "promotion": promotion,
        "sales": sales
    })

    return df


if __name__ == "__main__":
    df = generate_data()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "marketing_data.csv")

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)

    print("✅ Data generated with exogenous features!")