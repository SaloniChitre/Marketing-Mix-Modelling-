# 📊 Marketing Analytics & MMM AI Dashboard

This project is an end-to-end Data Science solution designed to analyze marketing performance and optimize ROI. It combines traditional econometric techniques with modern Large Language Models (LLMs) to provide both statistical rigor and actionable business insights.

## 🎯 Project Overview
In response to the growing complexity of marketing attribution, this dashboard simulates a marketing environment to:
1.  **Forecast Sales:** Using advanced time-series modeling.
2.  **Measure Impact:** Quantifying the "lift" from exogenous variables like Price and Promotions.
3.  **Automate Insights:** Using a Generative AI agent to translate complex data into strategic recommendations.

## 🚀 Key Technical Features

### 1. Econometric & Statistical Modeling
* **Model Selection:** Developed and compared multiple models, including **ARIMA**, **Prophet**, and **SARIMAX**.
* **Exogenous Integration:** Implemented a **SARIMAX** model that incorporates external factors (Price, Seasonality, and Marketing Spend) to reduce Mean Absolute Error (MAE) by ~20% compared to univariate models.
* **Validation Strategy:** Employed a **TimeSeriesSplit (Expanding Window)** cross-validation methodology. This ensures the model is validated chronologically, preventing data leakage and ensuring reliability in a business forecasting context.



### 2. AI-Powered Analytics (LangChain)
* **Agentic Framework:** Integrated a **LangChain** agent using `gpt-4o-mini` to act as a "Virtual Marketing Consultant."
* **Natural Language Generation:** The agent analyzes model performance metrics and provides data-driven budget allocation suggestions and ROI optimization strategies.

### 3. Interactive Visualization
* **Dynamic Dashboard:** Built with **Streamlit** to communicate insights clearly to stakeholders.
* **Data Storytelling:** Utilized **Plotly** for interactive time-series forecasts and **Matplotlib** for model benchmarking.

## 🛠️ Tech Stack
* **Language:** Python 3.12 (Pandas, NumPy, Scikit-learn)
* **Forecasting:** Statsmodels (SARIMAX), Prophet
* **AI/LLM:** LangChain, OpenAI API
* **Visualization:** Streamlit, Plotly, Matplotlib
* **Deployment/Dev:** VS Code, macOS, Git

## 📈 Performance Summary
* **Winning Model:** SARIMAX
* **Best MAE:** 911.57 (A 60% improvement over the Naive baseline)
* **Robustness:** Demonstrated stable performance across 4-fold rolling window validation.

## 📂 Project Structure
* `streamlit_app.py`: Main application script and UI logic.
* `.env`: Environment variables for secure API management (git-ignored).
* `requirements.txt`: Python dependencies.

---
**Developed by Saloni Chitre** *Targeting Data Science & Business Intelligence roles with a focus on marketing ROI and advanced analytics.*
