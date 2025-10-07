# Netherlands Electricity Price Forecast

This repository contains a Python script for predicting **day-ahead electricity prices in the Netherlands** using historical electricity and weather data. The model leverages **LightGBM** regression and uses both **historical** and **forecasted weather data** for short-term predictions. The workflow includes caching to reduce repeated API calls and accelerate repeated runs.

---

## Description

The script performs the following:

- Fetches **historical electricity prices** from ENTSO-E.
- Retrieves **historical weather data** from KNMI.
- Uses **Meteoserver GFS forecasts** for the next 48 hours.
- Performs **feature engineering**, including lagged prices, rolling statistics, and temporal features.
- Trains a **LightGBM regression model** using time-series cross-validation.
- Predicts **short-term electricity prices** in EUR/kWh for the next 48 hours.
- Caches both historical and forecast data to avoid repeated API calls for development purposes.

---

## Features

- **Historical Data Sources**
  - ENTSO-E Day-Ahead Prices (quarterly, pre 15min data is forward filled)
  - KNMI Historical Weather Data (hourly, forward filled)
- **Forecast Data**
  - Meteoserver GFS 48-hour Forecast (hourly, forward filled)
- **Feature Engineering**
  - Lagged prices (`price_lag_1`, `price_lag_2`, `price_lag_3`, `price_lag_96`)
  - Rolling statistics (`price_ma_3`, `price_ma_96`, `price_std_96`)
  - Temporal features: `hour`, `day_of_week`, `month`, `is_weekend`
  - Weather features: temperature, precipitation, wind speed, sunshine
- **Modeling**
  - LightGBM Regressor with time-series cross-validation
- **Caching**
  - Historical merged data cached in `nl_entsoe_knmi_merged.parquet`
  - Meteoserver forecast cached in `nl_meteoserver_forecast.parquet`
  - Reduces repeated API calls and speeds up testing
- **Output**
  - Forecast CSV: `nl_price_forecast_2days_quarterly.csv`
  - Trained LightGBM model: `nl_price_model_quarterly.pkl`

---

## Installation

To install this script, clone this repository and install required python modules.

Example (Linux/macOS):
```bash
git clone https://github.com/jsimonetti/daysahead.git
cd daysahead
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Required Environment Variables

Before running the script, you need to set the following environment variables:

| Variable | Description |
|----------|-------------|
| `ENTSOE_API_KEY` | Your API key for ENTSO-E to fetch day-ahead electricity prices. |
| `METEOSERVER_API_KEY` | Your API key for Meteoserver to fetch weather forecasts. |

Example (Linux/macOS):
```bash
export ENTSOE_API_KEY="your_entsoe_api_key"
export METEOSERVER_API_KEY="your_meteoserver_api_key"
```

## Usage

1.	Run the script
```bash
python3 nl_price_forecast_quarterly.py
```
2.	Outputs
- nl_price_model_quarterly.pkl → Trained LightGBM model
- nl_price_forecast_2days_quarterly.csv → 48-hour forecast in EUR/kWh
- Cached data files (.parquet) for faster subsequent runs
3.	Caching
- nl_entsoe_knmi_merged.parquet → historical electricity + weather data
- nl_meteoserver_forecast.parquet → 48-hour weather forecast

## Notes
- Forecasts are short-term (next 48 hours) using iterative predictions.
- Lagged prices and rolling statistics are used to capture recent trends.
- Temporal features capture hourly, daily, and weekly seasonality.
- Caching ensures offline testing is possible after the first run.

## License

This project is licensed under the MIT License. See LICENSE for details.
