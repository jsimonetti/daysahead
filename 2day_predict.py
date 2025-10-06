"""
Quarterly Electricity Price Prediction for the Netherlands
- ENTSO-E Hourly Day-Ahead Prices
- KNMI Historical Weather Data (forward-filled)
- Meteoserver.nl GFS Forecast for 48-hour prediction
- LightGBM Model
- Iterative 2-Day Forecast Using Meteoserver GFS + Actual Prices
- Predicted prices shown in EUR/kWh
"""

import pandas as pd
import warnings
from entsoe import EntsoePandasClient
import knmi
import meteoserver.weatherforecast as meteo
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
import tzlocal

warnings.filterwarnings("ignore")

# ------------------------
# 1. Settings
# ------------------------
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
if not ENTSOE_API_KEY:
    raise ValueError("Please set the ENTSOE_API_KEY environment variable.")
METEOSERVER_API_KEY = os.getenv("METEOSERVER_API_KEY")
if not METEOSERVER_API_KEY:
    raise ValueError("Please set the METEOSERVER_API_KEY environment variable.")
LOCATION = os.getenv("LOCATION")
if not LOCATION:
    raise ValueError("Please set the LOCATION environment to a near city name.")

COUNTRY_CODE = "NL"
STATION_CODE = 260  # De Bilt
MODEL_FILE = "nl_price_model_quarterly.pkl"
CACHE_FILE = "nl_entsoe_knmi_merged.parquet"
FORECAST_CACHE_FILE = "nl_meteoserver_forecast.parquet"

# ------------------------
# 2. Data Loading Functions
# ------------------------
def fetch_entsoe_prices_quarterly(api_key, country_code, start, end):
    print("Fetching ENTSO-E historical data...")
    client = EntsoePandasClient(api_key=api_key)
    prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    prices = prices.rename("price_eur_mwh").to_frame()
    prices = prices.asfreq(pd.tseries.offsets.Minute(15),'ffill')

    # Convert to local timezone
    prices = normalize_to_utc(prices)

    return prices

def fetch_knmi_weather_quarterly(station, start, end):
    print("Fetching KNMI historical data...")
    df_daily = knmi.get_day_data_dataframe(
        stations=[station],
        start=start,
        end=end,
        variables=["TG", "RH", "SQ", "FHX"]
    )
    df_daily["TG"] = df_daily["TG"] / 10.0
    df_daily["RH"] = df_daily["RH"] / 10.0
    df_daily["FHX"] = df_daily["FHX"] / 10.0
    df_daily = df_daily.rename(columns={
        "TG": "temp_avg",
        "RH": "precip_mm",
        "SQ": "sunshine_min",
        "FHX": "wind_speed"
    })

    df_hourly = pd.DataFrame(
        index=pd.date_range(start=df_daily.index.min(), end=df_daily.index.max() + pd.Timedelta(days=1), freq="H")[:-1]
    )

    for col in df_daily.columns:
        df_hourly[col] = df_daily[col].reindex(df_hourly.index, method='ffill')

    df_hourly = normalize_to_utc(df_hourly)
    return df_hourly.asfreq(pd.tseries.offsets.Minute(15),'ffill')

 
def load_and_merge_data_quarterly(cache_file=CACHE_FILE, timezone="Europe/Amsterdam"):
    # Check if cached file exists
    if os.path.exists(cache_file):
        print(f"Loading cached merged data from {cache_file}...")
        df = pd.read_parquet(cache_file)
        return df

    print("No cached data found. Fetching from APIs...")

    today_entso = pd.Timestamp.today(tz=timezone)
    today_knmi = datetime.today().strftime("%Y%m%d")

    start_entsoe = pd.Timestamp.today(tz=timezone) - pd.Timedelta(days=365)
    start_knmi = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")

    prices = fetch_entsoe_prices_quarterly(ENTSOE_API_KEY, COUNTRY_CODE, start_entsoe, today_entso)
    weather = fetch_knmi_weather_quarterly(STATION_CODE, start_knmi, today_knmi)

    df = prices.join(weather, how="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping data found. Check date ranges or API key.")

    # Feature Engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    for lag in [1, 2, 3, 96]:
        df[f'price_lag_{lag}'] = df['price_eur_mwh'].shift(lag)

    df['price_ma_3'] = df['price_eur_mwh'].shift(1).rolling(window=3).mean()
    df['price_ma_96'] = df['price_eur_mwh'].shift(1).rolling(window=96).mean()
    df['price_std_96'] = df['price_eur_mwh'].shift(1).rolling(window=96).std()

    df = df.dropna()

    # Save to cache
    df.to_parquet(cache_file)
    print(f"Merged data saved to cache: {cache_file}")

    return df

# ------------------------
# 3. Train & Evaluate Model
# ------------------------
def train_model_quarterly(df):
    print("Training model...")
    feature_cols = [
        "temp_avg", "precip_mm", "sunshine_min", "wind_speed",
        "hour", "day_of_week", "month", "is_weekend",
        "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_96",
        "price_ma_3", "price_ma_96", "price_std_96"
    ]

    X = df[feature_cols]
    y = df['price_eur_mwh']

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            min_gain_to_split=0.0,
            min_child_samples=5,
            max_depth=7,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))

    final_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        min_gain_to_split=0.0,
        min_child_samples=5,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Cross-validated MAE: {sum(maes)/len(maes):.2f} EUR/MWh")

    return final_model

# ------------------------
# 4. Fetch Meteoserver GFS Forecast
# ------------------------
def fetch_meteoserver_forecast(location, hours=48, cache_file=FORECAST_CACHE_FILE):
    # Check if cached forecast exists
    if os.path.exists(cache_file):
        print(f"Loading cached forecast from {cache_file}...")
        forecast_df = pd.read_parquet(cache_file)
        # Filter only the next `hours` for consistency
        forecast_df = forecast_df[forecast_df.index <= pd.Timestamp.now() + pd.Timedelta(hours=hours)]
        forecast_df =  normalize_to_utc(forecast_df)
        return forecast_df

    print("No cached forecast found. Fetching from Meteoserver API...")
    forecast_df = meteo.read_json_url_weatherforecast(
        key=METEOSERVER_API_KEY,
        location=location,
        model='GFS'
    )

    forecast_df['tijd'] = pd.to_datetime(forecast_df['tijd'], unit='s')
    forecast_df.set_index('tijd', inplace=True)
    forecast_df = forecast_df[forecast_df.index <= pd.Timestamp.now() + pd.Timedelta(hours=hours)]

    forecast_df = forecast_df.rename(columns={
        'temp2m': 'temp_avg',
        'neerslag': 'precip_mm',
        'windsnelheid10m': 'wind_speed',
        'zonuren': 'sunshine_min'
    })

    for col in ["temp_avg", "precip_mm", "wind_speed", "sunshine_min"]:
        if col not in forecast_df.columns:
            forecast_df[col] = 0
        forecast_df[col] = pd.to_numeric(forecast_df[col], errors='coerce')

    forecast_df = forecast_df[['temp_avg', 'precip_mm', 'wind_speed', 'sunshine_min']].iloc[:hours].asfreq(pd.tseries.offsets.Minute(15),'ffill')

    # Save to cache
    forecast_df.to_parquet(cache_file)
    forecast_df =  normalize_to_utc(forecast_df)
    print(f"Forecast saved to cache: {cache_file}")

    return forecast_df

# ------------------------
# 5. Iterative 2-Day Forecast with Actuals
# ------------------------
def predict_future_2days_with_actuals(model, df_hist, hours=48, timezone=tzlocal.get_localzone()):
    print(f"Predicting next {hours} hours with actual day-ahead prices...")

    future_weather = fetch_meteoserver_forecast(LOCATION, hours=hours)

    # Add temporal features
    future_weather['hour'] = future_weather.index.hour
    future_weather['day_of_week'] = future_weather.index.dayofweek
    future_weather['month'] = future_weather.index.month
    future_weather['is_weekend'] = future_weather['day_of_week'].isin([5,6]).astype(int)

    # Actual day-ahead prices for first 24 hours
    start_da = future_weather.index.min()
    end_da = start_da + pd.Timedelta(hours=24)
    # Ensure start and end are timezone-aware
    if start_da.tzinfo is None:
        start_da = start_da.tz_localize(timezone)
    if end_da.tzinfo is None:
        end_da = end_da.tz_localize(timezone)

    actual_prices = fetch_entsoe_prices_quarterly(
        ENTSOE_API_KEY, COUNTRY_CODE, start=start_da, end=end_da
    ).asfreq(pd.tseries.offsets.Minute(15),'ffill')['price_eur_mwh']

    actual_prices = normalize_to_utc(actual_prices)

    # Initialize lags
    last_row = df_hist.iloc[-1]
    lag_1 = last_row['price_eur_mwh']
    lag_2 = last_row['price_lag_1']
    lag_3 = last_row['price_lag_2']
    rolling_96 = list(df_hist['price_eur_mwh'].iloc[-96:])
    ma_3 = last_row['price_ma_3']

    predicted_prices = []

    for idx, row in future_weather.iterrows():
        ma_96 = sum(rolling_96[-96:]) / 96
        std_96 = pd.Series(rolling_96[-96:]).std()

        if idx in actual_prices.index:
            pred = actual_prices.loc[idx]
        else:
            X = pd.DataFrame([{
                "temp_avg": row['temp_avg'],
                "precip_mm": row['precip_mm'],
                "sunshine_min": row['sunshine_min'],
                "wind_speed": row['wind_speed'],
                "hour": row['hour'],
                "day_of_week": row['day_of_week'],
                "month": row['month'],
                "is_weekend": row['is_weekend'],
                "price_lag_1": lag_1,
                "price_lag_2": lag_2,
                "price_lag_3": lag_3,
                "price_lag_96": rolling_96[-96],
                "price_ma_3": ma_3,
                "price_ma_96": ma_96,
                "price_std_96": std_96
            }])
            pred = model.predict(X)[0]

        predicted_prices.append(pred)

        # Update lags and rolling window
        lag_3, lag_2, lag_1 = lag_2, lag_1, pred
        rolling_96.append(pred)
        if len(rolling_96) > 96:
            rolling_96 = rolling_96[-96:]
        ma_3 = (lag_1 + lag_2 + lag_3) / 3

    # Convert to EUR/kWh
    future_weather['predicted_price_eur_kwh'] = [p / 1000 for p in predicted_prices]

    future_weather.index = future_weather.index.tz_convert(timezone)

    return future_weather[['predicted_price_eur_kwh']]


def normalize_to_utc(df, totimezone="UTC"):
    """
    Converts all datetime columns in a pandas DataFrame to UTC timezone.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with all datetime columns converted to UTC
    """
    df = df.copy()  # avoid modifying the original DataFrame
    local_tz = tzlocal.get_localzone()
    
    # Convert index if it's a datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize(local_tz)
        
        df.index = df.index.tz_convert(totimezone)

    return df

# ------------------------
# 6. Main Function
# ------------------------
def main():
    # Load historical data
    df = load_and_merge_data_quarterly()
    print(f"Loaded dataset with {len(df)} entries.")

    model = train_model_quarterly(df)

    # Future 48-hour forecast
    future_preds = predict_future_2days_with_actuals(model, df)
    print("\nPredictions for the next 48 hours (EUR/kWh):")
    print(future_preds.round(4))

    future_preds.to_csv("nl_price_forecast_2days_quarterly.csv")
    print("\nForecast saved to nl_price_forecast_2days_quarterly.csv")

# ------------------------
# 7. Script Entry Point
# ------------------------
if __name__ == "__main__":
    main()