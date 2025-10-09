"""
Quarterly Electricity Price Prediction for the Netherlands
- ENTSO-E Hourly Day-Ahead Prices
- KNMI Historical Weather Data (forward-filled)
- Meteoserver.nl GFS Forecast for 48-hour prediction
- LightGBM Model
- Iterative 2-Day Forecast Using Meteoserver GFS + Actual Prices
- Predicted prices shown in EUR/kWh
"""

import warnings
import os
from datetime import datetime, timedelta
import joblib
import tzlocal
import holidays
from pathlib import Path
import time

import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
import knmi
import meteoserver.weatherforecast as meteo
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


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

COUNTRY_CODE = "NL"
STATION_CODE = 330  # Station(number=330, longitude=4.122, latitude=51.992, altitude=11.9, name='HOEK VAN HOLLAND')
LOCATION = "Hoek van Holland"
MODEL_FILE = "nl_price_model_quarterly.pkl"
CACHE_FILE = "nl_entsoe_knmi_merged.parquet"
DAY_AHEAD_CACHE_FILE = "nl_day_ahead.parquet"
FORECAST_CACHE_FILE = "nl_meteoserver_forecast.parquet"
WEEKS_TO_FETCH = 52
#TIMESERIES_N_SPLITS = 5  # Number of splits for TimeSeriesSplit cross-validation

# ------------------------
# 2. Data Loading Functions
# ------------------------
def fetch_entsoe_prices_quarterly(api_key, country_code, start, end):
    print("Fetching ENTSO-E data from", start, "to", end, "...")
    client = EntsoePandasClient(api_key=api_key)
    prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    prices = prices.rename("price_eur_mwh").to_frame()
    prices = prices.asfreq(pd.tseries.offsets.Minute(15),'ffill')

    # Convert to local timezone
    prices = normalize_to_utc(prices)

    return prices

def fetch_knmi_weather_quarterly(station, start, end):
    print("Fetching KNMI data from", start, "to", end, "...")
    df_hourly = knmi.get_hour_data_dataframe(
        stations=[station],
        start=start,
        end=end,
        variables=["T", "RH", "Q", "FH"]
    )
    df_hourly["T"] = df_hourly["T"] / 10.0 # convert to °C
    df_hourly["RH"] = df_hourly["RH"] / 10.0 # conver to mm
    df_hourly["FH"] = df_hourly["FH"] / 10.0 # convert to m/s
    df_hourly = df_hourly.rename(columns={
        "T": "temperature", # in °C
        "RH": "precipitation",  # in mm
        "Q": "sun_radiation", # in J/cm2
        "FH": "wind_speed" # in m/s
    })

    df_hourly = normalize_to_utc(df_hourly)
    return df_hourly.asfreq(pd.tseries.offsets.Minute(15),'ffill')

def fetch_knmi_weather_quarterlyOrg(station, start, end):
    print("Fetching KNMI data from", start, "to", end, "...")
    df_daily = knmi.get_day_data_dataframe(
        stations=[station],
        start=start,
        end=end,
        variables=["TX", "RH", "Q", "FHX"]
    )
    df_daily["TX"] = df_daily["TX"] / 10.0 # convert to °C
    df_daily["RH"] = df_daily["RH"] / 10.0 # conver to mm
    df_daily["FHX"] = df_daily["FHX"] / 10.0 # convert to m/s
    df_daily = df_daily.rename(columns={
        "TX": "temperature", # in °C
        "RH": "precipitation", # in mm
        "Q": "sun_radiation", # in J/cm2
        "FHX": "wind_speed" # in m/s
    })

    df_hourly = pd.DataFrame(
        index=pd.date_range(start=df_daily.index.min(), end=df_daily.index.max() + pd.Timedelta(days=1), freq="H")[:-1]
    )

    for col in df_daily.columns:
        df_hourly[col] = df_daily[col].reindex(df_hourly.index, method='ffill')

    df_hourly = normalize_to_utc(df_hourly)
    return df_hourly.asfreq(pd.tseries.offsets.Minute(15),'ffill')

def load_day_ahead(start_da, end_da, cache_file=DAY_AHEAD_CACHE_FILE):
    actual_prices = load_or_invalidate_parquet(cache_file, max_age_hours=12)
    # Check if cache read was successful
    if actual_prices is not None:
        return actual_prices.iloc[:, 0]
    
    actual_prices = fetch_entsoe_prices_quarterly(
        ENTSOE_API_KEY, COUNTRY_CODE, start=start_da, end=end_da
    ).asfreq(pd.tseries.offsets.Minute(15),'ffill')['price_eur_mwh']

    actual_prices = normalize_to_utc(actual_prices) 

    actual_prices.to_frame().to_parquet(cache_file)
    print(f"Data saved to cache: {cache_file}")
    return actual_prices

 
def load_and_merge_data_quarterly(cache_file=CACHE_FILE, timezone="Europe/Amsterdam"):
    df = load_or_invalidate_parquet(cache_file, max_age_hours=12)
    # Check if cache read was successful
    if df is not None:
        return df

    print("No cached data found. Fetching from APIs...")

    today_entso = pd.Timestamp.today(tz=timezone)
    today_knmi = datetime.today().strftime("%Y%m%d")

    start_entsoe = pd.Timestamp.today(tz=timezone) - pd.Timedelta(weeks=WEEKS_TO_FETCH)
    start_knmi = (datetime.today() - timedelta(weeks=WEEKS_TO_FETCH)).strftime("%Y%m%d")

    prices = fetch_entsoe_prices_quarterly(ENTSOE_API_KEY, COUNTRY_CODE, start_entsoe, today_entso)
    weather = fetch_knmi_weather_quarterly(STATION_CODE, start_knmi, today_knmi)


    df = prices.join(weather, how="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping data found. Check date ranges or API key.")

    # Feature Engineering
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    nl_holidays = holidays.NL(years=df.index.year.unique())
    df['is_holiday'] = df.index.normalize().isin(nl_holidays).astype(int)
    df['season'] = df.index.month.map({
        12: 0, 1: 0, 2: 0,    # winter
        3: 1, 4: 1, 5: 1,     # spring
        6: 2, 7: 2, 8: 2,     # summer
        9: 3, 10: 3, 11: 3    # autumn
    })
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
        "temperature", "precipitation", "sun_radiation", "wind_speed",
        "season", "day_of_week", "is_holiday", "hour", "is_weekend",
        "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_96",
        "price_ma_3", "price_ma_96", "price_std_96"
    ]

    X = df[feature_cols]
    y = df['price_eur_mwh']

    #tscv = TimeSeriesSplit(n_splits=TIMESERIES_N_SPLITS)
    #maes = []
    
    #for train_idx, test_idx in tscv.split(X):
    #    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    #    model = lgb.LGBMRegressor(
    #        n_estimators=1000,
    #        learning_rate=0.05,
    #        min_gain_to_split=0.0,
    #        min_child_samples=5,
    #        max_depth=7,
    #        random_state=42,
    #        verbose=-1
    #    )
    #    model.fit(X_train, y_train)
    #    preds = model.predict(X_test)
    #    maes.append(mean_absolute_error(y_test, preds))

    params = {
        'objective': 'regression',
        'num_leaves': 40,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,

        'learning_rate': 0.05,
        'n_estimators': 1000,
        'min_data_in_leaf': 50,
            #'min_gain_to_split':0.0,
            #'min_child_samples':5,
            #'max_depth':7,
            #'random_state':42,
            'verbose': -1
    }
    final_model = lgb.LGBMRegressor(**params)

    #final_model = lgb.LGBMRegressor(
    #    n_estimators=1000,
    #    learning_rate=0.05,
    #    min_gain_to_split=0.0,
    #    min_child_samples=5,
    #    max_depth=7,
    #    random_state=42,
    #    verbose=-1
    #)

    # Define weights — more recent → higher weight
    n = len(X)
    decay = 0.05  # smaller = slower decay, larger = faster
    weights = np.exp(np.linspace(-decay * n, 0, n))  # exponential weighting
    #weights = np.linspace(0.5, 1.0, n)
    #weights /= np.mean(weights)  # normalize to mean 1
    #weights = None

    final_model.fit(X, y, sample_weight=weights)
    joblib.dump(final_model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    return final_model

# ------------------------
# 4. Fetch Meteoserver GFS Forecast
# ------------------------
def fetch_meteoserver_forecast(location, hours=48, cache_file=FORECAST_CACHE_FILE):
    # Nieuwe data in deze API zijn beschikbaar om 5:30, 11:30, 17:30 en 23:30 Nederlandse tijd.
    forecast_df = load_or_invalidate_parquet(cache_file, max_age_hours=8)
    # Check if cache read was successful
    if forecast_df is not None:
        # Filter only the next `hours` for consistency
        forecast_df = forecast_df[forecast_df.index <= pd.Timestamp.now() + pd.Timedelta(hours=hours)]
        forecast_df =  normalize_to_utc(forecast_df)
        return forecast_df

    print("No cached forecast found. Fetching from Meteoserver API...")
    forecast_df = meteo.read_json_url_weatherforecast(
        key=METEOSERVER_API_KEY,
        location=location,
        model='HARMONIE'
    )

    forecast_df['tijd'] = pd.to_datetime(forecast_df['tijd'], unit='s')
    forecast_df.set_index('tijd', inplace=True)
    forecast_df = forecast_df[forecast_df.index <= pd.Timestamp.now() + pd.Timedelta(hours=hours)]

    forecast_df = forecast_df.rename(columns={
        'temp': 'temperature', # °C
        'neersl': 'precipitation', # mm
        'winds': 'wind_speed', # m/sec
        'gr': 'sun_radiation' # J/cm2
    })[['temperature', 'precipitation', 'wind_speed', 'sun_radiation']]

    forecast_df['temperature'] = pd.to_numeric(forecast_df['temperature'])
    forecast_df['precipitation'] = pd.to_numeric(forecast_df['precipitation'])
    forecast_df['sun_radiation'] = pd.to_numeric(forecast_df['sun_radiation'])
    forecast_df['wind_speed'] = pd.to_numeric(forecast_df['wind_speed'])

    forecast_df = forecast_df[['temperature', 'precipitation', 'wind_speed', 'sun_radiation']].iloc[:hours].asfreq(pd.tseries.offsets.Minute(15),'ffill')

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
    future_weather['day_of_week'] = future_weather.index.dayofweek
    future_weather['hour'] = future_weather.index.hour
    future_weather['season'] = future_weather.index.month.map({
        12: 0, 1: 0, 2: 0,    # winter
        3: 1, 4: 1, 5: 1,     # spring
        6: 2, 7: 2, 8: 2,     # summer
        9: 3, 10: 3, 11: 3    # autumn
    })
    future_weather['is_weekend'] = future_weather['day_of_week'].isin([5,6]).astype(int)
    nl_holidays = holidays.NL(years=future_weather.index.year.unique())
    future_weather['is_holiday'] = future_weather.index.normalize().isin(nl_holidays).astype(int)

    # Actual day-ahead prices 
    start_da = future_weather.index.min()
    end_da = start_da + pd.Timedelta(hours=48)
    # Ensure start and end are timezone-aware
    start_da = start_da.tz_convert(timezone)
    end_da = end_da.tz_convert(timezone)


    actual_prices = load_day_ahead(start_da, end_da)

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
                "temperature": row['temperature'],
                "precipitation": row['precipitation'],
                "sun_radiation": row['sun_radiation'],
                "wind_speed": row['wind_speed'],
                "day_of_week": row['day_of_week'],
                "hour": row['hour'],
                "season": row['season'],
                "is_holiday": row['is_holiday'],
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
    future_weather['actual_price_eur_kwh'] = actual_prices.reindex(future_weather.index).fillna(pd.NA) / 1000


    future_weather.index = future_weather.index.tz_convert(timezone)

    return future_weather[['predicted_price_eur_kwh', 'actual_price_eur_kwh','temperature','precipitation','wind_speed','sun_radiation']]


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
            df.index = df.index.tz_localize(
                local_tz,
                ambiguous='NaT',          # ambiguous times → NaT to avoid duplicates
                nonexistent='shift_forward' # nonexistent → shift forward
            )

        # Drop any rows with NaT index (ambiguous times)
        df = df[~df.index.isna()]
        
        df.index = df.index.tz_convert(totimezone)

        # Drop duplicates in case conversion caused collisions
        df = df[~df.index.duplicated(keep='first')]

    return df

def load_or_invalidate_parquet(path, max_age_hours=12):
    """
    Load a cached Parquet file if it's fresh; delete and return None if it's too old.
    
    Parameters
    ----------
    path : str or Path
        Path to the parquet file.
    max_age_hours : float, optional
        Maximum allowed file age in hours. Default is 12.
    
    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame if fresh, otherwise None.
    """
    file = Path(path)
    if not file.exists():
        return None

    # Calculate file age in hours
    age_hours = (time.time() - file.stat().st_mtime) / 3600

    if age_hours > max_age_hours:
        # File too old — remove and return None
        try:
            file.unlink()
            print(f"🗑️ Deleted expired parquet cache: {file.name} ({age_hours:.1f}h old)")
        except Exception as e:
            print(f"⚠️ Could not delete old parquet file {file}: {e}")
        return None

    # File is still fresh — load and return
    try:
        df = pd.read_parquet(file)
        print(f"✅ Loaded cached parquet: {file.name} ({age_hours:.1f}h old)")
        return df
    except Exception as e:
        print(f"⚠️ Failed to load parquet {file}: {e}")
        return None


# ------------------------
# 6. Main Function
# ------------------------
def main():
    quarterly = "15T"
    hourly = "1H"
    resample = quarterly  # Change to `hourly` for hourly model

    # Load historical data
    df = load_and_merge_data_quarterly()
    print(f"Loaded dataset with {len(df)} entries.")

    model = train_model_quarterly(df)

    # Future 48-hour forecast
    future_preds = predict_future_2days_with_actuals(model, df)
    print("\nPredictions for the next 48 hours (EUR/kWh):")
    print(future_preds.round(4))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_with_date = f"nl_price_forecast_2days_quarterly_{timestamp}.csv"

    future_preds.to_csv(filename_with_date)
    print("\nForecast saved to "+filename_with_date)

# ------------------------
# 7. Script Entry Point
# ------------------------
if __name__ == "__main__":
    main()
