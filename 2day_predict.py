import warnings
import os
from pathlib import Path
import tzlocal
import time
from datetime import datetime
import pandas as pd

import meteoserver.weatherforecast as meteo
import features
import xgboost as xgb

warnings.filterwarnings("ignore")

METEOSERVER_API_KEY = os.getenv("METEOSERVER_API_KEY")
if not METEOSERVER_API_KEY:
    raise ValueError("Please set the METEOSERVER_API_KEY environment variable.")

LOCATION = "Hoek van Holland"
MODEL_FILE = "daysahead_xgb_model.json"
FORECAST_CACHE_FILE = "nl_meteoserver_forecast.parquet"

def fetch_meteoserver_forecast(location=LOCATION, hours=48, cache_file=FORECAST_CACHE_FILE):
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
    forecast_df.index.rename('datetime', inplace=True)
    forecast_df = forecast_df[forecast_df.index <= pd.Timestamp.now() + pd.Timedelta(hours=hours)]

    forecast_df = forecast_df.rename(columns={
        'temp': 'temperature', # °C
        'neersl': 'precipitation', # mm
        'winds': 'wind_speed', # m/sec
        'gr': 'irradiance' # J/cm2
    })[['temperature', 'precipitation', 'wind_speed', 'irradiance']]

    forecast_df['temperature'] = pd.to_numeric(forecast_df['temperature'])
    forecast_df['precipitation'] = pd.to_numeric(forecast_df['precipitation'])
    forecast_df['irradiance'] = pd.to_numeric(forecast_df['irradiance'])
    forecast_df['wind_speed'] = pd.to_numeric(forecast_df['wind_speed'])

    forecast_df = forecast_df[['temperature', 'precipitation', 'wind_speed', 'irradiance']]

    # Save to cache
    forecast_df.to_parquet(cache_file)
    forecast_df = normalize_to_utc(forecast_df)
    print(f"Forecast saved to cache: {cache_file}")

    return forecast_df

# ------------------------
def predict_future_2days(model, hours=48, timezone=tzlocal.get_localzone()):
    print(f"Predicting next {hours} hours with actual day-ahead prices...")

    future  = fetch_meteoserver_forecast(hours=hours)
    X_new = features.create(future)

    # Ensure same feature order
    features_ordered = model.get_booster().feature_names
    X_new = X_new[features_ordered]


    # Now predict
    #return model.predict(X_new)  
    #return predictions

    predicted_prices = []
    for idx, row in X_new.iterrows():
        X = pd.DataFrame([row[features_ordered]])
        pred = model.predict(X)[0]

        predicted_prices.append(pred)

    future['predicted_price_eur_kwh'] = predicted_prices
    future.index = future.index.tz_convert(timezone)

    return future[['predicted_price_eur_kwh']]


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


def main():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    print(f"Loaded model from {MODEL_FILE}")

    future = predict_future_2days(model)
    print("\nPredictions for the next 48 hours (EUR/kWh):")
    print(future.round(4))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_with_date = f"price_forecast_2days_hourly_{timestamp}.csv"

    future.to_csv(filename_with_date)
    print("\nForecast saved to "+filename_with_date)

if __name__ == "__main__":
    main()
