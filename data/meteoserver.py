import requests
import time

import pandas as pd
from dataset import normalize

def get(locatie, key, retries=3, delay=10, wanted_cols=['temperature', 'precipitation', 'irradiance', 'wind_speed']):
    """
    Fetch hourly forecast data from weerlive.nl API.
    
    Args:
        locatie (str): Location name or code.
        key (str): Your API key.
        retries (int): Number of retry attempts.
        delay (int): Delay (seconds) between retries.
        wanted_cols (list): List of columns to return. If None, return all columns.

    Returns:
        dict | None: Decoded JSON data from the 'data' attribute, or None if all retries fail.
    """
    url = "https://data.meteoserver.nl/api/uurverwachting.php?"
    params = {"locatie": locatie, "key": key}
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx, 5xx)

            # Parse JSON
            json_data = response.json()

            # Return the "data" field if it exists
            if "data" in json_data:
                data = _as_pandas(json_data)
                data = data.rename(columns={
                    "temp": "temperature", # in °C
                    "neersl": "precipitation",  # in mm
                    "gr": "irradiance", # in J/cm2
                    "winds": "wind_speed" # in m/s
                })
                return data[wanted_cols] if wanted_cols else data
            else:
                print("⚠️ No 'uur_verw' field found in response.")
                return None

        except requests.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ All retry attempts failed.")
                return None


def _as_pandas(dataDict):
    """
    Convert the JSON dictionary from API to a Pandas DataFrame.
    
    Args:
        dataDict (dict): JSON response from API, expected to have 'uur_verw' key.
        numeric (bool): If True, convert all columns (except timestamp) to numeric.

    Returns:
        pd.DataFrame: Time-indexed DataFrame, optionally numeric, normalized to Europe/Amsterdam.
    """
    # Convert JSON list of dicts to DataFrame
    df = pd.DataFrame(dataDict.get('data', []))

    if df.empty:
        return df  # return empty DataFrame if no data

    # Convert timestamp column to datetime
    if 'tijd' not in df.columns:
        raise KeyError("API data missing 'tijd' column")
    df['tijd'] = pd.to_datetime(df['tijd'], unit='s')
    df.set_index('tijd', inplace=True)

    # Convert all columns except 'timestamp' to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return normalize(df)