import os
import meteoserver
import dataset


KEY = os.environ.get("METEO_KEY")
if not KEY:
    raise ValueError("METEO_KEY environment variable not set.")
LOCATION = "Hoek van Holland"


def main():
    df = meteoserver.get(LOCATION, KEY)
    print(df)
    combined = dataset.merge_with_parquet(df, "data.parquet", new=True)

if __name__ == "__main__":
    main()
