import os
import pandas as pd
import tzlocal

def normalize(df, totimezone="UTC"):
    """
    Converts all datetime columns in a pandas DataFrame to UTC timezone.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        totimezone (str): Target timezone, default is "UTC".
    
    Returns:
        pd.DataFrame: DataFrame with all datetime columns converted to UTC
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
        
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

def merge_with_parquet(df, filename, new=False, keep_parquet_on_conflict=True):
    """
    Merge a DataFrame with an existing parquet dataset, keeping parquet values
    when time indices overlap.

    Args:
        df (pd.DataFrame): The new dataset to merge.
        filename (str): Path to the parquet file.
        new (bool): If True, overwrite the parquet file with df. Defaults to False.

    Returns:
        pd.DataFrame: Merged dataset (time-indexed, parquet data kept on conflicts).
    """

    new = not os.path.exists(filename)
    if new:
        # create an empty dataframe
        existing_df = pd.DataFrame()
    else:
        # Load the parquet file
        existing_df = pd.read_parquet(filename)

    # Ensure both have datetime indices
    if not isinstance(existing_df.index, pd.DatetimeIndex):
        existing_df.index = pd.to_datetime(existing_df.index)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)


    # Combine and resolve duplicates (keep parquet data)
    if keep_parquet_on_conflict:
        combined = pd.concat([df, existing_df])
    else:
        combined = pd.concat([existing_df, df])

    combined.index = pd.to_datetime(combined.index)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    # Save back to parquet
    combined.to_parquet(filename, compression='zstd')

    return combined
