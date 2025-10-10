import numpy as np

FEATURES_FORECAST_AND_METRICS_AND_DD = ['temperature', 'precipitation', 'wind_speed', 'irradiance', 'day_of_week', 'hour', 'quarter', 'month', 'season', 'solar_performance', 'wind_performance', 'dd_heat', 'dd_cool']

def solar_performance_metric(rad, ambient, gamma=-0.0035, NOCT=45.0):
    """
    Compute an effective solar performance metric that accounts for irradiance,
    module temperature, and efficiency losses.

    This metric represents an adjusted irradiance value (in W/m²-equivalent)
    that approximates the combined effects of irradiance level, temperature,
    and cell performance characteristics. It is useful for estimating
    photovoltaic (PV) performance under non-standard test conditions.

    Parameters
    ----------
    rad : float or array_like
        Solar irradiance in joules per square centimeter per hour (J/cm²/h).
        Typical range: 0–3600 J/cm²/h (equivalent to 0–1000 W/m²).
    ambient : float or array_like
        Ambient temperature in degrees Celsius (°C).
    gamma : float, optional
        Temperature coefficient of power (per °C). Defaults to -0.0035.
        Negative values indicate power decreases with increasing temperature.
    NOCT : float, optional
        Nominal Operating Cell Temperature in degrees Celsius (°C).
        Defaults to 45.0. Used in estimating cell temperature rise under
        irradiance.

    Returns
    -------
    metric : float or ndarray
        Effective solar performance metric (0–1000 range), representing
        adjusted irradiance in W/m²-equivalent after accounting for
        temperature and irradiance nonlinearity.

    References
    ----------
    - IEC 61853-1: Photovoltaic (PV) module performance testing and energy rating
    - Duffie, J.A., & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*.

    """
    conv = 10000.0 / 3600.0  # J/cm²/h -> W/m²
    R = rad * conv      # W/m²

    T = ambient + (NOCT - 20.0) / 800.0 * R
    a = 1.0 - 0.2 * np.exp(-R / 200.0)
    irradiance_factor = (R / 1000.0) ** a

    temp_factor = 1.0 + gamma * (T - 25.0)
    temp_factor = np.maximum(temp_factor, 0.0)  # element-wise

    metric = 1000.0 * irradiance_factor * temp_factor
    metric = np.clip(metric, 0.0, 1000.0)       # element-wise clip

    return metric

def wind_performance_metric(wind_speeds, cut_in=3.0, rated=12.0, cut_out=25.0, derate_at_cutout=0.90):
    """
    Compute a wind turbine performance metric scaled from 0 to 1000,
    with a cubic ramp-up to rated speed and a slight derating between
    rated and cut-out.

    Parameters
    ----------
    wind_speeds : array-like
        List or array of wind speeds in m/s.
    cut_in : float, default=3.0
        Cut-in wind speed (below this, no power).
    rated : float, default=12.0
        Rated wind speed (power plateaus here).
    cut_out : float, default=25.0
        Cut-out wind speed (above this, turbine shuts down).
    derate_at_cutout : float, default=0.90
        Fraction of rated power at cut-out (e.g., 0.90 = 90%).

    Returns
    -------
    numpy.ndarray
        Integer array of performance metrics in range [0, 1000].
    """
    ws = np.array(wind_speeds, dtype=float)
    metric = np.zeros_like(ws)

    # Between cut-in and rated → cubic growth
    mask_ramp = (ws >= cut_in) & (ws < rated)
    denom = (rated**3 - cut_in**3) if rated > cut_in else 1.0
    metric[mask_ramp] = (ws[mask_ramp]**3 - cut_in**3) / denom

    # Between rated and cut-out → linear derating
    mask_derate = (ws >= rated) & (ws <= cut_out)
    if cut_out > rated:
        slope = (1.0 - derate_at_cutout) / (cut_out - rated)
        metric[mask_derate] = 1.0 - slope * (ws[mask_derate] - rated)
    else:
        metric[mask_derate] = 1.0

    # Normalize and scale to 0–1000 integer metric
    metric = np.clip(metric, 0.0, 1.0)
    metric_0_1000 = (metric * 1000).round().astype(int)

    return metric_0_1000

def degree_days_cool(temp, cutover=24):
    """
    Calculate cooling degree days (CDD) for a series of temperatures.

    Cooling degree days quantify the demand for cooling (e.g., air conditioning)
    based on how much and how long the daily temperature exceeds a given threshold.

    Parameters
    ----------
    temp : pandas.Series
        Series of daily mean temperatures (°C).
    cutover : float, optional
        Threshold temperature (°C) above which cooling is required.
        Defaults to 24°C.

    Returns
    -------
    pandas.Series
        Cooling degree days for each temperature value. Values are zero if
        the temperature is below or equal to the cutover.
    """
    return temp.apply(lambda T: cutover - T if T > cutover else 0)

def degree_days_heat(temp, cutover=18):
    """
    Calculate heating degree days (HDD) for a series of temperatures.

    Heating degree days quantify the demand for heating (e.g., space heating)
    based on how much and how long the daily temperature falls below a given threshold.

    Parameters
    ----------
    temp : pandas.Series
        Series of daily mean temperatures (°C).
    cutover : float, optional
        Threshold temperature (°C) below which heating is required.
        Defaults to 18°C.

    Returns
    -------
    pandas.Series
        Heating degree days for each temperature value. Values are zero if
        the temperature is above or equal to the cutover.
    """
    return temp.apply(lambda T: cutover - T if T < cutover else 0)

def create(df):
    """
    Perform feature engineering on a time-indexed DataFrame for energy modeling.

    This function generates time-based and weather-related features from
    the input DataFrame, which is assumed to have a DatetimeIndex and
    columns for 'temperature', 'irradiance', and 'wind_speed'. It also
    computes derived performance metrics for solar and wind energy, and
    heating/cooling degree days.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the following columns:
        - 'temperature' : float, ambient temperature in °C
        - 'irradiance' : float, solar irradiance in J/cm²/h
        - 'wind_speed' : float, wind speed in m/s
        The DataFrame index must be a pandas.DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the following additional columns:
        - 'day_of_week' : int, day of the week (0=Monday, 6=Sunday)
        - 'hour' : int, hour of the day (0–23)
        - 'quarter' : int, quarter of the year (1–4)
        - 'month' : int, month of the year (1–12)
        - 'season' : int, mapped season (0=winter, 1=spring, 2=summer, 3=autumn)
        - 'solar_performance' : float, estimated solar performance metric
        - 'wind_performance' : float, estimated wind performance metric
        - 'dd_heat' : float, heating degree days
        - 'dd_cool' : float, cooling degree days
    """
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['season'] = df.index.month.map({
        12: 0, 1: 0, 2: 0,    # winter
        3: 1, 4: 1, 5: 1,     # spring
        6: 2, 7: 2, 8: 2,     # summer
        9: 3, 10: 3, 11: 3    # autumn
    })
    df['solar_performance'] = solar_performance_metric(df['irradiance'], df['temperature']).astype(float)
    df['wind_performance'] = wind_performance_metric(df['wind_speed']).astype(float)
    df['dd_heat'] = degree_days_heat(df['temperature'])
    df['dd_cool'] = degree_days_cool(df['temperature'])
    return df


 