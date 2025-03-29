import datetime
import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
import requests
from geopandas.geodataframe import GeoDataFrame
from scipy.spatial import cKDTree
from shapely.geometry import Point

import config
from waze.alerts import Alerts

# Disable SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

TZ = "America/Santiago"

LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

ALERTS_BEGIN_TIMESTAMP = 1727740800000
LAST_UPDATE_THRESHOLD = 10000  # 10 seconds

PERIM_X = [-70.42034224747098, -70.36743722434367]
PERIM_Y = [-23.721724880116387, -23.511242421131792]

PERIM_AFTA = gpd.GeoDataFrame(geometry=gpd.points_from_xy(PERIM_X, PERIM_Y))
PERIM_AFTA.crs = "EPSG:4326"
PERIM_AFTA = PERIM_AFTA.to_crs("EPSG:3857")

logging.basicConfig(format=LOGGER_FORMAT)

logger = logging.getLogger("antof_traffic")
logger.setLevel(logging.INFO)


def get_data(
    since: int | None = None,
    until: int | None = None,
) -> Alerts:
    """
    Request alerts data from server

    Args:
        since: Millis timestamp with the begin of time to request data in UTC
        until: Millis timestamp with the end of time to request data in UTC
    """
    if config.SERVER_URL is None:
        raise requests.ConnectionError("Server URL don't defined")

    if (
        int(datetime.datetime.now(pytz.UTC).timestamp()) * 1000 - Alerts.last_update
        <= LAST_UPDATE_THRESHOLD
        and Alerts.last_update != 0
    ):
        return Alerts()

    args = f"since={since if since else ALERTS_BEGIN_TIMESTAMP}"
    args += f"&until={until}" if until else ""

    url = f"{config.SERVER_URL}/get-data?{args}"

    try:
        response = requests.get(url, timeout=10).json()
        alerts = Alerts(response.get("alerts", []))

        return alerts
    except requests.JSONDecodeError as e:
        logger.error("Error decoding JSON from request: %s", e)
    except requests.ConnectTimeout:
        logger.error("Server not respond")
    except requests.ConnectionError as e:
        logger.error("Error requesting the data, ensure that server is running: %s", e)

    raise requests.ConnectionError("Error requesting data from server")


def generate_aggregate_data(data: pd.DataFrame) -> Alerts:
    """
    Request to server generate aggregate data from a dict

    Args:
        data: Data that ask for aggregate data
    """
    if config.SERVER_URL is None:
        raise requests.ConnectionError("Server URL don't defined")

    url = f"{config.SERVER_URL}/aggregate"

    try:
        response = requests.post(url, None, data, timeout=10).json()

        return Alerts(response)
    except requests.JSONDecodeError as e:
        logger.error("Error decoding JSON from request: %s", e)
    except requests.ConnectTimeout:
        logger.error("Server not respond")
    except requests.ConnectionError as e:
        logger.error("Error requesting the data, ensure that server is running: %s", e)

    raise requests.ConnectionError("Error requesting data from server")


def update_timezone(data: gpd.GeoDataFrame, tz: str = TZ) -> gpd.GeoDataFrame:
    """
    Updates the timezone of event data
    """

    if not hasattr(data, "pub_millis"):
        return data

    data["pub_millis"] = pd.to_datetime(data["pub_millis"], unit="ms", utc=True)
    data["pub_millis"] = data["pub_millis"].dt.tz_convert(tz)
    data["end_pub_millis"] = pd.to_datetime(data["end_pub_millis"], unit="ms", utc=True)
    data["end_pub_millis"] = data["end_pub_millis"].dt.tz_convert(tz)

    return data


def convert_timestamp_tz(
    utc_timestamp: int,
    from_tz: pytz.BaseTzInfo = pytz.UTC,
    to_tz: pytz.BaseTzInfo = pytz.timezone(TZ),
) -> int:
    """
    Updates the timezone of a timestamp
    """

    date = datetime.datetime.fromtimestamp(utc_timestamp / 1000, from_tz)
    date = datetime.datetime.astimezone(date, to_tz)

    return int(date.timestamp()) * 1000


def freq_nearby(gdf: gpd.GeoDataFrame, nearby_meters=200) -> gpd.GeoDataFrame:
    """
    Counts how many nearby points are within a specified radius for each point in the GeoDataFrame.

    Args:
        - gdf: GeoDataFrame containing point geometries.
        - radius: Search radius in meters.

    Returns:
        - GeoDataFrame with an additional 'freq' column.
    """
    gdf2 = gdf.copy()
    assert gdf2.crs is not None, "GeoDataFrame empty"
    # Ensure the GeoDataFrame is in a projected CRS with units in meters
    if gdf2.crs.is_geographic:
        gdf2 = gdf2.to_crs(epsg=3857)

    if gdf2 is None:
        return gpd.GeoDataFrame()

    coords = np.vstack((gdf2.geometry.x, gdf2.geometry.y)).T

    tree = cKDTree(coords)

    indices = tree.query_ball_point(coords, r=nearby_meters)

    # Count neighbors excluding itself
    neighbor_counts = [len(ind) - 1 for ind in indices]

    # Frequency within the segment
    gdf2["freq"] = neighbor_counts

    return gdf2


def separate_coords(df: pd.DataFrame) -> GeoDataFrame:
    """
    Separate coordinates from a DataFrame into two columns, returning a GeoDataFrame
    """

    if not hasattr(df, "location"):
        logger.debug("Received empty dataframe, returning the same data")
        return gpd.GeoDataFrame(df)

    df2 = df.copy()
    df2["x"] = df2["location"].apply(lambda x: x["x"])
    df2["y"] = df2["location"].apply(lambda y: y["y"])
    df2 = df2.drop(columns="location")
    df2["geometry"] = df2.apply(lambda row: Point(row["x"], row["y"]), axis=1)  # type:ignore
    dfg = gpd.GeoDataFrame(df2, geometry="geometry")

    # Set the coorinates references system
    dfg = dfg.set_crs(epsg=4326)
    dfg = dfg.to_crs(epsg=3857)  # For map usage

    if dfg is None:
        dfg = gpd.GeoDataFrame()

    return dfg


def hourly_group(
    data: pd.DataFrame | gpd.GeoDataFrame, do_sum: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Transform an events DataFrame into an hourly report
    """

    df = data[["day_type", "hour", "pub_millis", "end_pub_millis"]].copy()

    df.reset_index(inplace=True, drop=True)

    days = (data["pub_millis"].max() - data["pub_millis"].min()).days

    days = 1 if days <= 0 else days

    # Group by day and day_type
    hourly_reports = (
        df.loc[:, ["day_type", "hour"]].groupby(["day_type", "hour"]).size().unstack(level=0)
    )

    # Create a index that include all days of the month
    all_hours = pd.Index(range(24), name="hour")

    # Include all hours of the day
    hourly_reports = hourly_reports.reindex(all_hours, fill_value=0)
    hourly_reports = hourly_reports.fillna(0)
    hourly_reports = hourly_reports.reset_index()

    if "s" not in hourly_reports.columns:
        hourly_reports["s"] = 0.0
    else:
        hourly_reports["s"] = hourly_reports["s"].astype(float)

    if "f" not in hourly_reports.columns:
        hourly_reports["f"] = 0.0
    else:
        hourly_reports["f"] = hourly_reports["f"].astype(float)

    if not do_sum:
        # Calculate ratio
        hourly_reports["f"] = hourly_reports["f"] / days
        hourly_reports["s"] = hourly_reports["s"] / days

    return hourly_reports


def daily_group(
    data: pd.DataFrame | gpd.GeoDataFrame,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Transform an events DataFrame into a daily report
    """

    df = data[["day_type", "day", "pub_millis", "end_pub_millis"]].copy()

    df.reset_index(inplace=True, drop=True)

    # Group by day and day_type
    daily_reports = (
        df.loc[:, ["day_type", "day"]].groupby(["day_type", "day"]).size().unstack(level=0)
    )

    # Create a index that include all days of the month
    all_days = pd.Index(range(1, 32), name="day")

    start_date = data["pub_millis"].min()
    end_date = data["pub_millis"].max()
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    months = 1 if months <= 0 else months
    # Fill with 0 the None values
    daily_reports = daily_reports.reindex(all_days, fill_value=0)

    if "s" not in daily_reports.columns:
        daily_reports["s"] = 0.0
    else:
        daily_reports["s"] = daily_reports["s"].astype(float)

    if "f" not in daily_reports.columns:
        daily_reports["f"] = 0.0
    else:
        daily_reports["f"] = daily_reports["f"].astype(float)

    # Calculate ratio
    daily_reports["f"] = daily_reports["f"] / months
    daily_reports["s"] = daily_reports["s"] / months
    return daily_reports
