import datetime
import json
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
import threading

import config
from analytics.alerts import Alerts

# Disable SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Using to ensure thread safety
# when retrieving data
_data_lock = threading.Lock()

TZ = "America/Santiago"

LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

ALERTS_BEGIN_TIMESTAMP = 1727740800000  # 2024/10/01
# Time to retrieve last singleton instance between graphics

MINUTES_BETWEEN_UPDATES_FROM_API = 2

logging.basicConfig(format=LOGGER_FORMAT)

logger = logging.getLogger("antof_traffic")
logger.setLevel(logging.INFO)


class SerializeError(Exception):
    """Error when serialization is not possible"""

    def __init__(self, data: pd.DataFrame | str):
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            super().__init__(f"Cannot serialize the data, getted columns {data.columns}")
        super().__init__(data)


def get_data(
    since: int | None = None,
    until: int | None = None,
) -> Alerts:
    """
    Request alerts data from server.

    Args:
        since: Millis timestamp with the begin of time to request data in UTC
        until: Millis timestamp with the end of time to request data in UTC

    """
    if config.SERVER_URL is None:
        msg = "Server URL doesn't defined"
        raise requests.ConnectionError(msg)

    # Ensure thread safety
    with _data_lock:
        # If an instance exists, return it; if a new instance is created, that
        # should be empty
        alerts = Alerts()
        if not alerts.is_empty:
            return alerts

        args = f"since={since if since else ALERTS_BEGIN_TIMESTAMP}"
        args += f"&until={until}" if until else ""

        url = f"{config.SERVER_URL}/get-data?{args}"

        try:
            response = requests.get(url, timeout=10).json()
            alerts = Alerts(response.get("alerts", []))

        except requests.JSONDecodeError:
            logger.exception("Error decoding JSON from requests")
        except requests.ConnectTimeout:
            logger.exception("Server not respond")
        except requests.ConnectionError:
            logger.exception("Error requesting the data, ensure that server is running")

        return alerts


def generate_aggregate_data(data: pd.DataFrame) -> Alerts:
    """
    Request to server generate aggregate data from a dict

    Args:
        data: Data that ask for aggregate data

    Returns:
        Alerts object containing aggregated data

    Raises:
        requests.ConnectionError: If server URL is not defined or connection fails
        SerializeError: If data cannot be serialized

    """
    if config.SERVER_URL is None:
        msg = "Server URL doesn't defined"
        raise requests.ConnectionError(msg)

    url = f"{config.SERVER_URL}/aggregate"

    # Batch size for requests (adjust as needed)
    batch_size = 1000
    total_rows = len(data)

    try:
        all_alerts = []

        # Process data in batches
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = data.iloc[start_idx:end_idx]

            logger.debug("Processing batch %d-%d of %d", start_idx, end_idx, total_rows)
            response = requests.post(url, json=serialize_data(batch), timeout=10)
            if response.status_code == 422:
                print("422 detail:", response.text)
            response.raise_for_status()
            response_data = response.json()

            all_alerts.extend(response_data.get("alerts", []))

        return Alerts.new_instance(all_alerts)

    except requests.JSONDecodeError:
        logger.exception("Error decoding JSON from request")
    except requests.ConnectTimeout:
        logger.exception("Server not respond")
    except requests.ConnectionError:
        logger.exception("Error requesting the data, ensure that server is running")
    except SerializeError:
        logger.exception("Error serializing data")

    msg = "Error requesting data from server"
    raise requests.ConnectionError(msg)


def serialize_data(data: pd.DataFrame) -> dict:
    """Convert the GeoDataFrame instance to a JSON string"""

    if not hasattr(data, "pub_millis"):
        raise SerializeError(data)

    has_timezone = data.pub_millis.dt.tz is not None

    if has_timezone:
        data.pub_millis = data.pub_millis.dt.tz_convert(pytz.UTC)

    data.pub_millis = (data.pub_millis.astype(np.int64) // 1_000_000).astype(np.int64)

    try:
        serialized = {"alerts": data.to_dict(orient="records")}
    except json.JSONDecodeError as e:
        raise SerializeError(e.msg) from None

    return serialized


def update_timezone(data: pd.DataFrame | gpd.GeoDataFrame, tz: str = TZ) -> pd.DataFrame | gpd.GeoDataFrame:
    """Update the timezone of event data."""

    if not hasattr(data, "pub_millis"):
        return data

    data["pub_millis"] = pd.to_datetime(data["pub_millis"], unit="ms", utc=True)
    data["pub_millis"] = data["pub_millis"].dt.tz_convert(tz)

    return data


def convert_timestamp_tz(
    utc_timestamp: int,
    from_tz: pytz.BaseTzInfo = pytz.UTC,
    to_tz: pytz.BaseTzInfo | None = None,
) -> int:
    """Update the timezone of a timestamp"""

    if not to_tz:
        to_tz = pytz.timezone(TZ)

    date = datetime.datetime.fromtimestamp(utc_timestamp / 1000, from_tz)
    date = datetime.datetime.astimezone(date, to_tz)

    return int(date.timestamp()) * 1000


def freq_nearby(gdf: gpd.GeoDataFrame, nearby_meters: int = 200) -> gpd.GeoDataFrame:
    """
    Count how many nearby points are within a specified radius for each point in the GeoDataFrame.
    Modify data inplace for avoid duplicate data and double memory usage.

    Args:
        gdf: GeoDataFrame containing point geometries.
        nearby_meters: Meters to the `GeoDataFrame` point in meters

    Returns:
        GeoDataFrame modified with an additional 'freq' column.

    """
    if gdf.crs is None:
        logger.warning("GeoDataFrame `crs` empty")
        return gdf

    # Ensure the GeoDataFrame is in a projected CRS with units in meters
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=32719)

    coords = np.vstack((gdf.geometry.x, gdf.geometry.y)).T

    tree = cKDTree(coords)

    indices = tree.query_ball_point(coords, r=nearby_meters)

    # Count neighbors excluding itself
    neighbor_counts = [len(ind) - 1 for ind in indices]

    # Frequency within the segment
    gdf["freq"] = neighbor_counts

    return gdf


def separate_coords(df: pd.DataFrame) -> GeoDataFrame:
    """Separate coordinates from a DataFrame into two columns, returning a GeoDataFrame"""

    if not hasattr(df, "location"):
        logger.debug("Received empty dataframe, returning the same data")
        return gpd.GeoDataFrame(df)

    df["x"] = df["location"].apply(lambda x: x["x"])
    df["y"] = df["location"].apply(lambda y: y["y"])
    df = df.drop(columns="location")
    df["geometry"] = df.apply(lambda row: Point(row["x"], row["y"]), axis=1)  # type:ignore
    dfg = gpd.GeoDataFrame(df, geometry="geometry")

    # Set the coorinates references system
    dfg = dfg.set_crs(epsg=4326)
    dfg = dfg.to_crs(epsg=32719)  # For map usage

    if dfg is None:
        dfg = gpd.GeoDataFrame()

    return dfg


def join_coords(df: pd.DataFrame) -> pd.Series:
    """Join coordinates from x and y columns"""
    if not hasattr(df, "x") or not hasattr(df, "y"):
        logger.debug("Received empty dataframe, returning the same data")
        return df["location"] if "location" in df else pd.Series(dtype=object)

    df["location"] = df.apply(lambda row: {"x": row["x"], "y": row["y"]}, axis=1)
    return df.drop(columns=["x", "y"])["location"]


def hourly_group(data: pd.DataFrame | gpd.GeoDataFrame, do_sum: bool = False) -> pd.DataFrame | gpd.GeoDataFrame:
    """Transform an events DataFrame into an hourly report."""

    df = data[["day_type", "hour", "pub_millis"]]

    df.reset_index(inplace=True, drop=True)

    days = (data["pub_millis"].max() - data["pub_millis"].min()).days

    days = 1 if days <= 0 else days

    # Group by day and day_type
    hourly_reports = df.loc[:, ["day_type", "hour"]].groupby(["day_type", "hour"]).size().unstack(level=0)

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
        hourly_reports["f"] /= days
        hourly_reports["s"] /= days

    return hourly_reports


def daily_group(
    data: pd.DataFrame | gpd.GeoDataFrame,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Transform an events DataFrame into a daily report."""

    df = data[["day_type", "day", "pub_millis"]]

    df.reset_index(inplace=True, drop=True)

    # Group by day and day_type
    daily_reports = df.loc[:, ["day_type", "day"]].groupby(["day_type", "day"]).size().unstack(level=0)

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
    daily_reports["f"] /= months
    daily_reports["s"] /= months
    return daily_reports
