from datetime import datetime
from enum import Enum
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz

from utils import utils

LAST_UPDATE_THRESHOLD = 10000  # 10 seconds


class AlertType(Enum):
    ACCIDENT = ("Accidentes",)
    JAM = ("Congestión",)
    HAZARD = ("Peligros",)
    ROAD_CLOSED = ("Cierres de vías",)
    ALL = ("Eventos",)


class Alerts:
    _instance = None
    last_update = 0

    def __new__(cls, *args, **kwargs):
        current_time = int(datetime.now(pytz.UTC).timestamp()) * 1000

        if cls._instance is None or current_time - cls.last_update > LAST_UPDATE_THRESHOLD:
            instance = super().__new__(cls)
            cls._instance = instance
            cls.last_update = current_time

        return cls._instance

    def __init__(self, data: list | None = None, alert_type: AlertType = AlertType.ALL) -> None:
        if not data:
            data = []

        if not hasattr(self, "data") and data:
            self.alert_type = alert_type
            df = pd.DataFrame(data)

            self.data = utils.separate_coords(df)
            self.data = utils.update_timezone(self.data, utils.TZ)

    def __add__(self, other: "Alerts") -> gpd.GeoDataFrame:
        if not other.is_empty:
            self.data = pd.concat((self.data, other.data), axis=0, ignore_index=True)
            self.data.drop_duplicates(("uuid"), inplace=True)

        return cast("gpd.GeoDataFrame", self.data)

    @property
    def is_empty(self) -> bool:
        return not hasattr(self, "data") or self.data.shape[0] == 0

    def filter_by_group_time(self, timedelta_min: int, inplace: bool = False) -> gpd.GeoDataFrame | pd.DataFrame:
        """
        Filter and group data by time intervals.

        Groups events into time intervals of specified duration and ensures
        consistent data types across columns.

        Args:
            timedelta_min: Size of time interval in minutes
            inplace: If True, modify data in place. If False, return a copy

        Returns:
            GeoDataFrame or DataFrame with events grouped by time intervals.
            Empty DataFrame if input is None or missing required columns.

        Notes:
            - Converts pub_millis to milliseconds if not already in that format
            - Creates interval_start column marking the start of each time window
            - Converts timestamps to specified timezone
            - Ensures consistent data types for 'group' and 'type' columns
            - Removes duplicate events within same interval/group/type
        """

        if self.data is None or "pub_millis" not in self.data.columns:
            return pd.DataFrame()
        if not inplace:
            self.data = self.data.copy()

        if not isinstance(self.data["pub_millis"].iloc[0], np.integer):
            self.data["pub_millis"] = round(
                self.data["pub_millis"].astype(np.int64, errors="ignore") / 1_000_000
            ).astype(np.int64)

        step = np.int64(60_000 * timedelta_min)  # step en milisegundos

        # Adjust `pub_millis` to the nearest step
        self.data["interval_start"] = ((self.data["pub_millis"]).to_numpy() // step) * step

        # Convert 'interval_start' to datetime
        self.data["interval_start"] = pd.to_datetime(self.data["interval_start"], unit="ms", utc=True)
        self.data["interval_start"] = self.data["interval_start"].dt.tz_convert(utils.TZ)

        # Ensure consistent types
        self.data["group"] = self.data["group"].astype(np.int16)
        self.data["type"] = self.data["type"].astype(str)

        # Keep only the unique events in the three-dimensional variables
        grouped_events = self.data.drop_duplicates(subset=["interval_start", "group", "type"])

        result = grouped_events.reset_index(drop=True)

        result["pub_millis"] = pd.to_datetime(result["pub_millis"], unit="ms", utc=True)
        result["pub_millis"] = result["pub_millis"].dt.tz_convert(utils.TZ)

        return result

    def group_by_day(self) -> pd.DataFrame:
        """
        Calculate average daily event counts by group (segment).

        Args:
            data: GeoDataFrame containing event data with 'group' and 'pub_millis' columns

        Returns:
            DataFrame with columns:
                - group: Group identifier
                - qty/day: Average number of events per day for each group,
                           sorted in descending order
        """

        grouped_day = (
            pd.DataFrame({
                "group": self.data.group.value_counts().keys(),
                "qty/day": self.data.group.value_counts().values
                / (self.data["pub_millis"].max() - self.data["pub_millis"].min()).days,
            })
        ).sort_values(ascending=False, by="qty/day")

        return grouped_day
