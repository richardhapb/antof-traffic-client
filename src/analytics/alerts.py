import gc
from datetime import datetime
from enum import Enum
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz

from utils import utils

LAST_UPDATE_THRESHOLD = 4000  # 4 seconds


class AlertType(Enum):
    """Types of accidents according to the Waze API"""

    ACCIDENT = ("Accidentes",)
    JAM = ("Congestión",)
    HAZARD = ("Peligros",)
    ROAD_CLOSED = ("Cierres de vías",)
    ALL = ("Eventos",)


class Alerts:
    """
    Main class where all the Alerts data is stored. This is a Singleton class that ensures
    ensures that the data is stored once and avoids duplicates, as the dataset is large
    requires a large amount of memory

    On the first call, the Alerts are initialized if data is passed as an argument in the constructor; otherwise,
    is initialized without a `data` attribute. If data is passed, it is initialized and returned with
    `data` attribute contains a `GeoDataFrame` with geodata

    If the `LAST_UPDATE_THRESHOLD` is exceeded, the data is initialized again, allowing for data updates when
    new data is available from the server
    """

    _instance: "Alerts | None" = None
    last_update: int = 0

    def __new__(cls, *args, **kwargs):
        current_time = int(datetime.now(pytz.UTC).timestamp()) * 1000

        if cls._instance is None or current_time - cls.last_update > LAST_UPDATE_THRESHOLD:
            # Ensure memory is freed as soon as possible
            cls.reset_instance()

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

            self.data: gpd.GeoDataFrame = utils.separate_coords(df)
            self.data.drop(columns=["end_pub_millis"], inplace=True)
            self.data = cast("gpd.GeoDataFrame", utils.update_timezone(self.data, utils.TZ))

    def __add__(self, other: "Alerts") -> gpd.GeoDataFrame:
        """
        Concatenate data with another alerts and drop duplicate returns. Modifies
        the `data` attribute of `Alerts` in place to avoid duplicate data and double
        memory usage
        """
        if not other.is_empty:
            self.data = gpd.GeoDataFrame(pd.concat((self.data, other.data), axis=0, ignore_index=True))
            self.data.drop_duplicates(("uuid"), inplace=True)

        return cast("gpd.GeoDataFrame", self.data)

    @classmethod
    def reset_instance(cls) -> None:
        """Clear the data and instance to prepare for another build"""
        if hasattr(cls, "_instance") and cls._instance:
            cls._instance = None
            gc.collect()

    @classmethod
    def new_instance(cls, data: list | None = None) -> "Alerts":
        """Create a new instance while ignoring the singleton instance"""
        if data is None:
            data = []
        instance = super().__new__(cls)
        instance.__init__(data)  # noqa: PLC2801
        return instance

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

        data = self.data.copy() if not inplace else self.data

        if not isinstance(data["pub_millis"].iloc[0], np.integer):
            data["pub_millis"] = round(data["pub_millis"].astype(np.int64, errors="ignore") / 1_000_000).astype(
                np.int64
            )

        step = np.int64(60_000 * timedelta_min)  # step en milisegundos

        # Adjust `pub_millis` to the nearest step
        data["interval_start"] = ((data["pub_millis"]).to_numpy() // step) * step

        # Convert 'interval_start' to datetime
        data["interval_start"] = pd.to_datetime(data["interval_start"], unit="ms", utc=True)
        data["interval_start"] = data["interval_start"].dt.tz_convert(utils.TZ)

        # Ensure consistent types
        data["group"] = data["group"].astype(np.int16)
        data["type"] = data["type"].astype(str)

        # Keep only the unique events in the three-dimensional variables
        data.drop_duplicates(subset=["interval_start", "group", "type"], inplace=True)
        data.drop(columns=["interval_start"], inplace=True)

        # Ensure pub_millis is properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(data["pub_millis"]):
            data["pub_millis"] = pd.to_datetime(data["pub_millis"], unit="ms", utc=True)
            data["pub_millis"] = data["pub_millis"].dt.tz_convert(utils.TZ)

        data.reset_index(drop=True, inplace=True)

        return data

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

        return (
            pd.DataFrame({
                "group": self.data.group.value_counts().keys(),
                "qty/day": self.data.group.value_counts().values
                / (self.data["pub_millis"].max() - self.data["pub_millis"].min()).days,
            })
        ).sort_values(ascending=False, by="qty/day")
