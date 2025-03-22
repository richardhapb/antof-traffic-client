import pandas as pd
import geopandas as gpd
import numpy as np
from enum import Enum
from utils import utils


class AlertType(Enum):
    ACCIDENT = ("Accidentes",)
    JAM = ("Congestión",)
    HAZARD = ("Peligros",)
    ROAD_CLOSED = ("Cierres de vías",)
    ALL = ("Eventos",)


class Alerts:
    def __init__(self, data: list, alert_type: AlertType = AlertType.ALL) -> None:
        df = pd.DataFrame(data)

        self.data = utils.separate_coords(df)
        self.data = utils.update_timezone(self.data, utils.TZ)
        self.alert_type = alert_type

    def __add__(self, other: "Alerts") -> gpd.GeoDataFrame:
        if other.data is not None and len(other.data) > 0:
            self.data = pd.concat((self.data, other.data), axis=0)

        return self.data

    @property
    def is_empty(self) -> bool:
         return not hasattr(self.data, "pub_millis")

    def filter_by_group_time(
        self, timedelta_min: int, inplace: bool = False
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        if self.data is None or "pub_millis" not in self.data.columns:
            return pd.DataFrame()
        events2 = self.data.copy()

        if not isinstance(events2["pub_millis"].iloc[0], np.integer):
            events2["pub_millis"] = round(
                events2["pub_millis"].astype(np.int64, errors="ignore") / 1_000_000
            ).astype(np.int64)

        step = np.int64(60_000 * timedelta_min)  # step en milisegundos

        # Calculate intervals by adjusting 'pub_millis' to the nearest multiple of 'step'
        events2["interval_start"] = ((events2["pub_millis"]) // step) * step

        # Convert 'interval_start' to datetime
        events2["interval_start"] = pd.to_datetime(events2["interval_start"], unit="ms", utc=True)
        events2["interval_start"] = events2["interval_start"].dt.tz_convert(utils.TZ)

        # Ensure consistent types
        events2["group"] = events2["group"].astype(np.int16)
        events2["type"] = events2["type"].astype(str)

        # If needed, filter events that occur in the same group, type and interval
        # For example, we might want events where there are multiple occurrences
        grouped_events = events2.drop_duplicates(subset=["interval_start", "group", "type"])

        result = grouped_events.reset_index(drop=True)

        result["pub_millis"] = pd.to_datetime(result["pub_millis"], unit="ms", utc=True)
        result["pub_millis"] = result["pub_millis"].dt.tz_convert(utils.TZ)

        if inplace:
            self.data = result

        return result

    def group_by_day(self) -> pd.DataFrame:
        grouped_day = (
            pd.DataFrame({
                "group": self.data.group.value_counts().keys(),
                "qty/day": self.data.group.value_counts().values
                / (self.data["pub_millis"].max() - self.data["pub_millis"].min()).days,
            })
        ).sort_values(ascending=False, by="qty/day")

        return grouped_day

