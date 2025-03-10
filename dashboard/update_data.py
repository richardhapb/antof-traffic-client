import datetime
import json
from typing import Any

import numpy as np
import pytz
from pymemcache.client.base import Client

from analytics.grouper import Grouper
from dashboard.dash_config import TZ
from dashboard.models import Alerts, TimeRange
from utils import utils
from waze.events import Events

# Connect to default port in 11211, connection timeout and request timeout to setted to 5 secs
cache_client = Client("localhost", connect_timeout=5, timeout=5)


def update_data(time_range_obj: TimeRange, alerts_obj: Alerts) -> None:
    """
    Retrieve data from the database or cache if it exists

    Args:
        time_range_obj -> TimeRange: TimeRange object a global variable
        alerts_obj -> Alerts: Data stored in memory a global variable

    Returns:
        Return -> None
    """

    time_range_obj.end_time = int((datetime.datetime.now(pytz.UTC)).timestamp() * 1000)

    init_time_get = time_range_obj.init_time
    end_time_get = time_range_obj.end_time

    cached = cache_client.get("alerts_data", {})
    time_range = get_time_range(cached)

    deserialized_cached = []
    fetch = not time_range or time_range.init_time < time_range_obj.init_time

    if cached:
        deserialized_cached = deserialize_dict(json.loads(cached.decode("utf-8")))

    if time_range and time_range.init_time < time_range_obj.init_time:
        end_time_get = time_range_obj.init_time
        time_range_obj.init_time = time_range.init_time

    try:
        alerts = []

        if fetch:
            alerts_query = utils.load_data(
                "alerts", mode="between", between=(init_time_get, end_time_get)
            )
            alerts_query.data.extend(deserialized_cached)
            alerts = Events(alerts_query.data)

            serialized_data = json.dumps(serialize_list(alerts.data), ensure_ascii=False).encode(
                "utf-8"
            )
            cache_client.set("alerts_data", serialized_data)
        else:
            alerts = Events(deserialized_cached)

        g = Grouper(alerts.to_gdf(tz=TZ))
        g.group((10, 20)).filter_by_group_time(60, True)
        alerts_obj.data = g
    except Exception as e:
        print("Something was wrong while updating alerts")
        print(f"Error: {e}")


def get_time_range(data: dict) -> TimeRange | None:
    if not data or not hasattr(data, "pubMillis"):
        return None

    init_time = np.min(data["pubMillis"])
    end_time = np.max(data["pubMillis"])

    return TimeRange(init_time, end_time)


def deserialize_dict(data: dict) -> list[dict[str, Any]]:
    if not data:
        return []
    keys = data.keys()
    return [dict(zip(keys, values)) for values in zip(*data.values())]


def serialize_list(data: list[dict[str, Any]]) -> dict[str, list]:
    if not data:
        return {}
    return {key: [item[key] for item in data] for key in data[0]}
