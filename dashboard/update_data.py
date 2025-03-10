import datetime
import json
import logging
import time

from typing import Any

import numpy as np
import pytz
from pymemcache.client.base import Client

from analytics.grouper import Grouper
from dashboard.dash_config import TZ
from dashboard.models import Alerts, TimeRange
from utils import utils
from waze.events import Events

# TODO: Generate a specific module for logging
FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to default port in 11211, connection timeout and request timeout set to 5 secs
cache_client = Client("localhost", connect_timeout=5, timeout=5)
logger.info("Memcached connection established")

try:
    cache_client.set('test_key', 'test_value')
    test_result = cache_client.get('test_key')
    logger.info("Memcached test: %s", "SUCCESS" if test_result == b'test_value' else "FAILED")
except Exception as e:
    logger.error("Memcached connection test failed: %s", str(e))


def update_data(time_range_obj: TimeRange, alerts_obj: Alerts) -> None:
    """
    Retrieve data from the database or cache if it exists

    Args:
        time_range_obj -> TimeRange: TimeRange object a global variable
        alerts_obj -> Alerts: Data stored in memory a global variable

    Returns:
        Return -> None
    """

    perf_init = time.perf_counter()

    time_range_obj.end_time = int((datetime.datetime.now(pytz.UTC)).timestamp() * 1000)

    init_time_get = time_range_obj.init_time
    end_time_get = time_range_obj.end_time

    logger.info("Getting data from cache")
    cached = cache_client.get("alerts_data")

    if cached is not None:
        logger.info("Data found in cache")
        deserialized_cached = deserialize_dict(json.loads(cached.decode("utf-8")))
    else:
        logger.info("No data found in cache")
        deserialized_cached = []

    time_range = get_time_range(cached)

    deserialized_cached = []
    fetch = not time_range or time_range.init_time > time_range_obj.init_time

    if cached:
        deserialized_cached = deserialize_dict(json.loads(cached.decode("utf-8")))

    if time_range and time_range.init_time < time_range_obj.init_time:
        end_time_get = time_range_obj.init_time
        time_range_obj.init_time = time_range.init_time

    try:
        alerts = deserialized_cached

        if fetch:
            alerts_query = utils.load_data(
                "alerts", mode="between", between=(init_time_get, end_time_get)
            )
            alerts_query.data.extend(deserialized_cached)
            alerts = alerts_query.data
        else:
            logger.info("Using data from cache in init_time")

        alerts_query = utils.load_data(
            "alerts", mode="since", epoch=time_range.end_time if time_range else time_range_obj.init_time
        )

        alerts.extend(alerts_query.data)
        alerts = Events(alerts)

        logger.info("Preparing to cache %d alerts", len(alerts.data))
        serialized_data = json.dumps(serialize_list(alerts.data), ensure_ascii=False).encode("utf-8")
        cache_result = cache_client.set("alerts_data", serialized_data)
        logger.info("Cache set result: %s", cache_result)

        g = Grouper(alerts.to_gdf(tz=TZ))
        g.group((10, 20)).filter_by_group_time(60, True)
        alerts_obj.data = g
    except Exception as e:
        logger.error("Something was wrong while updating alerts")
        logger.error(f"Error: {e}")

    logger.info("Process time -> %.3fs", time.perf_counter() - perf_init)


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
