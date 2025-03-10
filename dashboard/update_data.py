import datetime
import json
import logging
import time

from typing import Any

import numpy as np
import pytz

from analytics.grouper import Grouper
from dashboard.dash_config import TZ
from dashboard.models import Alerts, TimeRange
from utils import utils
from waze.events import Events

# TODO: Generate a specific module for logging
FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


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

    logger.info("Getting data from database")

    try:
        alerts_query = utils.load_data(
            "alerts", mode="between", between=(init_time_get, end_time_get)
        )

        g = Grouper(alerts_query.to_gdf(tz=TZ))
        g.group((10, 20)).filter_by_group_time(60, True)
        alerts_obj.data = g

        logger.info("Data in memory")
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
