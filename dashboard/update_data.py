import time

import requests

from dashboard.models import TimeRange
from utils import utils
from utils.utils import logger
from waze.alerts import Alerts


def update_data(time_range_obj: TimeRange, alerts_obj: Alerts) -> None:
    """
    Retrieve data from the server between the time range

    Args:
        time_range_obj -> TimeRange: TimeRange object a global variable
        alerts_obj -> Alerts: Data stored in memory a global variable

    Returns:
        Return -> None
    """

    perf_init = time.perf_counter()

    since_request: int = time_range_obj.init_time
    until_request: int | None = time_range_obj.end_time

    if alerts_obj.data.shape[0] > 0:
        min_timestamp = utils.convert_timestamp_tz(alerts_obj.data["pub_millis"].min())
        if min_timestamp >= time_range_obj.init_time:
            since_request = time_range_obj.end_time
            until_request = None

    logger.info("Getting data from server")

    try:
        alerts_response = utils.get_data(since_request, until_request)

        alerts_obj.data = alerts_obj + alerts_response

        logger.info("Data retrieved correctly")
    except requests.ConnectionError as e:
        logger.error("Error retrieving data from server: %s", e)
    except Exception as e:
        logger.error("Something was wrong while updating alerts")
        logger.error("Error: %s", e)

    logger.info("Process time -> %.3fs", time.perf_counter() - perf_init)


