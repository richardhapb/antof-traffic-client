import time
import datetime

import pytz
import requests

from dashboard.models import TimeRange
from utils import utils
from utils.utils import logger, TZ
from waze.alerts import Alerts
import config


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

    # Update to last time
    time_range_obj.end_time = int(datetime.datetime.now(pytz.timezone(TZ)).timestamp()) * 1000

    # Convert to UTC for match with the database
    since_request: int = utils.convert_timestamp_tz(
        time_range_obj.init_time, pytz.timezone(utils.TZ), pytz.UTC
    )
    until_request: int | None = utils.convert_timestamp_tz(
        time_range_obj.end_time, pytz.timezone(utils.TZ), pytz.UTC
    )

    if alerts_obj.data.shape[0] > 0:
        min_timestamp = utils.convert_timestamp_tz(
            int(alerts_obj.data["pub_millis"].min().timestamp() * 1000),
            pytz.timezone(utils.TZ),
            pytz.UTC,
        )
        if min_timestamp >= time_range_obj.init_time:
            since_request = until_request or since_request
            until_request = None

    logger.info("Getting data from server")

    try:
        alerts_response = utils.get_data(since_request, until_request)

        alerts_obj.data = alerts_obj + alerts_response

        logger.info("Data retrieved correctly, %i elements", alerts_response.data.shape[0])
    except requests.ConnectionError as e:
        logger.error("Error retrieving data from server: %s", e)
    except Exception as e:
        logger.error("Something was wrong while updating alerts")
        logger.error("Error: %s", e)

    logger.info("Process time -> %.3fs", time.perf_counter() - perf_init)


def update_data_from_api() -> None:
    """
    Send a trigger to server for retrieve data from API in async way. Then
    de data is retrieved using `update_data` function, that is fastes, because
    server use cache for send it.
    """
    if not config.SERVER_URL:
        logger.error("Server is not available for update data from API")

    perf_init = time.perf_counter()

    url = f"{config.SERVER_URL}/update-data"

    try:
        requests.get(url, timeout=10).json()

        logger.info("Data retrieved from API sucessfully")
        logger.info("Process time -> %.3fs", time.perf_counter() - perf_init)

    except requests.JSONDecodeError as e:
        logger.error("Error decoding JSON from request: %s", e)
    except requests.ConnectTimeout:
        logger.error("Server not respond")
    except requests.ConnectionError as e:
        logger.error("Error requesting the data, ensure that server is running: %s", e)
