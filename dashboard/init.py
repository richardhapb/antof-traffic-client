import datetime

from analytics.ml import init_mlflow
from dashboard.models import Model, TimeRange
from dashboard.update_data import update_data
from dashboard.update_model import load_model
from utils.utils import ALERTS_BEGIN_TIMESTAMP, convert_timestamp_tz
from waze.alerts import Alerts


def init_app(model_obj: Model, alerts_obj: Alerts) -> TimeRange:
    init_mlflow()

    since = convert_timestamp_tz(ALERTS_BEGIN_TIMESTAMP)

    end_time = convert_timestamp_tz(int(datetime.datetime.now().timestamp()) * 1000)
    time_range_obj = TimeRange(since, end_time)

    update_data(time_range_obj, alerts_obj)

    load_model(model_obj)

    return time_range_obj
