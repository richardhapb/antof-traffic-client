import datetime
import pytz

from analytics.ml import init_mlflow
from dashboard.models import Model, TimeRange
from dashboard.update_model import load_model
from utils.utils import convert_timestamp_tz


def init_app(model_obj: Model) -> TimeRange:
    init_mlflow()

    since = int((datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=30)).timestamp()) * 1000

    end_time = convert_timestamp_tz(int(datetime.datetime.now().timestamp()) * 1000)
    time_range_obj = TimeRange(since, end_time)

    load_model(model_obj)

    return time_range_obj
