import datetime
import pytz
from analytics.ml import init_mlflow


from dashboard.models import Model, Alerts, TimeRange
from dashboard.update_data import update_data
from dashboard.update_model import load_model
from dashboard.dash_config import TZ

init_mlflow()


def init_app(model_obj: Model, alerts_obj: Alerts) -> TimeRange:
    since = int(
        datetime.datetime(
            year=2024,
            month=10,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=pytz.timezone(TZ),
        ).timestamp()
        * 1000
    )

    end_time = int(datetime.datetime.now().timestamp()) * 1000
    time_range_obj = TimeRange(since, end_time)

    update_data(time_range_obj, alerts_obj)

    load_model(model_obj)

    return time_range_obj
