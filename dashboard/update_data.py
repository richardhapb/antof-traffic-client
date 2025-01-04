import pytz
import datetime

from utils import utils
from dashboard.dash_config import TZ
from dashboard.models import TimeRange, Alerts
from analytics.grouper import Grouper


def update_data(time_range_obj: TimeRange, alerts_obj: Alerts):
    time_range_obj.end_time = int(
        (
            datetime.datetime.now(pytz.timezone(TZ)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        ).timestamp()
        * 1000
    )

    time_range_obj.selected_time = time_range_obj.end_time - 30 * 24 * 60**2 * 1000

    try:
        alerts_query = utils.load_data(
            "alerts", mode="since", epoch=time_range_obj.init_time
        )
        g = Grouper(alerts_query.to_gdf(tz=TZ))
        g.group((10, 20)).filter_by_group_time(60, True)
        alerts_obj.data = g
    except Exception as e:
        print("Something was wrong while updating alerts")
        print(f"Error: {e}")
