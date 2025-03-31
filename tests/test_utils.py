import datetime

import geopandas as gpd
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import pytz

from utils import utils
from utils.utils import MINUTES_BETWEEN_UPDATES_FROM_API
from waze.alerts import Alerts


def generate_alerts_data():
    """Generate data for testing"""
    return [
        {
            "uuid": "16272ee8-9a60-4dea-a4b1-76a8281732d4",
            "reliability": 6,
            "type": "HAZARD",
            "roadType": 2,
            "magvar": 356.0,
            "subtype": "HAZARD_ON_SHOULDER_CAR_STOPPED",
            "location": {"x": -70.38938, "y": -23.628653},
            "street": "Av. Antonio Rendic",
            "pub_millis": 1741183860000,
            "end_pub_millis": 1741184089898,
            "group": 103,
            "day_type": "s",
            "week_day": 2,
            "day": 5,
            "hour": 11,
            "minute": 11,
        },
        {
            "uuid": "3ed960c0-13e9-441d-baf9-7a27181c35a4",
            "reliability": 6,
            "type": "JAM",
            "roadType": 6,
            "magvar": 210.0,
            "subtype": "JAM_HEAVY_TRAFFIC",
            "location": {"x": -70.39465, "y": -23.623049},
            "street": "Av. Edmundo Pérez Zujovic",
            "pub_millis": 1728938943000,
            "end_pub_millis": 1728890661000,
            "group": 104,
            "day_type": "s",
            "week_day": 0,
            "day": 14,
            "hour": 17,
            "minute": 49,
        },
        {
            "uuid": "f51a5fbc-1a4a-4a36-9c4e-b8c67598e1f3",
            "reliability": 5,
            "type": "JAM",
            "roadType": 2,
            "magvar": 44.0,
            "subtype": "JAM_STAND_STILL_TRAFFIC",
            "location": {"x": -70.392654, "y": -23.54594},
            "street": None,
            "pub_millis": 1732204310000,
            "end_pub_millis": 1732204732194,
            "group": 110,
            "day_type": "s",
            "week_day": 3,
            "day": 21,
            "hour": 12,
            "minute": 51,
        },
    ]


def generate_dummy_coord_df(n_nearby: int, n_total: int) -> pd.DataFrame:
    x_base = -70.42034224747098
    y_base = -23.721724880116387

    x = []
    y = []
    for _ in range(n_nearby):
        x.append(x_base)
        y.append(y_base)

    for n in range(n_total - n_nearby):
        x.append(x_base + n * 10)
        y.append(y_base + n * 10)

    df = pd.DataFrame({
        "dummy": ["data"] * n_total,
        "location": [{"x": a, "y": b} for a, b in zip(x, y)],
    })

    return df


def test_get_data():
    Alerts._instance = None
    alerts = utils.get_data()
    assert isinstance(alerts, Alerts)
    assert alerts.data.shape[0] > 0
    assert not alerts.is_empty
    assert isinstance(alerts.data, gpd.GeoDataFrame)


def test_get_data_multiple():
    """
    Test multiple requests for data in a short period of time

    Simulate behavior in the graph when multiple requests are made
    """
    Alerts._instance = None

    now = datetime.datetime.now(pytz.UTC)
    since = int((now - datetime.timedelta(days=30)).timestamp()) * 1000
    until = int((now - datetime.timedelta(minutes=MINUTES_BETWEEN_UPDATES_FROM_API)).timestamp()) * 1000

    calls_when_request_to_server = 2

    # Side effect
    alerts_empty = Alerts()
    alerts_full = Alerts(generate_alerts_data())

    # Reset data of Singleton class
    delattr(Alerts._instance, "data")
    Alerts._instance = None

    with patch("utils.utils.Alerts") as mock_alerts:
        mock_alerts.side_effect = [alerts_empty, alerts_full]

        alerts = utils.get_data(since, until)
        assert mock_alerts.call_count == calls_when_request_to_server

    for _ in range(4):
        alerts = utils.get_data(since, until)
        assert isinstance(alerts, Alerts)
        assert not alerts.is_empty
        assert alerts.data.shape[0] > 0


def test_update_timezone():
    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    assert curr_time.tzname() is None
    assert future_time.tzname() is None

    df = gpd.GeoDataFrame({"pub_millis": [curr_time], "end_pub_millis": [future_time]})

    df2 = utils.update_timezone(df, utils.TZ)

    assert str(df2["pub_millis"].dt.tz) == utils.TZ
    assert str(df2["end_pub_millis"].dt.tz) == utils.TZ


def test_separate_coords():
    n = 15
    df = generate_dummy_coord_df(10, n)

    gdf = utils.separate_coords(df)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.shape[0] == n


def test_freq_nearby():
    nearby = 10
    df = generate_dummy_coord_df(nearby, 15)

    freq_df = utils.freq_nearby(utils.separate_coords(df))

    assert freq_df["freq"].max() == nearby


# TODO: Implement aggregate data on server for test this
@pytest.mark.skip
def test_hourly_group():
    nearby = 10
    n = 15
    df = generate_dummy_coord_df(nearby, n)

    freq_df = utils.freq_nearby(utils.separate_coords(df))

    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    freq_df["pub_millis"] = [curr_time] * n
    freq_df["end_pub_millis"] = [future_time] * n

    df2 = utils.update_timezone(freq_df)
    df2["type"] = ["ACCIDENT"] * n

    df3 = utils.generate_aggregate_data(df2)

    hourly = utils.hourly_group(df3.data)

    assert hourly.shape[0] == 24


# TODO: Implement aggregate data on server for test this
@pytest.mark.skip
def test_daily_group():
    nearby = 10
    n = 15
    df = generate_dummy_coord_df(nearby, n)

    freq_df = utils.freq_nearby(utils.separate_coords(df))

    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    freq_df["pub_millis"] = [curr_time] * n
    freq_df["end_pub_millis"] = [future_time] * n

    df2 = utils.update_timezone(freq_df)
    df2["type"] = ["ACCIDENT"] * n

    df3 = utils.generate_aggregate_data(df2)

    daily = utils.daily_group(df3.data)

    assert daily.shape[0] == 31
