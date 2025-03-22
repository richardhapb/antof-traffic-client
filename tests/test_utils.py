import datetime

import geopandas as gpd
import pandas as pd
import pytest

from utils import utils
from waze.alerts import Alerts


def generate_dummy_coord_df(n_nearby: int, n_total: int) -> pd.DataFrame:
    x_base = -70.42034224747098
    y_base = -23.721724880116387

    x = []
    y = []
    for n in range(n_nearby):
        x.append(x_base)
        y.append(y_base)

    for n in range(n_total - n_nearby):
        x.append(x_base + n * 10)
        y.append(y_base + n * 10)

    df = pd.DataFrame(
        {
            "dummy": ["data"] * n_total,
            "location": [{"x": a, "y": b} for a, b in zip(x, y)],
        }
    )

    return df


def test_load_data():
    alerts = utils.get_data()
    assert isinstance(alerts, Alerts)
    assert alerts.data.shape[0] > 0
    assert not alerts.is_empty
    assert isinstance(alerts.data, gpd.GeoDataFrame)


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
@pytest.mark.skip()
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
@pytest.mark.skip()
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
