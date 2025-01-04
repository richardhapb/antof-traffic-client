import datetime
import pandas as pd
import geopandas as gpd

from waze.events import Events
from utils import utils


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
    events = utils.load_data("alerts")
    assert isinstance(events, Events)
    assert len(events.data) > 0
    assert isinstance(events.data, list)
    assert isinstance(events.data[0], dict)


def test_update_timezone():
    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    assert curr_time.tzname() is None
    assert future_time.tzname() is None

    df = pd.DataFrame({"pubMillis": [curr_time], "endreport": [future_time]})

    df2 = utils.update_timezone(df, "America/Santiago")

    assert str(df2["pubMillis"].dt.tz) == "America/Santiago"
    assert str(df2["endreport"].dt.tz) == "America/Santiago"


def test_get_holidays():
    holidays = utils.get_holidays()
    assert isinstance(holidays, list)
    assert len(holidays) > 0


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


def test_extract_event():
    nearby = 10
    n = 15
    df = generate_dummy_coord_df(nearby, n)

    freq_df = utils.freq_nearby(utils.separate_coords(df))

    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    freq_df["pubMillis"] = [curr_time] * n
    freq_df["endreport"] = [future_time] * n

    df2 = utils.update_timezone(freq_df, "America/Santiago")
    df2["type"] = ["ACCIDENT"] * n

    extra_cols = ["freq", "geometry", "type"]
    df3 = utils.extract_event(df2, ["ACCIDENT"], extra_cols)

    for c in df3.columns:
        assert c in extra_cols + ["inicio", "fin"]


def test_hourly_group():
    nearby = 10
    n = 15
    df = generate_dummy_coord_df(nearby, n)

    freq_df = utils.freq_nearby(utils.separate_coords(df))

    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    freq_df["pubMillis"] = [curr_time] * n
    freq_df["endreport"] = [future_time] * n

    df2 = utils.update_timezone(freq_df, "America/Santiago")
    df2["type"] = ["ACCIDENT"] * n

    extra_cols = ["freq", "geometry", "type", "day_type", "hour"]
    df3 = utils.extract_event(df2, ["ACCIDENT"], extra_cols)

    hourly = utils.hourly_group(df3)

    assert hourly.shape[0] == 24


def test_daily_group():
    nearby = 10
    n = 15
    df = generate_dummy_coord_df(nearby, n)

    freq_df = utils.freq_nearby(utils.separate_coords(df))

    curr_time = datetime.datetime.now()
    future_time = curr_time + datetime.timedelta(hours=2)

    freq_df["pubMillis"] = [curr_time] * n
    freq_df["endreport"] = [future_time] * n

    df2 = utils.update_timezone(freq_df, "America/Santiago")
    df2["type"] = ["ACCIDENT"] * n

    extra_cols = ["freq", "geometry", "type", "day_type", "day"]
    df3 = utils.extract_event(df2, ["ACCIDENT"], extra_cols)

    daily = utils.daily_group(df3)

    assert daily.shape[0] == 31
