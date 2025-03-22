from typing import Any
from unittest.mock import patch, MagicMock
import datetime

import psycopg2
import pytest
import numpy as np
from waze.events import Events

ALERTS_STRUCTURE = Events.db_columns_map["alerts"]
JAMS_STRUCTURE = Events.db_columns_map["jams"]
N_ELEMENTS = 10

today = int(datetime.datetime.now().timestamp() * 1000)
TIME_DELTA = 3 * 24 * 60**2 * 1000  # 3 days
another_day = today - TIME_DELTA

# ALERTS DUMMY DATA

alerts_dummy_elements = []

uuid = 0

for _ in range(N_ELEMENTS):
    element = []
    for k, v in ALERTS_STRUCTURE.items():
        if not isinstance(v, dict):
            if k == "uuid":
                element.append(uuid)
                uuid += 1
            elif k == "pub_millis":
                element.append(np.random.choice([today, another_day]))
            elif k == "end_pub_millis":
                element.append(
                    np.random.choice([element[-1] + 60**2] + [None])
                )  # 1 hour
            else:
                element.append(np.random.randint(0, 20))
    alerts_dummy_elements.append(tuple(element))

ALERTS_DUMMY_ELEMENTS = alerts_dummy_elements

# JAMS DUMMY DATA

jams_dummy_elements = []

uuid = 0

for _ in range(N_ELEMENTS):
    element = []
    for k, v in JAMS_STRUCTURE.items():
        if not isinstance(v, dict):
            if k == "uuid":
                element.append(uuid)
                uuid += 1
            elif k == "pub_millis":
                element.append(np.random.choice([today, another_day]))
            elif k == "end_pub_millis":
                element.append(
                    np.random.choice([element[-1] + 60**2] + [None])
                )  # 1 hour
            else:
                element.append(np.random.randint(0, 20))
    jams_dummy_elements.append(tuple(element))

JAMS_DUMMY_ELEMENTS = jams_dummy_elements


@pytest.fixture(name="mock_db_connection_alerts")
def fixture_db_connection_alerts():
    """
    Simule a db connection for Alerts
    Args:
    :param None: No parameters.
    Returns:
        Return type: Mock DB connection
    """
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_connection.__class__ = psycopg2.extensions.connection

    mock_cursor.fetchall.return_value = ALERTS_DUMMY_ELEMENTS
    mock_cursor.fetchone.return_value = [0]
    return mock_connection


@pytest.fixture(name="mock_db_connection_jams")
def fixture_db_connection_jams():
    """
    Simule a db connection
    Args:
    :param None: No parameters.
    Returns:
        Return type: Mock DB connection
    """
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_connection.__class__ = psycopg2.extensions.connection

    mock_cursor.fetchall.return_value = JAMS_DUMMY_ELEMENTS
    mock_cursor.fetchone.return_value = [0]
    return mock_connection


def test_format_data():
    """
    Test formating of events data
    Args:
    :param None: No parameters.
    Returns:
        Return list[dict[str, Any]]: Data formated
    """
    data_alerts = Events.format_data(ALERTS_DUMMY_ELEMENTS, "alerts")
    data_jams = Events.format_data(JAMS_DUMMY_ELEMENTS, "jams")

    assert data_alerts[0].keys() == ALERTS_STRUCTURE.keys()
    assert data_jams[0].keys() == JAMS_STRUCTURE.keys() - ["line", "segments"]


@patch("psycopg2.connect")
def test_fetch_from_db(mock_connect, mock_db_connection_alerts):
    mock_connect.return_value = mock_db_connection_alerts

    events = Events(table_name="alerts")
    events.fetch_from_db(mode="last_24h")

    assert len(events.data) == N_ELEMENTS
    assert isinstance(events.data, list)
    assert events.data[0].keys() == ALERTS_STRUCTURE.keys()
    assert events.pending_end_pub_milliss == {
        e["uuid"]
        for e in events.data
        for k, v in e.items()
        if k == "end_pub_millis" and v is None
    }
    assert events.index_map.keys() == {e["uuid"] for e in events.data}


@patch("psycopg2.connect")
def test_get_all_from_db(mock_connect, mock_db_connection_alerts) -> None:
    """
    Test the database input method
    Args:
    :param mock_connect: psycopg2.connect method reference
    :param mock_db_connection_alerts: fixture function for test
    Returns:
        Return None
    """
    mock_connect.return_value = mock_db_connection_alerts

    events = Events(table_name="alerts")
    data = events.get_all_from_db(mode="last_24h")

    mock_db_connection_alerts.cursor.assert_called_once()

    assert data is not None and len(data) == N_ELEMENTS
    assert Events.format_data(ALERTS_DUMMY_ELEMENTS, "alerts") == data


def generate_new_elements_dummy(new_diff_count: int) -> list[dict[str, Any]]:
    """
    Generate new_elements for the new_diff_count insert data
    Args:
    :param news_diff_count: int: {}
    Returns:
        Return list[dict[str, any]]: News elements to insert
    """
    new_elements = []

    count = 0
    for e in ALERTS_DUMMY_ELEMENTS:
        if count >= new_diff_count:
            break
        new_elements.append((e[0] + 30,) + e[:1])
        count += 1

    new_data = Events.format_data(new_elements, "alerts")
    for n in new_data:
        n["location"] = {"x": np.random.randint(0, 20), "y": np.random.randint(0, 20)}

    return new_data


@patch("psycopg2.connect")
@pytest.mark.parametrize("new_diff_elements", range(0, N_ELEMENTS + 1))
def test_insert_to_db(mock_connect, mock_db_connection_alerts, new_diff_elements):
    mock_connect.return_value = mock_db_connection_alerts

    new_data = generate_new_elements_dummy(new_diff_elements)

    events = Events(
        new_data,
        table_name="alerts",
    )

    events.insert_to_db()

    mock_cursor = mock_db_connection_alerts.cursor()
    args_list = mock_cursor.execute.call_args_list

    calls_location_insertion = 0
    calls_alert_insertion = 0
    for a in args_list:
        calls_location_insertion += (
            1 if "INSERT INTO alerts_location" in a.args[0] else 0
        )
        calls_alert_insertion += 1 if "INSERT INTO alerts (" in a.args[0] else 0

    assert calls_location_insertion == new_diff_elements
    assert calls_alert_insertion == new_diff_elements
