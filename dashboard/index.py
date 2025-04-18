import datetime
import os
import time
from collections.abc import Hashable
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from dash import Dash, Input, Output, html
from shapely.geometry import Point, Polygon

from analytics.grouper import Grouper
from dashboard.components import graphs, header, maps, metadata, ml_params, tables
from dashboard.init import init_app
from dashboard.models import Model
from dashboard.train import train
from dashboard.update_data import update_data, update_data_from_api
from dashboard.update_model import load_model
from utils import utils
from utils.utils import MINUTES_BETWEEN_UPDATES_FROM_API, TZ, logger

model = Model()

app = Dash(
    __name__,
    update_title="",
    meta_tags=metadata.meta_tags,
)

time_range = init_app(model)


# This function initializes the schedulers
# and ensure to initialize one per worker
# because in production, multiple workers are created
def initialize_scheduler() -> None:
    """Initialize jobs that are executed periodically"""

    scheduler = BackgroundScheduler()
    worker = int(os.environ.get("APSC", "0"), 0)

    if scheduler.running or worker:
        logger.info("Scheduler initialized, skiping initialization")
        return

    logger.info("Initializing scheduler with worker %i", worker)
    scheduler.add_job(
        update_data_from_api,
        "interval",
        minutes=MINUTES_BETWEEN_UPDATES_FROM_API,
        id="update-data-API",
        replace_existing=True,
    )
    scheduler.add_job(train, "interval", days=30, id="train-model", replace_existing=True)
    scheduler.add_job(
        load_model,
        "interval",
        args=[model],
        days=30,
        minutes=5,
        id="load-model",
        replace_existing=True,
    )
    scheduler.start()


initialize_scheduler()

names = {
    "all": "Evento",
    "ACCIDENT": "Accidente",
    "JAM": "Congestión",
    "HAZARD": "Peligro",
    "ROAD_CLOSED": "Camino cerrado",
}

app._favicon = "favicon.png"
app.title = "Gestión del tráfico en Antofagasta"

server = app.server
app.layout = html.Div(
    [
        header.get_header(),
        html.Div(
            [maps.main_map, tables.street_table],
            className="row",
        ),
        html.Div(
            [
                graphs.hourly_graph,
                graphs.daily_graph,
            ],
            className="row",
        ),
        html.Div(
            [
                tables.last_events_table,
                graphs.scatter_graph,
            ],
            className="row",
        ),
        html.H1("Modelo predictivo", className="title_ml"),
        html.Div(
            [
                ml_params.get_ml_params_component(model),
                maps.ml_map,
            ],
            className="row",
        ),
    ],
    className="container",
)


# CALLBACKS


def update_ml(  # noqa: PLR0913, PLR0917
    hour: int,
    day_type: int,
    week_day: int,
    day: int,
    kind: str,
    higlighted_segment: int | None = None,
) -> tuple[go.Figure, list[dict[Hashable, Any]]]:
    """Prepare data for updating ML in graphics"""

    if model is None or model.model is None:
        return go.Figure(), []

    alerts = update_data(time_range)

    if alerts.is_empty:
        return go.Figure(), []

    g = Grouper(cast("gpd.GeoDataFrame", alerts.data), (10, 20))

    x_var = pd.DataFrame(
        {
            "day_type": [day_type],
            "hour": [hour],
            "week_day": [week_day],
            "day": [day],
            "group": [0],
            "type_ACCIDENT": [1 if "ACCIDENT" in kind else 0],
            "type_JAM": [1 if "JAM" in kind else 0],
            "type_HAZARD": [1 if "HAZARD" in kind else 0],
            "type_ROAD_CLOSED": [1 if "ROAD_CLOSED" in kind else 0],
        },
        columns=model.model.feature_names_in_,
    )

    if kind == "all":
        x_var["type_ACCIDENT"] = 1
        kind = "ACCIDENT"

    if g.x_grid is None or g.y_grid is None:
        logger.error("x_grid and y_grid must not be None, %s, %s", g.x_grid, g.y_grid)
        return go.Figure(), []

    # Convert coordinates to a group of points (x, y)
    points = [
        Point(x, y) for x_row, y_row in zip(g.x_grid, g.y_grid, strict=True) for x, y in zip(x_row, y_row, strict=True)
    ]

    if points:
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3857")
        # Convert to EPSG:4326 (lat/lon)
        gdf = gdf.to_crs(epsg=4326)

        if gdf is None:
            msg = "Something was wrong, the geometry is null, verify the points"
            raise ValueError(msg)
        # Extract converted coordinates
        lats = sorted(set(gdf.geometry.y.tolist()))
        lons = sorted(set(gdf.geometry.x.tolist()))

    else:
        lats = []
        lons = []

    polygons = []

    for j in range(len(lons) - 1):
        for i in range(len(lats) - 1):
            poly = Polygon([
                (lons[j], lats[i]),
                (lons[j + 1], lats[i]),
                (lons[j + 1], lats[i + 1]),
                (lons[j], lats[i + 1]),
            ])
            polygons.append(poly)

    # Convert to GeoDataFrame
    gdf_polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    gdf_polygons["probability"] = [
        (
            model.model.predict_proba(x_var.assign(group=g.calc_quadrant(j, i)))[0][1]
            if g.calc_quadrant(j, i) in set(alerts.data["group"])
            else 0
        )
        for j in range(len(lons) - 1)
        for i in range(len(lats) - 1)
    ]
    gdf_polygons["segment"] = [
        f"Segmento {g.calc_quadrant(j, i)}" for j in range(len(lons) - 1) for i in range(len(lats) - 1)
    ]
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=gdf_polygons.__geo_interface__,
            locations=gdf_polygons.index,
            z=gdf_polygons["probability"],
            text=gdf_polygons["segment"],
            hovertemplate="<b>%{text}</b><br>Probabilidad de evento: %{z:.3f}<extra></extra>",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            marker_opacity=0.3,
            marker_line_width=1,
        )
    )

    if higlighted_segment:
        # Filter the selected polygon
        selected_poly = gdf_polygons[gdf_polygons["segment"] == f"Segmento {higlighted_segment}"]

        # Extract the coordinates
        lat_coords = selected_poly.geometry.apply(lambda poly: [point[1] for point in poly.exterior.coords]).values[0]
        lon_coords = selected_poly.geometry.apply(lambda poly: [point[0] for point in poly.exterior.coords]).values[0]

        fig.add_trace(
            go.Scattermapbox(
                lat=lat_coords,
                lon=lon_coords,
                mode="lines",
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line={"color": "red", "width": 2},
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        mapbox={
            "style": "carto-darkmatter",
            "center": {
                "lat": lats[int(len(lats) / 2)],
                "lon": lons[int(len(lons) / 2)],
            },
            "zoom": 11.5,
        },
        title={
            "text": f"Modelo predictivo:<br>Probabilidad de {names[kind].lower()}",
            "font": {
                "color": "#ccc",
                "weight": 600,
                "family": "Verdana",
                "size": 20,
            },
        },
        margin={"r": 0, "t": 95, "l": 0, "b": 0},
    )

    gdf_polygons["segment"] = gdf_polygons["segment"].str.replace("Segmento ", "", regex=False)
    gdf_polygons["probability"] = gdf_polygons["probability"].round(3)

    table_data = (
        gdf_polygons.loc[:, ["segment", "probability"]]
        .rename(columns={"segment": "Segmento", "probability": "Probabilidad"})
        .sort_values(by="Probabilidad", ascending=False)
        .to_dict("records")
    )

    return fig, table_data


@app.callback(
    [
        Output("map", "figure"),
        Output("hourly", "figure"),
        Output("table", "data"),
        Output("daily", "figure"),
        Output("scatter", "figure"),
        Output("date_range", "end_date"),
    ],
    [
        Input("dd_type", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("table", "active_cell"),
    ],
)
def update_graphs(
    kind: str, start_date: str | None, end_date: str | None, active_cell: list[dict[Hashable, Any]] | None
) -> tuple[go.Figure, go.Figure, list[dict[Hashable, Any]], go.Figure, go.Figure, datetime.datetime]:
    """Update all exploratory graphics"""

    default_return = (
        go.Figure(),
        go.Figure(),
        [],
        go.Figure(),
        go.Figure(),
        datetime.datetime.now(pytz.timezone(TZ)),
    )
    if start_date is None or end_date is None:
        return default_return

    start = (
        (datetime.datetime.fromisoformat(start_date))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(tz=pytz.timezone(TZ))
    )
    end = (
        (datetime.datetime.fromisoformat(end_date))
        .replace(hour=23, minute=59, second=59, microsecond=0)
        .astimezone(tz=pytz.timezone(TZ))
    )

    time_range.init_time = int(start.astimezone(pytz.UTC).timestamp()) * 1000
    time_range.end_time = int(end.astimezone(pytz.UTC).timestamp()) * 1000

    alerts = update_data(time_range)

    if alerts.is_empty:
        return default_return

    perf_init = time.perf_counter()

    if end <= start:
        end = start + datetime.timedelta(hours=23, minutes=59)  # One day of difference

    filtered_alerts = alerts.data

    filtered_alerts = filtered_alerts[(filtered_alerts["pub_millis"] >= start) & (filtered_alerts["pub_millis"] <= end)]

    scatter_data_acc = filtered_alerts[filtered_alerts["type"] == "ACCIDENT"]
    scatter_data_acc = utils.hourly_group(scatter_data_acc, do_sum=True)

    scatter_data_jam = filtered_alerts[filtered_alerts["type"] == "JAM"]
    scatter_data_jam = utils.hourly_group(scatter_data_jam, do_sum=True)

    if kind is not None and kind != "all":
        streets_data = filtered_alerts[filtered_alerts["type"] == kind].groupby("street")["type"].count().reset_index()
    else:
        streets_data = filtered_alerts.groupby("street")["type"].count().reset_index()

    streets_data = streets_data.rename(columns={"street": "Calle", "type": "Eventos"})
    streets_data = streets_data.sort_values(by="Eventos", ascending=False)

    table_data = streets_data.to_dict("records")

    if active_cell is not None:
        filtered_alerts = filtered_alerts.loc[
            filtered_alerts["street"] == streets_data.iloc[active_cell["row"]]["Calle"]  # type: ignore
        ]

    events = filtered_alerts if kind == "all" else filtered_alerts[filtered_alerts["type"] == kind]

    hourly = utils.hourly_group(events).reset_index().copy()
    hourly = pd.melt(
        hourly,
        id_vars=["hour"],
        value_vars=["s", "f"],
        var_name="day_type",
        value_name="events",
    )

    daily = utils.daily_group(events).reset_index().copy()
    daily = pd.melt(
        daily,
        id_vars=["day"],
        value_vars=["s", "f"],
        var_name="day_type",
        value_name="events",
    )

    map_data = cast("gpd.GeoDataFrame", filtered_alerts)
    utils.freq_nearby(map_data, nearby_meters=200)
    if map_data is None:
        msg = "Map data is empty"
        raise ValueError(msg)
    map_data["freq"] = map_data.apply(lambda x: x["freq"] if x["freq"] > 0 else 1, axis=1)

    map_data["time"] = map_data.pub_millis.apply(lambda x: x.strftime("%H:%M:%S"))
    map_data["date"] = map_data.pub_millis.apply(lambda x: x.strftime("%d-%m-%Y"))

    map_data = map_data[map_data["type"] == kind] if kind and kind != "all" else map_data

    map_fig = go.Figure(
        px.scatter_map(
            map_data,
            lat="y",
            lon="x",
            color="type",
            hover_name="uuid",
            size="freq",
            zoom=11,
            labels={
                "type": "Tipo de evento",
                "freq": "Frecuencia",
                "x": "Longitud",
                "y": "Latitud",
                "street": "Calle",
                "time": "Hora de reporte",
                "date": "Fecha de reporte",
                "group": "Segmento",
            },
            hover_data={"street": True, "time": True, "date": True, "group": True},
            opacity=0.8,
        )
    )

    hourly_fig = go.Figure()
    for day_type in ["s", "f"]:
        hourly_fig.add_trace(
            go.Bar(
                x=hourly["hour"].astype(str) + ":00",
                y=hourly[hourly["day_type"] == day_type]["events"],
                opacity=0.8,
                name={"f": "Fin de semana / Feriado", "s": "Semana"}[day_type],
                marker_color={"f": "salmon", "s": "skyblue"}[day_type],
                hoverlabel={
                    "bgcolor": "lightblue",
                    "font_size": 12,
                    "font_family": "Arial",
                    "font_color": "black",
                },
            ),
        )

    # Polygon style
    hourly_fig.update_traces(
        hovertemplate="x: %{x}<br>y: %{y}<extra></extra>",
    )

    hourly_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        autosize=True,
        width=None,
        title={
            "text": f"Promedio de {names[kind].lower()} por hora del día",
            "font": {
                "color": "#ccc",
                "weight": 600,
                "family": "Verdana",
                "size": 18,
            },
            "pad": {"b": 30, "l": 50},
            "x": 0.5,
        },
        grid_ygap=1,
        # nogrid
        yaxis={"title": f"Promedio de {names[kind].lower()} reportados", "showgrid": False},
        xaxis={"title": "Hora del día", "showgrid": False},
        barmode="group",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgb(37, 37, 37)",
        },
        margin={"t": 120, "b": 20, "l": 100, "r": 30},
    )

    daily_fig = go.Figure()
    for day_type in ["s", "f"]:
        daily_fig.add_trace(
            go.Bar(
                x=daily["day"].astype(str),
                y=daily[daily["day_type"] == day_type]["events"],
                opacity=0.8,
                name={"f": "Fin de semana / Feriado", "s": "Semana"}[day_type],
                marker_color={"f": "rgb(210, 41, 252)", "s": "rgb(34, 243, 173)"}[day_type],
                hoverlabel={
                    "bgcolor": "lightgreen",
                    "font_size": 12,
                    "font_family": "Arial",
                    "font_color": "black",
                },
            ),
        )

    # Tooltip style
    daily_fig.update_traces(
        hovertemplate="x: %{x}<br>y: %{y}<extra></extra>",
    )

    daily_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        autosize=True,
        width=None,
        title={
            "text": f"Promedio de {names[kind].lower()} por día del mes",
            "font": {
                "color": "#ccc",
                "weight": 600,
                "family": "Verdana",
                "size": 18,
            },
            "pad": {"b": 30, "l": 50},
            "x": 0.5,
        },
        grid_ygap=1,
        # nogrid
        yaxis={"title": f"Promedio de {names[kind].lower()} reportados", "showgrid": False},
        xaxis={"title": "Día del mes", "showgrid": False},
        barmode="group",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgb(37, 37, 37)",
        },
        margin={"t": 120, "b": 20, "l": 100, "r": 30},
    )

    map_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel_font={"color": "#fff"},
        showlegend=False,
        title={
            "text": f"Reportes de {names[kind].lower()}",
            "font": {
                "color": "#ccc",
                "weight": 600,
                "family": "Verdana",
                "size": 20,
            },
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"b": 30, "l": 50},
        },
        margin={"t": 80, "b": 20, "l": 40, "r": 30},
        map_style="dark",
    )

    if kind == "all":
        map_fig.update_layout(
            showlegend=True,
            legend_font={"color": "#ccc"},
        )

    # SCATTER PLOT

    scatter_data = scatter_data_jam.join(scatter_data_acc, on="hour", lsuffix="_jam", rsuffix="_accident").reset_index()
    scatter_data = scatter_data[
        (scatter_data["s_jam"] > 0)
        & (scatter_data["s_accident"] > 0)
        & (scatter_data["f_jam"] > 0)
        & (scatter_data["f_accident"] > 0)
    ]

    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=scatter_data["s_jam"],
            y=scatter_data["s_accident"],
            mode="markers",
            # "Bubble" style
            marker={
                "color": "rgba(54, 170, 263, 0.6)",
                "size": 20,
                "line": {"width": 1, "color": "DarkSlateGrey"},
            },
            name="Día de semana",
        )
    )

    scatter_fig.add_trace(
        go.Scatter(
            x=scatter_data["f_jam"],
            y=scatter_data["f_accident"],
            mode="markers",
            # "Bubble" style
            marker={
                "color": "rgba(170, 54, 263, 0.6)",
                "size": 20,
                "line": {"width": 1, "color": "DarkSlateGrey"},
            },
            name="Fin de semana / Feriado",
            xaxis="x2",
            yaxis="y2",
        )
    )

    scatter_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#ccc",
            "weight": 500,
            "family": "Verdana",
            "size": 14,
        },
        xaxis2={
            "title": "Eventos de congestión en fin de semana",
            "overlaying": "x",
            "side": "top",
            "showgrid": False,
            "title_font": {"size": 12},
            "showline": False,
            "zeroline": False,
        },
        yaxis2={
            "title": "Eventos de accidentes en fin de semana",
            "overlaying": "y",  # Overlays the axis on the first one
            "side": "right",  # Positions the secondary axis
            "showgrid": False,
            "title_font": {"size": 12},
            "showline": False,
            "zeroline": False,
        },
        hoverlabel_font={"color": "#fff"},
        title="Relación entre congestión de tráfico y accidentes",
        xaxis={
            "title": "Eventos de congestión día de semana",
            "showgrid": False,
            "title_font": {"size": 12},
        },
        yaxis={
            "title": "Eventos de accidente día de semana",
            "showgrid": False,
            "title_font": {"size": 12},
        },
        showlegend=True,
        margin={"t": 180},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.45,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgb(37, 37, 37)",
            "font": {
                "size": 10,
                "color": "white",
            },
        },
    )

    logger.info("Update graphs callback completed.")
    logger.info("Process time -> %.3fs", time.perf_counter() - perf_init)

    return map_fig, hourly_fig, table_data, daily_fig, scatter_fig, end


@app.callback(
    [Output("ML", "figure"), Output("table_ml", "data")],
    [
        Input("dd_type_ml", "value"),
        Input("date_ml", "date"),
        Input("dd_hour_ml", "value"),
        Input("table_ml", "active_cell"),
        Input("table_ml", "data"),
        Input("table_ml", "page_current"),
        Input("table_ml", "page_size"),
    ],
)
def update_ml_graphs(  # noqa: PLR0913, PLR0917
    kind: str | None,
    date_value: str,
    hour: int,
    active_cell: dict | None,
    table_data: list[dict[Hashable, Any]] | None,
    page_current: int,
    page_size: int,
) -> tuple[go.Figure, list[dict[Hashable, Any]]]:
    """Update ML graphs with filters selected by user"""

    alerts = update_data(time_range)

    if alerts.is_empty:
        return go.Figure(), []
    if kind is None:
        kind = "ACCIDENT"

    date_ml = datetime.date.today() if not date_value else datetime.date.fromisoformat(date_value)

    perf_init = time.perf_counter()

    day = date_ml.day
    week_day = date_ml.weekday()
    saturday = 5
    day_type = 0 if (date_ml.weekday() >= saturday) else 1  # TODO: Introduce holidays here

    segment = None
    if table_data and active_cell:
        global_index = page_current * page_size + active_cell["row"]
        segment = table_data[global_index]["Segmento"]

    group_fig, table_data = update_ml(hour, day_type, week_day, day, kind, segment)

    logger.info("Update ML graphs callback completed.")
    logger.info("Process time -> %.3fs", time.perf_counter() - perf_init)

    return group_fig, table_data


@app.callback(
    Output("table_last", "data"),
    Input("dd_type", "value"),
)
def update_last_events(kind: str) -> list[dict[Hashable, Any]]:
    """Update the last 20 events in table"""
    alerts = update_data(time_range)
    if alerts.is_empty:
        return []
    last_events = alerts.data.sort_values(by="pub_millis", ascending=False)

    last_events["date"] = last_events.pub_millis.apply(lambda x: x.strftime("%d-%m-%Y"))
    last_events["date"] = last_events.pub_millis.dt.strftime("%d/%m/%Y")

    last_events["hour"] = last_events.apply(lambda row: f"{int(row['hour']):02}:{int(row['minute']):02}", axis=1)

    last_events = last_events[["type", "group", "hour", "date", "street"]]

    if last_events.shape[0] == 0:
        return []

    if kind is not None and kind != "all":
        last_events = last_events[last_events["type"] == kind]

    last_events = last_events.iloc[:20]

    if last_events.shape[0] == 0:
        return []

    last_events = last_events.rename(
        columns={
            "type": "Tipo",
            "group": "Segmento",
            "hour": "Hora",
            "date": "Fecha",
            "street": "Calle",
        }
    )

    return last_events.to_dict("records")


# Clear filter in table selections
@app.callback(Output("table_ml", "active_cell"), Input("table_ml_clear", "n_clicks"))
def clear_ml(_: int) -> None:
    """Clear filters on ML table"""
    return


@app.callback(Output("table", "active_cell"), Input("table_clear", "n_clicks"))
def clear(_: int) -> None:
    """Clear filters on ML table"""
    return


@app.callback(Output("date_range", "max_date_allowed"), Input("interval_update", "n_intervals"))
def update_max_date_allowed(_: int) -> datetime.datetime:
    """Update the max_data_allowed field within the date range, necessary to keep it current"""
    return datetime.datetime.now(pytz.timezone(TZ))


# Show instructions button
app.clientside_callback(
    """
        function(n_clicks){
            if (typeof(n_clicks) === 'undefined' || n_clicks === null){
                return 'instructions hidden';
            }
            if (n_clicks % 2 === 1){
                return 'instructions show';
            }
            return 'instructions hidden';
        }
        """,
    Output("instructions", "className"),
    Input("toggle_instructions", "n_clicks"),
)
