import copy
import datetime

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
from dashboard.update_data import update_data
from dashboard.update_model import load_model
from utils import utils
from utils.utils import TZ
from waze.alerts import Alerts

model = Model()
alerts = Alerts({})

time_range = init_app(model, alerts)

# Train and Load the model dinamically
scheduler = BackgroundScheduler()
scheduler.add_job(update_data, "interval", args=[time_range, alerts], minutes=5)
scheduler.add_job(train, "interval", days=30)
scheduler.add_job(load_model, "interval", args=[model], days=30, minutes=5)
scheduler.start()

names = {
    "all": "Evento",
    "ACCIDENT": "Accidente",
    "JAM": "Congestión",
    "HAZARD": "Peligro",
    "ROAD_CLOSED": "Camino cerrado",
}

app = Dash(
    __name__,
    update_title="",
    meta_tags=metadata.meta_tags,
)
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


## CALLBACKS


def update_ML(
    hour: int,
    day_type: int,
    week_day: int,
    day: int,
    kind: str,
    higlighted_segment: int | None = None,
):
    if model is None or model.model is None:
        return go.Figure()

    if alerts.is_empty:
        return go.Figure()

    g = Grouper(alerts.data, (10, 20))

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

    assert g.x_grid is not None, "Error: x_grid must not be None"
    assert g.y_grid is not None, "Error: y_grid must not be None"

    # Convert coordinates to a group of points (x, y)
    points = [Point(x, y) for x_row, y_row in zip(g.x_grid, g.y_grid) for x, y in zip(x_row, y_row)]

    if points:
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3857")
        # Convert to EPSG:4326 (lat/lon)
        gdf = gdf.to_crs(epsg=4326)

        if gdf is None:
            raise ValueError("Something was wrong, the geometry is null, verify the points")
        # Extract converted coordinates
        lats = sorted(list(set(gdf.geometry.y.tolist())))
        lons = sorted(list(set(gdf.geometry.x.tolist())))

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

    # Convertir a GeoDataFrame
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
        f"Segmento {g.calc_quadrant(j, i)}"
        for j in range(len(lons) - 1)
        for i in range(len(lats) - 1)
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
        # Filtra el polígono seleccionado
        selected_poly = gdf_polygons[gdf_polygons["segment"] == f"Segmento {higlighted_segment}"]

        # Extrae coordenadas del polígono seleccionado
        lat_coords = selected_poly.geometry.apply(
            lambda poly: [point[1] for point in poly.exterior.coords]
        ).values[0]
        lon_coords = selected_poly.geometry.apply(
            lambda poly: [point[0] for point in poly.exterior.coords]
        ).values[0]

        fig.add_trace(
            go.Scattermapbox(
                lat=lat_coords,
                lon=lon_coords,
                mode="lines",
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(color="red", width=2),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(
                lat=lats[int(len(lats) / 2)],
                lon=lons[int(len(lons) / 2)],
            ),
            zoom=11.5,
        ),
        title=dict(
            text=f"Modelo predictivo:<br>Probabilidad de {names[kind].lower()}",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=20,
            ),
        ),
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
def update_graphs(kind, start_date, end_date, active_cell):
    STANDARD_RETURN = (
        go.Figure(),
        go.Figure(),
        [],
        None,
        go.Figure(),
        datetime.datetime.now(pytz.timezone(TZ)),
    )
    if start_date is None or end_date is None:
        return STANDARD_RETURN

    if alerts.is_empty:
        return STANDARD_RETURN

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

    if end <= start:
        end = start.timestamp() + (23 * 60 * 60 + 60**2 * 59) * 1000

    filtered_alerts = copy.deepcopy(alerts.data)

    filtered_alerts = filtered_alerts[
        (filtered_alerts["pub_millis"] >= start) & (filtered_alerts["pub_millis"] <= end)
    ]

    scatter_data_acc = filtered_alerts[filtered_alerts["type"] == "ACCIDENT"]
    scatter_data_acc = utils.hourly_group(scatter_data_acc, do_sum=True)

    scatter_data_jam = filtered_alerts[filtered_alerts["type"] == "JAM"]
    scatter_data_jam = utils.hourly_group(scatter_data_jam, do_sum=True)

    if kind is not None and kind != "all":
        streets_data = (
            filtered_alerts[filtered_alerts["type"] == kind]
            .groupby("street")["type"]
            .count()
            .reset_index()
        )
    else:
        streets_data = filtered_alerts.groupby("street")["type"].count().reset_index()

    streets_data = streets_data.rename(columns={"street": "Calle", "type": "Eventos"})
    streets_data = streets_data.sort_values(by="Eventos", ascending=False)

    table_data = streets_data.to_dict("records")

    if active_cell is not None:
        filtered_alerts = filtered_alerts.loc[
            filtered_alerts["street"] == streets_data.iloc[active_cell["row"]]["Calle"]
        ]

    if kind == "all":
        events = filtered_alerts
    else:
        events = filtered_alerts[filtered_alerts["type"] == kind]

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

    map_data = copy.deepcopy(filtered_alerts)
    map_data = utils.freq_nearby(map_data, nearby_meters=200)
    if map_data is None:
        raise ValueError("Map data is empty")
    map_data["freq"] = map_data.apply(lambda x: x["freq"] if x["freq"] > 0 else 1, axis=1)

    map_data["time"] = map_data.pub_millis.apply(lambda x: x.strftime("%H:%M:%S"))
    map_data["date"] = map_data.pub_millis.apply(lambda x: x.strftime("%d-%m-%Y"))

    map_data["type"] = map_data["type"].map(names)

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
                hoverlabel=dict(
                    bgcolor="lightblue",
                    font_size=12,
                    font_family="Arial",
                    font_color="black",
                ),
            ),
        )

    # Configurar el estilo del tooltip para todas las trazas
    hourly_fig.update_traces(
        hovertemplate="x: %{x}<br>y: %{y}<extra></extra>",
    )

    hourly_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        autosize=True,
        width=None,
        title=dict(
            text=f"Promedio de {names[kind].lower()} por hora del día",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=18,
            ),
            pad=dict(b=30, l=50),
            x=0.5,
        ),
        grid_ygap=1,
        # nogrid
        yaxis=dict(title=f"Promedio de {names[kind].lower()} reportados", showgrid=False),
        xaxis=dict(title="Hora del día", showgrid=False),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgb(37, 37, 37)",
        ),
        margin=dict(t=120, b=20, l=100, r=30),
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
                hoverlabel=dict(
                    bgcolor="lightgreen",
                    font_size=12,
                    font_family="Arial",
                    font_color="black",
                ),
            ),
        )

    # Configurar el estilo del tooltip para todas las trazas
    daily_fig.update_traces(
        hovertemplate="x: %{x}<br>y: %{y}<extra></extra>",
    )

    daily_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        autosize=True,
        width=None,
        title=dict(
            text=f"Promedio de {names[kind].lower()} por día del mes",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=18,
            ),
            pad=dict(b=30, l=50),
            x=0.5,
        ),
        grid_ygap=1,
        # nogrid
        yaxis=dict(title=f"Promedio de {names[kind].lower()} reportados", showgrid=False),
        xaxis=dict(title="Día del mes", showgrid=False),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgb(37, 37, 37)",
        ),
        margin=dict(t=120, b=20, l=100, r=30),
    )

    map_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel_font=dict(color="#fff"),
        showlegend=False,
        title=dict(
            text=f"Reportes de {names[kind].lower()}",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=20,
            ),
            xanchor="left",
            yanchor="top",
            pad=dict(b=30, l=50),
        ),
        margin=dict(t=80, b=20, l=40, r=30),
        map_style="dark",
    )

    if kind == "all":
        map_fig.update_layout(
            showlegend=True,
            legend_font=dict(color="#ccc"),
        )

    ## SCATTER PLOT

    scatter_data = scatter_data_jam.join(
        scatter_data_acc, on="hour", lsuffix="_jam", rsuffix="_accident"
    ).reset_index()
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
            marker=dict(  # Tamaños variados para el efecto de burbuja
                color="rgba(54, 170, 263, 0.6)",  # Color azul con transparencia (0.6)
                size=20,
                line=dict(width=1, color="DarkSlateGrey"),  # Borde alrededor de las burbujas
            ),
            name="Día de semana",
        )
    )

    scatter_fig.add_trace(
        go.Scatter(
            x=scatter_data["f_jam"],
            y=scatter_data["f_accident"],
            mode="markers",
            marker=dict(  # Tamaños variados para el efecto de burbuja
                color="rgba(170, 54, 263, 0.6)",  # Color azul con transparencia (0.6)
                size=20,
                line=dict(width=1, color="DarkSlateGrey"),  # Borde alrededor de las burbujas
            ),
            name="Fin de semana / Feriado",
            xaxis="x2",
            yaxis="y2",
        )
    )

    # Configuración adicional para un estilo atractivo
    scatter_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#ccc",
            weight=500,
            family="Verdana",
            size=14,
        ),
        xaxis2=dict(
            title="Eventos de congestión en fin de semana",
            overlaying="x",
            side="top",
            showgrid=False,
            title_font=dict(size=12),
            showline=False,
            zeroline=False,
        ),
        yaxis2=dict(
            title="Eventos de accidentes en fin de semana",
            overlaying="y",  # Superpone el eje sobre el primero
            side="right",  # Ubica el eje secundario a la derecha
            showgrid=False,  # Opcional para ocultar la cuadrícula del segundo eje
            title_font=dict(size=12),
            showline=False,
            zeroline=False,
        ),
        hoverlabel_font=dict(color="#fff"),
        title="Relación entre congestión de tráfico y accidentes",
        xaxis=dict(
            title="Eventos de congestión día de semana",
            showgrid=False,
            title_font=dict(size=12),
        ),
        yaxis=dict(
            title="Eventos de accidente día de semana",
            showgrid=False,
            title_font=dict(size=12),
        ),
        showlegend=True,
        margin=dict(t=180),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.45,
            xanchor="right",
            x=1,
            bgcolor="rgb(37, 37, 37)",
            font=dict(
                size=10,
                color="white",
            ),
        ),
    )

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
def update_ml_graphs(kind, date_value, hour, active_cell, table_data, page_current, page_size):
    if alerts.is_empty:
        return go.Figure(), []
    if kind is None:
        kind = "ACCIDENT"
    if date_value is None:
        date_ml = datetime.date.today()
    else:
        date_ml = datetime.date.fromisoformat(date_value)

    day = date_ml.day
    week_day = date_ml.weekday()
    day_type = 0 if (date_ml.weekday() >= 5) else 1  # TODO: Introduce holidays here

    segment = None
    if table_data and active_cell:
        global_index = page_current * page_size + active_cell["row"]
        segment = table_data[global_index]["Segmento"]

    group_fig, table_data = update_ML(hour, day_type, week_day, day, kind, segment)

    return group_fig, table_data


@app.callback(
    [Output("table_last", "data"), Output("date_range", "max_date_allowed")],
    Input("dd_type", "value"),
)
def update_last_events(kind):
    if alerts.is_empty:
        return [], datetime.datetime.now(pytz.timezone(TZ))
    last_events = alerts.data.sort_values(by="pub_millis", ascending=False)

    if last_events.shape[0] == 0:
        return [], [datetime.datetime.now(pytz.timezone(TZ))]

    if kind != "all" and kind is not None:
        last_events = last_events[last_events["type"] == kind]

    last_events = last_events.iloc[:20]

    if last_events.shape[0] == 0:
        return [], datetime.datetime.now(pytz.timezone(TZ))

    last_events["hour"] = last_events.apply(
        lambda row: f"{int(row['hour']):02}:{int(row['minute']):02}", axis=1
    )

    last_events["date"] = last_events.pub_millis.dt.strftime("%d/%m/%Y")
    last_events["type"] = last_events.type.map(names)

    last_events = last_events[["type", "group", "hour", "date", "street"]]

    last_events = last_events.rename(
        columns={
            "type": "Tipo",
            "group": "Segmento",
            "hour": "Hora",
            "date": "Fecha",
            "street": "Calle",
        }
    )

    return last_events.to_dict("records"), datetime.datetime.now(pytz.timezone(TZ))


# Clear filter in table selections
@app.callback(Output("table_ml", "active_cell"), Input("table_ml_clear", "n_clicks"))
def clear_ml(_):
    return None


@app.callback(Output("table", "active_cell"), Input("table_clear", "n_clicks"))
def clear(_):
    return None


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
