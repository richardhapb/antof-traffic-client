from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
from utils import utils
import datetime
import pytz
import plotly.graph_objs as go
import pandas as pd
import mlflow
from analytics.ml import init_mlflow
from analytics.grouper import Grouper
import geopandas as gpd
from shapely.geometry import Point, Polygon

init_mlflow()
model = mlflow.sklearn.load_model("models:/XGBClassifier/149")

names = {
    "all": "Evento",
    "ACCIDENT": "Accidente",
    "JAM": "Congestión",
    "HAZARD": "Peligro",
    "ROAD_CLOSED": "Camino cerrado",
}


tz = "America/Santiago"

app = Dash(__name__)

since = int(
    datetime.datetime(
        year=2024,
        month=10,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=pytz.timezone(tz),
    ).timestamp()
    * 1000
)

selected_time = int(
    (
        datetime.datetime.now(pytz.timezone(tz)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    ).timestamp()
    * 1000
)

alerts = utils.load_data("alerts", mode="since", epoch=since)
alerts = alerts.to_gdf(tz=tz)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Inteligencia de Tráfico para la Gestión Urbana en Antofagasta"
                ),
                html.H2(
                    "Análisis de patrones y eventos reportados por usuarios para gestión de tráfico",
                    className="subtitle",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="dd_type",
                            options=[
                                {"label": "Todos", "value": "all"},
                                {"label": "Accidentes", "value": "ACCIDENT"},
                                {"label": "Congestión", "value": "JAM"},
                                {"label": "Peligros", "value": "HAZARD"},
                                {"label": "Caminos cerrados", "value": "ROAD_CLOSED"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                        dcc.DatePickerRange(
                            start_date_placeholder_text="Desde",
                            end_date_placeholder_text="Hasta",
                            id="date_range",
                            min_date_allowed=datetime.datetime.fromtimestamp(
                                since / 1000, pytz.timezone(tz)
                            ),
                            max_date_allowed=datetime.datetime.now(pytz.timezone(tz)),
                            initial_visible_month=datetime.datetime.fromtimestamp(
                                selected_time / 1000, pytz.timezone(tz)
                            ),
                            display_format="DD-MM-YYYY",
                            calendar_orientation="vertical",
                            start_date=datetime.datetime.fromtimestamp(
                                selected_time / 1000, pytz.timezone(tz)
                            ),
                            end_date=datetime.datetime.now(pytz.timezone(tz)),
                            first_day_of_week=1,
                        ),
                    ],
                    className="controls",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="map", className="plot"),
                    ],
                    className="plot-container",
                ),
                html.Div(
                    [
                        html.H3("Eventos por calle"),
                        dash_table.DataTable(
                            id="table",
                            columns=[
                                {"name": i, "id": i} for i in ["Calle", "Eventos"]
                            ],
                            data=[],
                            filter_action="native",
                            sort_action="native",
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "center"},
                            style_data_conditional=[
                                {
                                    "if": {"column_id": "Eventos"},
                                    "width": "100px",
                                    "textAlign": "center",
                                },
                                {
                                    "if": {"column_id": "Calle"},
                                    "width": "350px",
                                    "textAlign": "left",
                                },
                            ],
                            style_header={
                                "backgroundColor": "rgba(30,30,30,0.6)",
                                "color": "#ccc",
                                "fontWeight": "semibold",
                                "fontFamily": "Verdana",
                                "fontSize": "1rem",
                                "textAlign": "center",
                                "border": "1px solid #555",
                            },
                            style_data={
                                "backgroundColor": "rgba(30,30,30,0.6)",
                                "color": "#fff",
                                "fontWeight": "lighter",
                                "fontFamily": "Verdana",
                                "fontSize": "1rem",
                                "textAlign": "center",
                                "border": "1px solid #555",
                            },
                            style_filter={
                                "backgroundColor": "rgba(30,30,30,0.6)",
                                "color": "#000",
                                "fontWeight": "semibold",
                                "fontFamily": "Verdana",
                                "fontSize": "1rem",
                                "textAlign": "center",
                                "border": "1px solid #555",
                            },
                        ),
                    ],
                    className="plot-container",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="hourly",
                            className="plot",
                        )
                    ],
                    className="plot-container",
                ),
                html.Div(
                    [html.H3("Otro gráfico"), dcc.Graph(id="ML3", className="plot")],
                    className="plot-container",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Parámetros para el modelo predictivo"),
                                html.H3("Tipo de evento"),
                                dcc.Dropdown(
                                    id="dd_type_ml",
                                    options=[
                                        {"label": "Accidentes", "value": "ACCIDENT"},
                                        {"label": "Congestión", "value": "JAM"},
                                        {"label": "Peligros", "value": "HAZARD"},
                                        {
                                            "label": "Caminos cerrados",
                                            "value": "ROAD_CLOSED",
                                        },
                                    ],
                                    value="ACCIDENT",
                                    clearable=False,
                                ),
                                html.H3("Hora y fecha"),
                                dcc.Dropdown(
                                    id="dd_hour_ml",
                                    options=[
                                        {"label": "00:00 - 00:59", "value": 0},
                                        {"label": "01:00 - 01:59", "value": 1},
                                        {"label": "02:00 - 02:59", "value": 2},
                                        {"label": "03:00 - 03:59", "value": 3},
                                        {"label": "04:00 - 04:59", "value": 4},
                                        {"label": "05:00 - 05:59", "value": 5},
                                        {"label": "06:00 - 06:59", "value": 6},
                                        {"label": "07:00 - 07:59", "value": 7},
                                        {"label": "08:00 - 08:59", "value": 8},
                                        {"label": "09:00 - 09:59", "value": 9},
                                        {"label": "10:00 - 10:59", "value": 10},
                                        {"label": "11:00 - 11:59", "value": 11},
                                        {"label": "12:00 - 12:59", "value": 12},
                                        {"label": "13:00 - 13:59", "value": 13},
                                        {"label": "14:00 - 14:59", "value": 14},
                                        {"label": "15:00 - 15:59", "value": 15},
                                        {"label": "16:00 - 16:59", "value": 16},
                                        {"label": "17:00 - 17:59", "value": 17},
                                        {"label": "18:00 - 18:59", "value": 18},
                                        {"label": "19:00 - 19:59", "value": 19},
                                        {"label": "20:00 - 20:59", "value": 20},
                                        {"label": "21:00 - 21:59", "value": 21},
                                        {"label": "22:00 - 22:59", "value": 22},
                                        {"label": "23:00 - 23:59", "value": 23},
                                    ],
                                    value=7,
                                    clearable=False,
                                ),
                                dcc.DatePickerSingle(
                                    id="date_ml",
                                    min_date_allowed=datetime.date(2024, 10, 1),
                                    max_date_allowed=datetime.date(2025, 12, 31),
                                    initial_visible_month=datetime.date.today(),
                                    date=datetime.date.today(),
                                    display_format="DD/MM/YYYY",
                                    first_day_of_week=1,
                                ),
                                html.Div(
                                    [
                                        html.H3("Segmentos y probabilidades"),
                                        dash_table.DataTable(
                                            id="table_ml",
                                            columns=[
                                                {"name": i, "id": i}
                                                for i in ["Segmento", "Probabilidad"]
                                            ],
                                            data=[],
                                            filter_action="native",
                                            sort_action="native",
                                            page_action="native",
                                            page_current=0,
                                            page_size=10,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"textAlign": "center"},
                                            style_data_conditional=[
                                                {
                                                    "if": {"column_id": "Segmento"},
                                                    "width": "100px",
                                                    "textAlign": "center",
                                                },
                                                {
                                                    "if": {
                                                        "column_id": "Probabilidad evento"
                                                    },
                                                    "width": "350px",
                                                    "textAlign": "left",
                                                },
                                            ],
                                            style_header={
                                                "backgroundColor": "rgba(30,30,30,0.6)",
                                                "color": "#ccc",
                                                "fontWeight": "semibold",
                                                "fontFamily": "Verdana",
                                                "fontSize": "1rem",
                                                "textAlign": "center",
                                                "border": "1px solid #555",
                                            },
                                            style_data={
                                                "backgroundColor": "rgba(30,30,30,0.6)",
                                                "color": "#fff",
                                                "fontWeight": "lighter",
                                                "fontFamily": "Verdana",
                                                "fontSize": "1rem",
                                                "textAlign": "center",
                                                "border": "1px solid #555",
                                            },
                                            style_filter={
                                                "backgroundColor": "rgba(30,30,30,0.6)",
                                                "color": "#000",
                                                "fontWeight": "semibold",
                                                "fontFamily": "Verdana",
                                                "fontSize": "1rem",
                                                "textAlign": "center",
                                                "border": "1px solid #555",
                                            },
                                        ),
                                    ],
                                    className="plot-container",
                                ),
                            ],
                            className="ml-zone",
                        ),
                    ],
                    className="plot-container",
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="ML",
                            className="ml-map",
                            config={
                                "scrollZoom": True  # Habilita el zoom de desplazamiento en el mapa
                            },
                        )
                    ],
                    className="plot-container ml-zone",
                ),
            ],
            className="row",
        ),
    ],
    className="container",
)


def update_ML(
    hour: int,
    day_type: int,
    week_day: int,
    kind: str,
    higlighted_segment: int | None = None,
):
    g = Grouper(alerts)
    g.group((10, 20))

    x_var = pd.DataFrame(
        {
            "day_type": [day_type],
            "hour": [hour],
            "week_day": [week_day],
            "group": [0],
            "type_ACCIDENT": [1 if "ACCIDENT" in kind else 0],
            "type_JAM": [1 if "JAM" in kind else 0],
            "type_HAZARD": [1 if "HAZARD" in kind else 0],
            "type_ROAD_CLOSED": [1 if "ROAD_CLOSED" in kind else 0],
        },
        columns=model.feature_names_in_,
    )

    if kind == "all":
        x_var["type_ACCIDENT"] = 1
        kind = "ACCIDENT"

    # Convierte las coordenadas en una lista de puntos (x, y)
    points = [
        Point(x, y)
        for x_row, y_row in zip(g.x_grid, g.y_grid)
        for x, y in zip(x_row, y_row)
    ]

    # Crear un GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3857")
    # Convierte a EPSG:4326 (lat/lon)
    gdf = gdf.to_crs(epsg=4326)

    # Extrae las coordenadas convertidas para Plotly
    lats = sorted(list(set(gdf.geometry.y.tolist())))
    lons = sorted(list(set(gdf.geometry.x.tolist())))

    polygons = []

    for j in range(len(lons) - 1):
        for i in range(len(lats) - 1):
            poly = Polygon(
                [
                    (lons[j], lats[i]),
                    (lons[j + 1], lats[i]),
                    (lons[j + 1], lats[i + 1]),
                    (lons[j], lats[i + 1]),
                ]
            )
            polygons.append(poly)

    # Convertir a GeoDataFrame
    gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # Añadir datos de ejemplo al GeoDataFrame, como probabilidades
    gdf_polygons["probability"] = [
        model.predict_proba(x_var.assign(group=g.calc_quadrant(j, i)))[0][1]
        if g.calc_quadrant(j, i) in set(g.data["group"])
        else 0
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
            hovertemplate="<b>%{text}</b><br>Probabilidad de evento: %{z:.2f}<extra></extra>",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            marker_opacity=0.2,
            marker_line_width=1,
        )
    )

    if higlighted_segment:
        # Filtra el polígono seleccionado
        selected_poly = gdf_polygons[
            gdf_polygons["segment"] == f"Segmento {higlighted_segment}"
        ]

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
                fillcolor="rgba(255, 0, 0, 0.3)",  # Relleno rojo con opacidad
                line=dict(color="red", width=2),  # Borde rojo más grueso
                hoverinfo="skip",  # Evita que interfiera con el hover de Choropleth
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

    gdf_polygons["segment"] = gdf_polygons["segment"].str.replace(
        "Segmento ", "", regex=False
    )
    gdf_polygons["probability"] = gdf_polygons["probability"].round(2)

    table_data = (
        gdf_polygons[["segment", "probability"]]
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
    ],
    [
        Input("dd_type", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("table", "active_cell"),
    ],
)
def update_graphs(kind, start_date, end_date, active_cell):
    if start_date is None or end_date is None:
        return go.Figure(), go.Figure(), {}, None

    start = (
        (datetime.datetime.fromisoformat(start_date))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(tz=pytz.timezone(tz))
    )
    end = (
        (datetime.datetime.fromisoformat(end_date))
        .replace(hour=23, minute=59, second=59, microsecond=0)
        .astimezone(tz=pytz.timezone(tz))
    )

    if end <= start:
        end = start + (23 * 60 * 60 + 60**2 * 59) * 1000

    filtered_alerts = alerts[
        (alerts["pubMillis"] >= start) & (alerts["pubMillis"] <= end)
    ]
    streets_data = filtered_alerts.groupby("street")["type"].count().reset_index()
    streets_data = streets_data.rename(columns={"street": "Calle", "type": "Eventos"})
    streets_data = streets_data.sort_values(by="Eventos", ascending=False)

    table_data = streets_data.to_dict("records")

    if active_cell is not None:
        filtered_alerts = filtered_alerts[
            filtered_alerts["street"] == streets_data.iloc[active_cell["row"]]["Calle"]
        ]

    extra_cols = ["day_type", "week_day", "day", "hour", "minute"]
    if kind == "all":
        events = utils.extract_event(
            filtered_alerts,
            ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"],
            extra_cols,
        )
    else:
        events = utils.extract_event(filtered_alerts, [kind], extra_cols)

    hourly = utils.hourly_group(events).reset_index().copy()
    hourly = pd.melt(
        hourly,
        id_vars=["hour"],
        value_vars=["s", "f"],
        var_name="day_type",
        value_name="events",
    )

    if kind != "all":
        map_data = filtered_alerts[filtered_alerts["type"] == kind].copy()
    else:
        map_data = filtered_alerts.copy()
    map_data = utils.freq_nearby(map_data, nearby_meters=200)
    map_data["freq"] = map_data.apply(
        lambda x: x["freq"] if x["freq"] > 0 else 1, axis=1
    )

    map_data["time"] = map_data["pubMillis"].apply(lambda x: x.strftime("%H:%M:%S"))
    map_data["date"] = map_data["pubMillis"].apply(lambda x: x.strftime("%d-%m-%Y"))

    map_data["type"] = map_data["type"].map(names)

    map_fig = go.Figure(
        px.scatter_map(
            map_data,
            lat="y",
            lon="x",
            color="type",
            hover_name="uuid",
            size="freq",
            zoom=10.5,
            labels={
                "type": "Tipo de evento",
                "freq": "Frecuencia",
                "x": "Longitud",
                "y": "Latitud",
                "street": "Calle",
                "time": "Hora de reporte",
                "date": "Fecha de reporte",
            },
            hover_data={
                "street": True,
                "time": True,
                "date": True,
            },
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
            ),
        )

    hourly_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        title=dict(
            text=f"Promedio de {names[kind].lower()} por hora del día",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=20,
            ),
            pad=dict(b=30, l=50),
        ),
        grid_ygap=1,
        # nogrid
        yaxis=dict(
            title=f"Promedio de {names[kind].lower()} reportados", showgrid=False
        ),
        xaxis=dict(title="Hora del día", showgrid=False),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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

    return map_fig, hourly_fig, table_data


@app.callback(
    [Output("ML", "figure"), Output("table_ml", "data")],
    [
        Input("dd_type_ml", "value"),
        Input("date_ml", "date"),
        Input("dd_hour_ml", "value"),
        Input("table_ml", "active_cell"),
        Input("table_ml", "data"),
    ],
)
def update_ml_graphs(kind, date_value, hour, active_cell, table_data):
    if kind is None:
        kind = "ACCIDENT"
    if date_value is None:
        date_ml = datetime.date.today()
    else:
        date_ml = datetime.date.fromisoformat(date_value)

    week_day = date_ml.weekday()
    day_type = (
        0 if (date_ml.weekday() >= 5) | (date_value in utils.get_holidays()) else 1
    )

    segment = None
    if table_data and active_cell:
        segment = table_data[active_cell["row"]]["Segmento"]

    group_fig, table_data = update_ML(hour, day_type, week_day, kind, segment)

    return group_fig, table_data
