import copy
import datetime

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from dash import Dash, Input, Output, dash_table, dcc, html
from shapely.geometry import Point, Polygon

import mlflow
from analytics.grouper import Grouper
from analytics.ml import ML, init_mlflow
from train import train
from utils import utils

init_mlflow()

TZ = "America/Santiago"

selected_time = int(datetime.datetime.now().timestamp()) * 1000
since = selected_time
alerts = None

def update_data():
    global selected_time, since, alerts
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

    selected_time = int(
        (
            datetime.datetime.now(pytz.timezone(TZ)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            - datetime.timedelta(days=30)
        ).timestamp()
        * 1000
    )

    try:
        alerts_query = utils.load_data("alerts", mode="since", epoch=since)
        alerts = Grouper(alerts_query.to_gdf(tz=TZ))
        alerts.group((10, 20)).filter_by_group_time(60, True)
    except Exception as e:
        print("Something was wrong while updating alerts")
        print(f"Error: {e}")
update_data()

model = None
last_model = 0

def load_model():
    global model, last_model
    MODEL_NAME="XGBClassifier"
    last_model = ML.get_last_model(MODEL_NAME)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{last_model}")
    print(f"Model {MODEL_NAME} version {last_model} successfully loaded")

# Train and Load the model dinamically
scheduler = BackgroundScheduler()
scheduler.add_job(update_data, 'interval', minutes=5)
scheduler.add_job(train, 'interval', days=30)
scheduler.add_job(load_model, 'interval', days=30, minutes=5)
scheduler.start()

load_model()


names = {
    "all": "Evento",
    "ACCIDENT": "Accidente",
    "JAM": "Congestión",
    "HAZARD": "Peligro",
    "ROAD_CLOSED": "Camino cerrado",
}



app = Dash(__name__, update_title=None, meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no",
        },
        {
            "name": "description",
            "content": (
                "Dashboard interactivo para el análisis del tráfico vehicular en Antofagasta. "
                "Predicciones basadas en machine learning con datos de la API de Waze."
            ),
        },
        {
            "name": "keywords",
            "content": (
                "tráfico vehicular, Antofagasta, análisis de tráfico, Dash, machine learning, "
                "predicción de tráfico, rutas seguras, API Waze, eventos de tráfico, accidentes"
            ),
        },
        {
            "name": "author",
            "content": "Richard Peña B.",
        },
        {
            "property": "og:title",
            "content": "Gestión del Tráfico Vehicular en Antofagasta - Dashboard",
        },
        {
            "property": "og:description",
            "content": (
                "Explora patrones de tráfico y predicciones de rutas seguras con un dashboard"
                "basado en datos de la API de Waze y técnicas de machine learning."
            ),
        },
    ],)
app._favicon = ("favicon.png")
app.title = "Gestión del tráfico en Antofagasta"
server = app.server

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
                html.Button("Instrucciones", "toggle_instructions"),
                html.Div(
                    id="instructions",
                    children=[
                        html.H3("Instrucciones:"),
                        html.Ul(
                            [
                                html.Li(
                                    "El dashboard se divide en dos secciones principales, la primera corresponde a representaciones descriptivas de los datos, la cual se compone de los 6 primeros gráficos/tablas, siendo el de correlación el último. La segunda sección corresponde al modelo predictivo, compuesto por el mapa, las entradas de parámetros y la tabla de información."
                                ),
                                html.Li(
                                    "La primera sección es controlada por los filtros de tipo de evento y rango, se exceptúa en el filtro de tipo el gráfico de correlación, para el caso de los últimos eventos solo se toma en cuenta el filtro de tipo."
                                ),
                                html.Li(
                                    "Los gráficos son interactivos, se pueden filtrar por calle, haciendo click en la tabla de calles, como también en las leyendas"
                                ),
                                html.Li(
                                    "El modelo predictivo está basado en un modelo de Machine Learning y segmentado por parámetros temporales (tipo de día, día de la semana, día del mes, hora del día) y parámetros geospaciales, los cuales se resumen en el segmento asociado."
                                ),
                                html.Li(
                                    "La tabla del modelo predictivo también es interactiva, se puede visualizar el segmento asociado haciendo click en la fila correspondiente."
                                ),
                                html.Li(
                                    "Los datos están disponibles a partir del 1 de octubre de 2024, por lo que se pueden aplicar filtros de fecha hasta ese punto."
                                ),
                                html.Li(
                                    "El objetivo del modelo predictivo es valorar la probabilidad de que un evento ocurra bajo los parámetros previamente establecidos, tiene aplicaciones como la de visualizar la ruta con menor probabilidad de evento desde un segmento A a un segmento B, también para poder identificar patrones para la gestión del tráfico."
                                ),
                            ],
                        ),
                    ],
                    className="instructions",
                ),
                html.Div(
                    children=[
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
                                since / 1000, pytz.timezone(TZ)
                            ),
                            max_date_allowed=datetime.datetime.now(pytz.timezone(TZ)),
                            initial_visible_month=datetime.datetime.fromtimestamp(
                                selected_time / 1000, pytz.timezone(TZ)
                            ),
                            display_format="DD-MM-YYYY",
                            calendar_orientation="vertical",
                            start_date=datetime.datetime.fromtimestamp(
                                selected_time / 1000, pytz.timezone(TZ)
                            ),
                            end_date=datetime.datetime.now(pytz.timezone(TZ)),
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
                    id="map-zone",
                    children=[
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
                            style_cell={"textAlign": "center", "height": "auto", "whiteSpace": "normal"},
                            style_data_conditional=[
                                {
                                    "if": {"column_id": "Eventos"},
                                    "width": "27%",
                                    "textAlign": "center",
                                },
                                {
                                    "if": {"column_id": "Calle"},
                                    "width": "73%",
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
                        html.Button("Limpiar selección", id="table_clear"),
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
                    [
                        dcc.Graph(
                            id="daily",
                            className="plot",
                        )
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
                        html.H3("Últimos eventos reportados"),
                        dash_table.DataTable(
                            id="table_last",
                            columns=[
                                {"name": i, "id": i}
                                for i in ["Tipo", "Fecha", "Hora", "Calle", "Segmento"]
                            ],
                            data=[],
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            cell_selectable=False,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "center",
                                "whiteSpace": "normal",
                                "height": "auto",
                                "wordBreak": "break-all"
                            },
                            style_data_conditional=[ 
                            {
                                "if": {"column_id": "Tipo"},
                                "width": "20%"
                            },
                            {
                                "if": {"column_id": "Fecha"},
                                "width": "20%"
                            },
                            {
                                "if": {"column_id": "Hora"},
                                "width": "10%"
                            },
                            {
                                    "if": {"column_id": "Calle"},
                                    "width": "40%"
                            },
                            {
                                    "if": {"column_id": "Segmento"},
                                    "width": "10%"
                            }
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
                        ),
                    ],
                    className="plot-container",
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="scatter",
                            className="plot",
                        )
                    ],
                    className="plot-container",
                ),
            ],
            className="row",
        ),
        html.H1("Modelo predictivo", className="title_ml"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div([
                                    html.H2("Parámetros para el modelo"),
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
                                ],
                                className="parameters_ml",
                                ),
                                html.Div(
                                    [
                                        html.H3("Segmentos y probabilidades"),
                                        html.Button("Limpiar selección", id="table_ml_clear"),
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
                                            style_cell={"textAlign": "center", "height": "auto", "whiteSpace": "normal"},
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
                                        html.P(f"Versión del modelo: {last_model}", id="model_version")
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
                                "scrollZoom": True  
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
    day: int,
    kind: str,
    higlighted_segment: int | None = None,
):
    if model is None:
        return go.Figure()
    g = alerts

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
        columns=model.feature_names_in_,
    )

    if kind == "all":
        x_var["type_ACCIDENT"] = 1
        kind = "ACCIDENT"

    assert g is not None, "Error: g must not be None"
    assert g.x_grid is not None, "Error: x_grid must not be None"
    assert g.y_grid is not None, "Error: y_grid must not be None"
    # Convierte las coordenadas en una lista de puntos (x, y)
    points = [
        Point(x, y)
        for x_row, y_row in zip(g.x_grid, g.y_grid)
        for x, y in zip(x_row, y_row)
    ]

    if points:
        # Crear un GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3857")
        # Convierte a EPSG:4326 (lat/lon)
        gdf = gdf.to_crs(epsg=4326)

        if gdf is None:
            raise ValueError("Something was wrong, the geometry is null, verify the points")
        # Extrae las coordenadas convertidas para Plotly
        lats = sorted(list(set(gdf.geometry.y.tolist())))
        lons = sorted(list(set(gdf.geometry.x.tolist())))

    else:
        lats = []
        lons = []

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
    gdf_polygons:gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
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

    gdf_polygons["segment"] = gdf_polygons["segment"].str.replace(
        "Segmento ", "", regex=False
    )
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
    ],
    [
        Input("dd_type", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("table", "active_cell"),
    ],
)
def update_graphs(kind, start_date, end_date, active_cell):
    STANDARD_RETURN = go.Figure(), go.Figure(), {}, None
    if start_date is None or end_date is None:
        return STANDARD_RETURN

    extra_cols = ["day_type", "week_day", "day", "hour", "minute"]

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

    filtered_alerts = copy.deepcopy(alerts)

    if filtered_alerts is None or filtered_alerts.data.shape[0] == 0:
        return STANDARD_RETURN

    filtered_alerts.data = filtered_alerts.data[(filtered_alerts.data["pubMillis"] >= start) & (filtered_alerts.data["pubMillis"] <= end)]
    


    scatter_data_acc = utils.extract_event(
        filtered_alerts.data,
        ["ACCIDENT"],
        extra_col=extra_cols,
    )

    scatter_data_acc = utils.hourly_group(scatter_data_acc, sum=True)

    scatter_data_jam = utils.extract_event(
        filtered_alerts.data,
        ["JAM"],
        extra_col=extra_cols,
    )

    scatter_data_jam = utils.hourly_group(scatter_data_jam, sum=True)

    if kind is not None and kind != "all":
        streets_data = (
            filtered_alerts.data[filtered_alerts.data["type"] == kind]
            .groupby("street")["type"]
            .count()
            .reset_index()
        )
    else:
        streets_data = filtered_alerts.data.groupby("street")["type"].count().reset_index()

    streets_data = streets_data.rename(columns={"street": "Calle", "type": "Eventos"})
    streets_data = streets_data.sort_values(by="Eventos", ascending=False)

    table_data = streets_data.to_dict("records")

    if active_cell is not None:
        filtered_alerts = filtered_alerts.data.loc[
            filtered_alerts.data["street"] == streets_data.iloc[active_cell["row"]]["Calle"]
        ]


    if kind == "all":
        events = utils.extract_event(
            filtered_alerts.data,
            ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"],
            extra_cols,
        )
    else:
        events = utils.extract_event(filtered_alerts.data, [kind], extra_cols)

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

    if kind != "all":
        map_data = copy.deepcopy(filtered_alerts)
        map_data = map_data.data[map_data.data["type"] == kind]
    else:
        map_data = copy.deepcopy(filtered_alerts)
    map_data.data = utils.freq_nearby(map_data.data, nearby_meters=200)
    if map_data.data is None:
        raise ValueError("Map data is None")
    map_data.data["freq"] = map_data.data.apply(
        lambda x: x["freq"] if x["freq"] > 0 else 1, axis=1
    )

    map_data.data["time"] = map_data.data.pubMillis.apply(lambda x: x.strftime("%H:%M:%S"))
    map_data.data["date"] = map_data.data.pubMillis.apply(lambda x: x.strftime("%d-%m-%Y"))

    map_data.data["type"] = map_data.data["type"].map(names)

    map_fig = go.Figure(
        px.scatter_map(
            map_data.data,
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
            x=0.5
        ),
        grid_ygap=1,
        # nogrid
        yaxis=dict(
            title=f"Promedio de {names[kind].lower()} reportados", showgrid=False
        ),
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
                marker_color={"f": "rgb(210, 41, 252)", "s": "rgb(34, 243, 173)"}[
                    day_type
                ],
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
            x=0.5
        ),
        grid_ygap=1,
        # nogrid
        yaxis=dict(
            title=f"Promedio de {names[kind].lower()} reportados", showgrid=False
        ),
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
                line=dict(
                    width=1, color="DarkSlateGrey"
                ),  # Borde alrededor de las burbujas
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
                line=dict(
                    width=1, color="DarkSlateGrey"
                ),  # Borde alrededor de las burbujas
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

    return map_fig, hourly_fig, table_data, daily_fig, scatter_fig


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
def update_ml_graphs(
    kind, date_value, hour, active_cell, table_data, page_current, page_size
):
    if kind is None:
        kind = "ACCIDENT"
    if date_value is None:
        date_ml = datetime.date.today()
    else:
        date_ml = datetime.date.fromisoformat(date_value)

    day = date_ml.day
    week_day = date_ml.weekday()
    day_type = (
        0 if (date_ml.weekday() >= 5) | (date_value in utils.get_holidays()) else 1
    )

    segment = None
    if table_data and active_cell:
        global_index = page_current * page_size + active_cell["row"]
        segment = table_data[global_index]["Segmento"]

    group_fig, table_data = update_ML(hour, day_type, week_day, day, kind, segment)

    return group_fig, table_data


@app.callback(
    [
        Output("table_last", "data"),
        Output("date_range", "max_date_allowed")
    ],
    Input("dd_type", "value"),
)
def update_last_events(kind):
    last_events = alerts.data.sort_values(by="pubMillis", ascending=False)

    if last_events.shape[0] == 0:
        return [], datetime.datetime.now(pytz.timezone(TZ))

    concepts = [n for n in names.keys() if n != "all"]
    if kind != "all" and kind is not None:
        last_events = last_events[last_events["type"] == kind]
        concepts = [kind]

    last_events = last_events.iloc[:20]

    if last_events.shape[0] == 0:
        return [], datetime.datetime.now(pytz.timezone(TZ))

    last_events = utils.extract_event(
        last_events, concepts, ["type", "group", "hour", "minute", "street"]
    )
    last_events["hour"] = last_events.apply(
        lambda row: f"{int(row['hour']):02}:{int(row['minute']):02}", axis=1
    )
    last_events["date"] = last_events.inicio.dt.strftime("%d/%m/%Y")

    last_events["type"] = last_events.type.map(names)

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
@app.callback(
        Output("table_ml", "active_cell"),
        Input("table_ml_clear", "n_clicks")
        )
def clear_ml(_):
    return None

@app.callback(
        Output("table", "active_cell"),
        Input("table_clear", "n_clicks")
        )
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

        Output('instructions', 'className'),
        Input('toggle_instructions', 'n_clicks')
        )
