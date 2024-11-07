from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
from utils import utils
import datetime
import pytz
import plotly.graph_objs as go
import pandas as pd


tz = "America/Santiago"

app = Dash(__name__)

since = int(
    (
        datetime.datetime.now(pytz.timezone(tz)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        - datetime.timedelta(days=30)
    ).timestamp()
    * 1000
)

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
                                since / 1000, pytz.timezone(tz)
                            ),
                            display_format="DD-MM-YYYY",
                            calendar_orientation="vertical",
                            start_date=datetime.datetime.fromtimestamp(
                                since / 1000, pytz.timezone(tz)
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
                        dcc.Graph(
                            id="hourly",
                            className="plot",
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
                        )
                    ],
                    className="plot-container",
                ),
            ],
            className="row",
        ),
    ],
    className="container",
)


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
        return go.Figure(), go.Figure()

    start = int((datetime.datetime.fromisoformat(start_date)).timestamp() * 1000)
    end = int((datetime.datetime.fromisoformat(end_date)).timestamp() * 1000)

    alerts = utils.load_data("alerts", mode="between", between=(start, end))
    alerts = alerts.to_gdf(tz=tz)
    alerts = utils.freq_nearby(alerts, nearby_meters=200)
    alerts["freq"] = alerts.apply(lambda x: x["freq"] if x["freq"] > 0 else 1, axis=1)
    alerts = utils.filter_nearby(alerts, threshold=200)

    streets_data = alerts.groupby("street")["type"].count().reset_index()
    streets_data["Calle"] = streets_data["street"]
    streets_data["Eventos"] = streets_data["type"]
    streets_data = streets_data.sort_values(by="Eventos", ascending=False)

    table_data = streets_data.to_dict("records")

    if active_cell is not None:
        alerts = alerts[
            alerts["street"] == streets_data.iloc[active_cell["row"]]["street"]
        ]

    if kind == "all":
        events = utils.extract_event(
            alerts, ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]
        )
    else:
        events = utils.extract_event(alerts, [kind])
    hourly = utils.hourly_group(events).reset_index().copy()
    hourly = pd.melt(
        hourly,
        id_vars=["hour"],
        value_vars=["s", "f"],
        var_name="day_type",
        value_name="events",
    )

    names = {
        "all": "Evento",
        "ACCIDENT": "Accidente",
        "JAM": "Congestión",
        "HAZARD": "Peligro",
        "ROAD_CLOSED": "Camino cerrado",
    }

    if kind != "all":
        map_data = alerts[alerts["type"] == kind].copy()
    else:
        map_data = alerts

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
